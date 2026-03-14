import os
import time

from src.bot.acoes import clicar
from src.bot.controle_fps import ControleFPS, obter_fps_alvo
from src.bot.captura import CapturaAssincrona
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.luta import Luta
from src.bot.visao import encontrar_templates_em_paralelo
from src.utils.log import log
from src.utils.visao_debug import atualizar_metricas

TITULO_JOGO = "Champions"
BOTOES_TREINO = [
  ("jogar_novamente", "assets/jogar-novamente-button.png"),
]


def obter_monitor_jogo():
  log("Procurando janela do jogo...")
  janela = None

  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(1)

  log("Janela encontrada!")

  if focar_jogo(TITULO_JOGO):
    log("Janela do jogo focada automaticamente.")
    time.sleep(0.5)  # ← MUITO IMPORTANTE
  else:
    log("Nao foi possivel focar automaticamente a janela.")

  return obter_bbox_janela(janela)


def processar_layout_treino(frame, monitor):
  resultados = encontrar_templates_em_paralelo(
    frame,
    [(nome, caminho_template) for nome, caminho_template in BOTOES_TREINO],
  )

  for nome_botao, _ in BOTOES_TREINO:
    botao = resultados.get(nome_botao)
    if not botao:
      continue

    log(f"Botao ({nome_botao}) encontrado no treino.")
    clicar(
      monitor["left"] + botao["x"],
      monitor["top"] + botao["y"],
    )
    time.sleep(2)
    return True

  return False


def iniciar_treino(monitor=None):
  if monitor is None:
    monitor = obter_monitor_jogo()

  fps_alvo = obter_fps_alvo()
  controle_fps = ControleFPS(fps_alvo)
  log(f"Treino iniciado com FPS alvo: {fps_alvo}")

  exibir_debug = os.getenv("BOT_DEBUG", "1").strip() == "1"
  luta = Luta(exibir_debug=exibir_debug)
  capturador = CapturaAssincrona(monitor)
  capturador.iniciar()

  try:
    while True:
      inicio_ciclo = controle_fps.iniciar_ciclo()

      try:
        frame = capturador.obter_frame(timeout=0.05)
        if frame is None:
          continue

        if processar_layout_treino(frame, monitor):
          continue

        luta.processar_frame(frame)
      finally:
        controle_fps.finalizar_ciclo(inicio_ciclo)
        atualizar_metricas(controle_fps.obter_metricas())
  finally:
    capturador.parar()
