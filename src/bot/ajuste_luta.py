import os
import time
import cv2

from src.bot.acoes import clicar
from src.bot.controle_fps import ControleFPS, obter_fps_alvo
from src.bot.captura import CapturaAssincrona
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.luta import Luta
from src.bot.rois import ROI_JOGAR_NOVAMENTE, obter_roi
from src.bot.visao import encontrar_templates_em_paralelo
from src.utils.log import log
from src.utils.visao_debug import atualizar_metricas

TITULO_JOGO = "Champions"
BOTOES_AJUSTE = [
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
    time.sleep(0.5)
  else:
    log("Nao foi possivel focar automaticamente a janela.")

  return obter_bbox_janela(janela)


def processar_layout_ajuste(frame, monitor, roi_jogar_novamente):
  consultas = []
  for nome, caminho_template in BOTOES_AJUSTE:
    if nome == "jogar_novamente":
      consultas.append((nome, caminho_template, 0.8, roi_jogar_novamente))
    else:
      consultas.append((nome, caminho_template))

  resultados = encontrar_templates_em_paralelo(
    frame,
    consultas,
  )

  for nome_botao, _ in BOTOES_AJUSTE:
    botao = resultados.get(nome_botao)
    if not botao:
      continue

    log(f"Botao ({nome_botao}) encontrado no ajuste. Focando a janela do jogo...")
    focar_jogo(TITULO_JOGO)
    time.sleep(0.2)
    clicar(
      monitor["left"] + botao["x"],
      monitor["top"] + botao["y"],
    )
    time.sleep(2)
    return True

  return False


def iniciar_ajuste_luta(monitor=None):
  if monitor is None:
    monitor = obter_monitor_jogo()

  fps_alvo = obter_fps_alvo()
  controle_fps = ControleFPS(fps_alvo)
  log(f"Modo de Ajuste de Luta iniciado com FPS alvo: {fps_alvo}")

  exibir_debug = os.getenv("BOT_DEBUG", "1").strip() == "1"
  roi_jogar_novamente = obter_roi(ROI_JOGAR_NOVAMENTE)
  log(f"ROI jogar_novamente (x1,y1,x2,y2): {roi_jogar_novamente}")
  
  luta = Luta(exibir_debug=exibir_debug)
  capturador = CapturaAssincrona(monitor)
  capturador.iniciar()
  modo_reinicio = False

  try:
    while True:
      inicio_ciclo = controle_fps.iniciar_ciclo()

      try:
        frame = capturador.obter_frame(timeout=0.05)
        if frame is None:
          continue

        if modo_reinicio:
          cv2.waitKey(1)  # Mantém as janelas do OpenCV responsivas
          if processar_layout_ajuste(frame, monitor, roi_jogar_novamente):
            modo_reinicio = False
          continue

        status_luta = luta.processar_frame(frame)
        if status_luta and status_luta.get("terminal") is not None:
          modo_reinicio = True
          log(
            f"Entrando em modo reinicio apos terminal={status_luta.get('terminal')}. "
            "Pausado ate detectar jogar novamente."
          )
      finally:
        controle_fps.finalizar_ciclo(inicio_ciclo)
        metricas = controle_fps.obter_metricas()
        atualizar_metricas(metricas)
  finally:
    capturador.parar()
