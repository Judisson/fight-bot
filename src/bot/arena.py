import time

from src.bot.acoes import clicar
from src.bot.controle_fps import ControleFPS, obter_fps_alvo
from src.bot.captura import capturar_tela
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.luta import Luta
from src.bot.visao import encontrar_template

TITULO_JOGO = "Champions"

BOTOES_ARENA = [
  ("selecao_rapida", "assets/selecao-rapida-button.png"),
  ("encontrar_partida", "assets/encontrar-partida-button.png"),
  ("continuar", "assets/continuar-button.png"),
  ("aceitar", "assets/aceitar-button.png"),
  ("proxima_serie", "assets/proxima-serie-button.png"),
  ("jogar_novamente", "assets/jogar-novamente-button.png"),
]


def obter_monitor_jogo():
  print("Procurando janela do jogo...")
  janela = None

  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(1)

  print("Janela encontrada!")

  if focar_jogo(TITULO_JOGO):
    print("Janela do jogo focada automaticamente.")
    time.sleep(0.5)  # ← MUITO IMPORTANTE
  else:
    print("Nao foi possivel focar automaticamente a janela.")

  return obter_bbox_janela(janela)


def processar_layout_arena(frame, monitor):
  for nome_botao, caminho_template in BOTOES_ARENA:
    botao = encontrar_template(frame, caminho_template)
    if not botao:
      continue

    print(f"Botao ({nome_botao}) encontrado!")
    clicar(
      monitor["left"] + botao["x"],
      monitor["top"] + botao["y"],
    )
    time.sleep(2)
    return True

  return False


def iniciar_arena(monitor=None):
  if monitor is None:
    monitor = obter_monitor_jogo()

  fps_alvo = obter_fps_alvo()
  controle_fps = ControleFPS(fps_alvo)
  print(f"Arena iniciada com FPS alvo: {fps_alvo}")

  luta = Luta(exibir_debug=True)

  while True:
    inicio_ciclo = controle_fps.iniciar_ciclo()

    try:
      frame = capturar_tela(monitor)

      if processar_layout_arena(frame, monitor):
        continue

      luta.processar_frame(frame)
    finally:
      controle_fps.finalizar_ciclo(inicio_ciclo)
