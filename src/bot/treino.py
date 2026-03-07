import time

from src.bot.controle_fps import ControleFPS, obter_fps_alvo
from src.bot.captura import capturar_tela
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.luta import Luta

TITULO_JOGO = "Champions"


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


def iniciar_treino(monitor=None):
  if monitor is None:
    monitor = obter_monitor_jogo()

  fps_alvo = obter_fps_alvo()
  controle_fps = ControleFPS(fps_alvo)
  print(f"Treino iniciado com FPS alvo: {fps_alvo}")

  luta = Luta(exibir_debug=True)

  while True:
    inicio_ciclo = controle_fps.iniciar_ciclo()

    try:
      frame = capturar_tela(monitor)
      luta.processar_frame(frame)
    finally:
      controle_fps.finalizar_ciclo(inicio_ciclo)
