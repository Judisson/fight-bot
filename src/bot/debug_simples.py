import os
import time

import cv2

from src.bot.captura import CapturaAssincrona
from src.bot.controle_fps import ControleFPS
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.utils.log import log
from src.visao.personagens import RastreadorPersonagens, desenhar_info_personagens

TITULO_JOGO = "Champions"
NOME_JANELA = "Debug Simples YOLO"


def _obter_fps_debug():
  valor = os.getenv("BOT_DEBUG_SIMPLE_FPS", "60").strip()
  try:
    fps = int(valor)
  except ValueError:
    fps = 60
  return max(30, min(fps, 240))


def _obter_downscale():
  valor = os.getenv("BOT_DEBUG_SIMPLE_DOWNSCALE", "2").strip()
  try:
    fator = int(valor)
  except ValueError:
    fator = 2
  return max(1, min(fator, 4))


def _obter_monitor_jogo():
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


def _desenhar_status(frame, metricas, info_yolo, info_personagens):
  backend = "GPU" if info_yolo["usa_gpu"] else "CPU"
  dispositivo = info_yolo["device"]
  async_txt = "ON" if info_yolo["assincrono"] else "OFF"
  deteccao = "OK" if info_personagens else "SEM_ALVO"

  linhas = [
    f"FPS: {metricas['fps_real']:.1f}/{metricas['fps_alvo']} | proc={metricas['proc_ms']:.2f}ms",
    f"YOLO: {backend} dev={dispositivo} async={async_txt} imgsz={info_yolo['imgsz']}",
    f"Deteccao: {deteccao}",
  ]

  for i, linha in enumerate(linhas):
    y = 30 + (i * 28)
    cv2.putText(
      frame,
      linha,
      (12, y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      (0, 255, 255),
      2,
      cv2.LINE_AA,
    )


def iniciar_debug_simples(monitor=None):
  if monitor is None:
    monitor = _obter_monitor_jogo()

  fps_alvo = _obter_fps_debug()
  downscale = _obter_downscale()
  controle_fps = ControleFPS(fps_alvo)
  capturador = CapturaAssincrona(monitor)
  capturador.iniciar()

  rastreador = RastreadorPersonagens(
    usar_yolo=True,
    caminho_modelo="src/modelos/personagens.pt",
    somente_yolo=True,
  )
  info_yolo = rastreador.obter_info_yolo()

  log(f"[DEBUG_SIMPLES] FPS alvo={fps_alvo} | downscale={downscale}")
  if info_yolo["modelo_carregado"]:
    backend = "GPU" if info_yolo["usa_gpu"] else "CPU"
    nome_gpu = info_yolo["nome_gpu"] or "N/A"
    log(
      "[DEBUG_SIMPLES] YOLO ativo | "
      f"backend={backend} | device={info_yolo['device']} | "
      f"gpu={nome_gpu} | async={info_yolo['assincrono']} | imgsz={info_yolo['imgsz']}"
    )
  else:
    log("[DEBUG_SIMPLES] YOLO nao carregou. Verifique ultralytics/modelo.")

  try:
    while True:
      inicio_ciclo = controle_fps.iniciar_ciclo()
      try:
        frame = capturador.obter_frame(timeout=0.05)
        if frame is None:
          continue

        if downscale > 1:
          frame_debug = frame[::downscale, ::downscale].copy()
        else:
          frame_debug = frame

        info_personagens = rastreador.detectar(frame_debug)
        if info_personagens:
          frame_debug = desenhar_info_personagens(frame_debug, info_personagens)

        metricas = controle_fps.obter_metricas()
        _desenhar_status(frame_debug, metricas, info_yolo, info_personagens)
        cv2.imshow(NOME_JANELA, frame_debug)

        if cv2.waitKey(1) == 27:
          log("[DEBUG_SIMPLES] Encerrado pelo usuario (ESC).")
          break
      finally:
        controle_fps.finalizar_ciclo(inicio_ciclo)
  finally:
    rastreador.encerrar()
    capturador.parar()
    cv2.destroyAllWindows()
