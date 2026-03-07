import time

import cv2

from src.bot.captura import capturar_tela
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.calibracao.config_vida import (
  carregar_configuracao_vida,
  definir_configuracao_vida,
  salvar_configuracao_vida,
)
from src.visao.barras_vida import obter_bboxes_vida

TITULO_JOGO = "Champions"
NOME_JANELA = "Calibracao de Vida"
JANELA_JOGADOR = "Calibracao - Barra Jogador"
JANELA_INIMIGO = "Calibracao - Barra Inimigo"
JANELA_MASCARA = "Calibracao - Mascara Cor"

_mouse = {"x": 0, "y": 0}


def _on_mouse(evento, x, y, _flags, _params):
  if evento == cv2.EVENT_MOUSEMOVE:
    _mouse["x"] = x
    _mouse["y"] = y


def _obter_monitor_jogo():
  print("Procurando janela do jogo...")
  janela = None

  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(0.5)

  print("Janela encontrada!")

  if focar_jogo(TITULO_JOGO):
    print("Janela focada automaticamente.")
  else:
    print("Nao foi possivel focar automaticamente a janela.")

  return obter_bbox_janela(janela)


def _clamp_config(config):
  config["y_inicio"] = max(0, config["y_inicio"])
  config["y_fim"] = max(config["y_inicio"] + 1, config["y_fim"])
  config["x1_jogador"] = max(0, config["x1_jogador"])
  config["x2_jogador"] = max(config["x1_jogador"] + 1, config["x2_jogador"])
  config["offset_direita_inimigo"] = max(0, config["offset_direita_inimigo"])
  config["largura_inimigo"] = max(1, config["largura_inimigo"])


def _mascara_cor_preview(roi):
  if roi.size == 0:
    return None

  altura, _, _ = roi.shape
  y1 = int(altura * 0.25)
  y2 = max(y1 + 1, int(altura * 0.75))
  faixa = roi[y1:y2, :]

  hsv = cv2.cvtColor(faixa, cv2.COLOR_BGR2HSV)
  mascara = cv2.inRange(hsv, (0, 0, 50), (179, 255, 255))
  return mascara


def iniciar_calibracao():
  monitor = _obter_monitor_jogo()
  config = carregar_configuracao_vida(forcar_reload=True)
  definir_configuracao_vida(config)

  print("Calibracao iniciada. Teclas:")
  print("Q/A y_inicio | W/S y_fim")
  print("E/D x1_jogador | R/F x2_jogador")
  print("T/G offset_direita_inimigo | Y/H largura_inimigo")
  print("P salva | ESC encerra")

  cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
  cv2.namedWindow(JANELA_JOGADOR, cv2.WINDOW_NORMAL)
  cv2.namedWindow(JANELA_INIMIGO, cv2.WINDOW_NORMAL)
  cv2.namedWindow(JANELA_MASCARA, cv2.WINDOW_NORMAL)
  cv2.setMouseCallback(NOME_JANELA, _on_mouse)

  while True:
    frame = capturar_tela(monitor)
    bbox_jogador, bbox_inimigo = obter_bboxes_vida(frame)

    x1, y1, x2, y2 = bbox_jogador
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    roi_jogador = frame[y1:y2, x1:x2]

    x1, y1, x2, y2 = bbox_inimigo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 120), 2)
    roi_inimigo = frame[y1:y2, x1:x2]

    mx = _mouse["x"]
    my = _mouse["y"]
    abs_x = monitor["left"] + mx
    abs_y = monitor["top"] + my

    cv2.line(frame, (mx, 0), (mx, frame.shape[0] - 1), (255, 255, 255), 1)
    cv2.line(frame, (0, my), (frame.shape[1] - 1, my), (255, 255, 255), 1)
    cv2.putText(
      frame,
      f"Local: ({mx}, {my})  Absoluto: ({abs_x}, {abs_y})",
      (20, 28),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
    )

    cv2.putText(
      frame,
      (
        "Q/A y0 W/S y1 E/D x1 R/F x2 "
        "T/G off_dir Y/H larg_inim | P salvar | ESC sair"
      ),
      (20, frame.shape[0] - 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )

    cv2.putText(
      frame,
      (
        f"CFG y0={config['y_inicio']} y1={config['y_fim']} "
        f"x1={config['x1_jogador']} x2={config['x2_jogador']} "
        f"off_dir={config['offset_direita_inimigo']} larg_ini={config['largura_inimigo']}"
      ),
      (20, 55),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )

    cv2.imshow(NOME_JANELA, frame)
    if roi_jogador.size > 0:
      cv2.imshow(JANELA_JOGADOR, roi_jogador)
      mascara = _mascara_cor_preview(roi_jogador)
      if mascara is not None:
        cv2.imshow(JANELA_MASCARA, mascara)
    if roi_inimigo.size > 0:
      cv2.imshow(JANELA_INIMIGO, roi_inimigo)
    tecla = cv2.waitKey(1) & 0xFF

    alterou = True
    if tecla == ord("q"):
      config["y_inicio"] -= 1
    elif tecla == ord("a"):
      config["y_inicio"] += 1
    elif tecla == ord("w"):
      config["y_fim"] -= 1
    elif tecla == ord("s"):
      config["y_fim"] += 1
    elif tecla == ord("e"):
      config["x1_jogador"] -= 1
    elif tecla == ord("d"):
      config["x1_jogador"] += 1
    elif tecla == ord("r"):
      config["x2_jogador"] -= 1
    elif tecla == ord("f"):
      config["x2_jogador"] += 1
    elif tecla == ord("t"):
      config["offset_direita_inimigo"] -= 1
    elif tecla == ord("g"):
      config["offset_direita_inimigo"] += 1
    elif tecla == ord("y"):
      config["largura_inimigo"] -= 1
    elif tecla == ord("h"):
      config["largura_inimigo"] += 1
    elif tecla == ord("p"):
      _clamp_config(config)
      salvar_configuracao_vida(config)
      print("Configuracao salva em data/calibracao_vida.json")
      alterou = False
    elif tecla == 27:
      break
    else:
      alterou = False

    if alterou:
      _clamp_config(config)
      definir_configuracao_vida(config)

  cv2.destroyWindow(NOME_JANELA)
  cv2.destroyWindow(JANELA_JOGADOR)
  cv2.destroyWindow(JANELA_INIMIGO)
  cv2.destroyWindow(JANELA_MASCARA)


