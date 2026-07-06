import time

import cv2
import numpy as np

from src.bot.captura import capturar_tela
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.calibracao.config_vida import (
  carregar_configuracao_vida,
  definir_configuracao_vida,
  salvar_configuracao_vida,
)
from src.calibracao.config_especial import (
  carregar_configuracao_especial,
  definir_configuracao_especial,
  salvar_configuracao_especial,
)
from src.utils.log import log
from src.visao.barras_vida import obter_bboxes_vida

TITULO_JOGO = "Champions"
NOME_JANELA = "Calibracao de Vida e Especial"
JANELA_JOGADOR = "Calibracao - Barra Jogador (Alvo)"
JANELA_INIMIGO = "Calibracao - Barra Inimigo (Alvo)"
JANELA_MASCARA = "Calibracao - Mascara Cor (Alvo)"

_mouse = {"x": 0, "y": 0}
_clique = {"x": -1, "y": -1, "disparado": False}


def _on_mouse(evento, x, y, _flags, _params):
  if evento == cv2.EVENT_MOUSEMOVE:
    _mouse["x"] = x
    _mouse["y"] = y
  elif evento == cv2.EVENT_LBUTTONDOWN:
    _clique["x"] = x
    _clique["y"] = y
    _clique["disparado"] = True


def _obter_monitor_jogo():
  log("Procurando janela do jogo...")
  janela = None

  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(0.5)

  log("Janela encontrada!")

  if focar_jogo(TITULO_JOGO):
    log("Janela focada automaticamente.")
  else:
    log("Nao foi possivel focar automaticamente a janela.")

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


def _aplicar_tecla_config(tecla, config):
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
  else:
    return False
  return True


def _texto_cfg_curto(config):
  return (
    f"y0={config['y_inicio']} y1={config['y_fim']} "
    f"x1={config['x1_jogador']} x2={config['x2_jogador']} "
    f"off_dir={config['offset_direita_inimigo']} larg_ini={config['largura_inimigo']}"
  )


def iniciar_calibracao():
  monitor = _obter_monitor_jogo()
  config_vida = carregar_configuracao_vida(forcar_reload=True)
  config_especial = carregar_configuracao_especial(forcar_reload=True)
  definir_configuracao_vida(config_vida)
  definir_configuracao_especial(config_especial)
  alvo = "vida"
  alvo_especial = "e1"

  log("Calibracao iniciada. Teclas:")
  log("1 = VIDA | 4/5/6 = JOGADOR (E1/E2/E3) | 7/8/9 = INIMIGO (E1/E2/E3)")
  log("Q/A y_inicio | W/S y_fim (vida)")
  log("E/D x1_jogador | R/F x2_jogador")
  log("T/G offset_direita_inimigo | Y/H largura_inimigo")
  log("Clique na janela do jogo para definir o pixel do especial selecionado")
  log("P salva todas as configs | ESC encerra")

  cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
  cv2.namedWindow(JANELA_JOGADOR, cv2.WINDOW_NORMAL)
  cv2.namedWindow(JANELA_INIMIGO, cv2.WINDOW_NORMAL)
  cv2.namedWindow(JANELA_MASCARA, cv2.WINDOW_NORMAL)
  cv2.setMouseCallback(NOME_JANELA, _on_mouse)

  while True:
    frame = capturar_tela(monitor)
    bbox_vida_jogador, bbox_vida_inimigo = obter_bboxes_vida(frame)

    vx1, vy1, vx2, vy2 = bbox_vida_jogador
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 220, 255), 2)

    vx1, vy1, vx2, vy2 = bbox_vida_inimigo
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 200, 120), 2)

    if _clique["disparado"]:
      _clique["disparado"] = False
      cx = _clique["x"]
      cy = _clique["y"]
      if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
        if alvo.startswith("especial"):
          cor = [int(c) for c in frame[cy, cx]]
          grupo = "jogador" if alvo == "especial_jogador" else "inimigo"
          config_especial[grupo][alvo_especial]["x"] = cx
          config_especial[grupo][alvo_especial]["y"] = cy
          config_especial[grupo][alvo_especial]["color"] = cor
          log(f"Calibrado {grupo.upper()} {alvo_especial.upper()}: ({cx}, {cy}) Cor BGR: {cor}")

    # Desenhar pontos do jogador (azul)
    for key in ["e1", "e2", "e3"]:
      pt = config_especial["jogador"][key]
      px, py = pt["x"], pt["y"]
      if px > 0 and py > 0:
        cor_calibrada = tuple(pt["color"])
        cv2.circle(frame, (px, py), 6, (255, 0, 0), 1)  # Circulo azul de destaque
        cv2.circle(frame, (px, py), 3, cor_calibrada, -1)
        cv2.putText(
          frame,
          f"J-{key.upper()}",
          (px + 8, py + 4),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.38,
          (255, 120, 0),
          1,
          cv2.LINE_AA,
        )

    # Desenhar pontos do inimigo (verde)
    for key in ["e1", "e2", "e3"]:
      pt = config_especial["inimigo"][key]
      px, py = pt["x"], pt["y"]
      if px > 0 and py > 0:
        cor_calibrada = tuple(pt["color"])
        cv2.circle(frame, (px, py), 6, (0, 255, 0), 1)  # Circulo verde de destaque
        cv2.circle(frame, (px, py), 3, cor_calibrada, -1)
        cv2.putText(
          frame,
          f"I-{key.upper()}",
          (px + 8, py + 4),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.38,
          (0, 255, 0),
          1,
          cv2.LINE_AA,
        )

    if alvo == "vida":
      x1, y1, x2, y2 = bbox_vida_jogador
      x1i, y1i, x2i, y2i = bbox_vida_inimigo
      config_atual = config_vida
      cor_alvo = (0, 255, 255)
    else:
      x1, y1, x2, y2 = 0, 0, 0, 0
      x1i, y1i, x2i, y2i = 0, 0, 0, 0
      config_atual = None
      cor_alvo = (255, 120, 0) if alvo == "especial_jogador" else (0, 255, 0)

    if config_atual is not None:
      cv2.rectangle(frame, (x1, y1), (x2, y2), cor_alvo, 3)
      cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), cor_alvo, 3)
      roi_jogador = frame[y1:y2, x1:x2]
      roi_inimigo = frame[y1i:y2i, x1i:x2i]
    else:
      roi_jogador = np.array([])
      roi_inimigo = np.array([])

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
      "1 VIDA | 4/5/6 JOG_ESP | 7/8/9 INI_ESP | P salvar | ESC sair",
      (20, frame.shape[0] - 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.48,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )

    if alvo.startswith("especial"):
      grupo = "jogador" if alvo == "especial_jogador" else "inimigo"
      pt_atual = config_especial[grupo][alvo_especial]
      texto_cfg = f"GRUPO={grupo.upper()} PT={alvo_especial.upper()} x={pt_atual['x']} y={pt_atual['y']} cor={pt_atual['color']}"
    else:
      texto_cfg = _texto_cfg_curto(config_atual)

    cv2.putText(
      frame,
      f"ALVO: {alvo.upper()} | CFG: {texto_cfg}",
      (20, 55),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      cor_alvo,
      1,
      cv2.LINE_AA,
    )

    cv2.putText(
      frame,
      f"VIDA: {_texto_cfg_curto(config_vida)}",
      (20, 78),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.45,
      (0, 220, 255),
      1,
      cv2.LINE_AA,
    )

    def _fmt_pt(pt):
      return f"({pt['x']},{pt['y']})={pt['color']}"

    cv2.putText(
      frame,
      f"JOG_ESP: E1={_fmt_pt(config_especial['jogador']['e1'])} E2={_fmt_pt(config_especial['jogador']['e2'])} E3={_fmt_pt(config_especial['jogador']['e3'])}",
      (20, 98),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (255, 120, 0),
      1,
      cv2.LINE_AA,
    )

    cv2.putText(
      frame,
      f"INI_ESP: E1={_fmt_pt(config_especial['inimigo']['e1'])} E2={_fmt_pt(config_especial['inimigo']['e2'])} E3={_fmt_pt(config_especial['inimigo']['e3'])}",
      (20, 118),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (0, 255, 0),
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
    alterou = False

    if tecla == ord("1"):
      alvo = "vida"
    elif tecla == ord("4"):
      alvo = "especial_jogador"
      alvo_especial = "e1"
      log("Calibracao ativa para Especial 1 do Jogador (J-E1)")
    elif tecla == ord("5"):
      alvo = "especial_jogador"
      alvo_especial = "e2"
      log("Calibracao ativa para Especial 2 do Jogador (J-E2)")
    elif tecla == ord("6"):
      alvo = "especial_jogador"
      alvo_especial = "e3"
      log("Calibracao ativa para Especial 3 do Jogador (J-E3)")
    elif tecla == ord("7"):
      alvo = "especial_inimigo"
      alvo_especial = "e1"
      log("Calibracao ativa para Especial 1 do Inimigo (I-E1)")
    elif tecla == ord("8"):
      alvo = "especial_inimigo"
      alvo_especial = "e2"
      log("Calibracao ativa para Especial 2 do Inimigo (I-E2)")
    elif tecla == ord("9"):
      alvo = "especial_inimigo"
      alvo_especial = "e3"
      log("Calibracao ativa para Especial 3 do Inimigo (I-E3)")
    elif tecla == ord("p"):
      _clamp_config(config_vida)
      salvar_configuracao_vida(config_vida)
      salvar_configuracao_especial(config_especial)
      log("Configuracoes salvas!")
    elif tecla == 27:
      break
    else:
      if config_atual is not None:
        alterou = _aplicar_tecla_config(tecla, config_atual)

    if alterou and config_atual is not None:
      _clamp_config(config_atual)
      if alvo == "vida":
        definir_configuracao_vida(config_atual)

  cv2.destroyWindow(NOME_JANELA)
  cv2.destroyWindow(JANELA_JOGADOR)
  cv2.destroyWindow(JANELA_INIMIGO)
  cv2.destroyWindow(JANELA_MASCARA)
