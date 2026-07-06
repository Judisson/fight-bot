import os

import cv2

from src.utils.log import log

_CONTADOR_TENTATIVAS = 0

try:
  LOG_PODER_CADA = max(1, int(os.getenv("BOT_LOG_PODER_CADA", "30")))
except ValueError:
  LOG_PODER_CADA = 30


def _match_cor_pixel(cor_atual, cor_alvo, tol):
  return all(abs(int(c1) - int(c2)) <= tol for c1, c2 in zip(cor_atual, cor_alvo))


def obter_info_poder(frame, log_segmentos=False):
  global _CONTADOR_TENTATIVAS

  bbox_jogador = (0, 0, 0, 0)
  bbox_inimigo = (0, 0, 0, 0)

  # Valores padrão
  nivel_jogador = 0
  segmentos_jogador_pct = [0.0, 0.0, 0.0]
  poder_jogador_pct = 0.0
  tem_especial_jogador = False

  nivel_inimigo = 0
  segmentos_inimigo_pct = [0.0, 0.0, 0.0]
  poder_inimigo_pct = 0.0
  tem_especial_inimigo = False

  from src.calibracao.config_especial import carregar_configuracao_especial
  config_esp = carregar_configuracao_especial()

  if frame is not None:
    tol = config_esp.get("tolerancia", 30)
    altura, largura = frame.shape[:2]

    # 1. Jogador Especial
    cfg_jog = config_esp.get("jogador", {})
    usa_pixel_jogador = (
      cfg_jog.get("e1", {}).get("x", 0) > 0 and cfg_jog.get("e1", {}).get("y", 0) > 0 and
      cfg_jog.get("e2", {}).get("x", 0) > 0 and cfg_jog.get("e2", {}).get("y", 0) > 0 and
      cfg_jog.get("e3", {}).get("x", 0) > 0 and cfg_jog.get("e3", {}).get("y", 0) > 0
    )

    if usa_pixel_jogador:
      match_j1 = False
      match_j2 = False
      match_j3 = False

      p1 = cfg_jog["e1"]
      if 0 <= p1["x"] < largura and 0 <= p1["y"] < altura:
        match_j1 = _match_cor_pixel(frame[p1["y"], p1["x"]], p1["color"], tol)
      p2 = cfg_jog["e2"]
      if 0 <= p2["x"] < largura and 0 <= p2["y"] < altura:
        match_j2 = _match_cor_pixel(frame[p2["y"], p2["x"]], p2["color"], tol)
      p3 = cfg_jog["e3"]
      if 0 <= p3["x"] < largura and 0 <= p3["y"] < altura:
        match_j3 = _match_cor_pixel(frame[p3["y"], p3["x"]], p3["color"], tol)

      if match_j3:
        nivel_jogador = 3
      elif match_j2:
        nivel_jogador = 2
      elif match_j1:
        nivel_jogador = 1

      segmentos_jogador_pct = [
        100.0 if match_j1 else 0.0,
        100.0 if match_j2 else 0.0,
        100.0 if match_j3 else 0.0
      ]
      poder_jogador_pct = round((nivel_jogador / 3.0) * 100.0, 1)
      tem_especial_jogador = (nivel_jogador >= 1)

    # 2. Inimigo Especial
    cfg_ini = config_esp.get("inimigo", {})
    usa_pixel_inimigo = (
      cfg_ini.get("e1", {}).get("x", 0) > 0 and cfg_ini.get("e1", {}).get("y", 0) > 0 and
      cfg_ini.get("e2", {}).get("x", 0) > 0 and cfg_ini.get("e2", {}).get("y", 0) > 0 and
      cfg_ini.get("e3", {}).get("x", 0) > 0 and cfg_ini.get("e3", {}).get("y", 0) > 0
    )

    if usa_pixel_inimigo:
      match_i1 = False
      match_i2 = False
      match_i3 = False

      p1 = cfg_ini["e1"]
      if 0 <= p1["x"] < largura and 0 <= p1["y"] < altura:
        match_i1 = _match_cor_pixel(frame[p1["y"], p1["x"]], p1["color"], tol)
      p2 = cfg_ini["e2"]
      if 0 <= p2["x"] < largura and 0 <= p2["y"] < altura:
        match_i2 = _match_cor_pixel(frame[p2["y"], p2["x"]], p2["color"], tol)
      p3 = cfg_ini["e3"]
      if 0 <= p3["x"] < largura and 0 <= p3["y"] < altura:
        match_i3 = _match_cor_pixel(frame[p3["y"], p3["x"]], p3["color"], tol)

      if match_i3:
        nivel_inimigo = 3
      elif match_i2:
        nivel_inimigo = 2
      elif match_i1:
        nivel_inimigo = 1

      segmentos_inimigo_pct = [
        100.0 if match_i1 else 0.0,
        100.0 if match_i2 else 0.0,
        100.0 if match_i3 else 0.0
      ]
      poder_inimigo_pct = round((nivel_inimigo / 3.0) * 100.0, 1)
      tem_especial_inimigo = (nivel_inimigo >= 1)

  _CONTADOR_TENTATIVAS += 1
  if log_segmentos and ((_CONTADOR_TENTATIVAS % LOG_PODER_CADA) == 0):
    log(
      f"[PODER_SEG #{_CONTADOR_TENTATIVAS}] "
      f"JOG E={nivel_jogador} seg={segmentos_jogador_pct} | "
      f"INI E={nivel_inimigo} seg={segmentos_inimigo_pct}"
    )

  return {
    "bbox_jogador": bbox_jogador,
    "bbox_inimigo": bbox_inimigo,
    "poder_jogador_pct": poder_jogador_pct,
    "poder_inimigo_pct": poder_inimigo_pct,
    "nivel_especial_jogador": nivel_jogador,
    "nivel_especial_inimigo": nivel_inimigo,
    "segmentos_jogador_pct": segmentos_jogador_pct,
    "segmentos_inimigo_pct": segmentos_inimigo_pct,
    "tem_especial_jogador": tem_especial_jogador,
    "tem_especial_inimigo": tem_especial_inimigo,
  }


def _formatar_pct(valor):
  if valor is None:
    return "?"
  return f"{valor:.1f}%"


def _formatar_segmentos(segmentos):
  partes = []
  for valor in segmentos:
    if valor is None:
      partes.append("?")
    else:
      partes.append(f"{valor:.0f}")
  return "[" + ",".join(partes) + "]"


def desenhar_info_poder(frame, info_poder):
  # Desenha do jogador se as bboxes forem validas (no pixel check, as bboxes sao 0,0,0,0 por default, entao nao desenha nada)
  x1, y1, x2, y2 = info_poder["bbox_jogador"]
  if x2 > x1:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
    cv2.putText(
      frame,
      (
        f"Poder voce: {_formatar_pct(info_poder['poder_jogador_pct'])} "
        f"E={info_poder['nivel_especial_jogador']} seg={_formatar_segmentos(info_poder['segmentos_jogador_pct'])}"
      ),
      (x1, max(20, y1 - 8)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (255, 200, 80),
      1,
      cv2.LINE_AA,
    )

  x1, y1, x2, y2 = info_poder["bbox_inimigo"]
  if x2 > x1:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 80, 255), 2)
    cv2.putText(
      frame,
      (
        f"Poder inimigo: {_formatar_pct(info_poder['poder_inimigo_pct'])} "
        f"E={info_poder['nivel_especial_inimigo']} seg={_formatar_segmentos(info_poder['segmentos_inimigo_pct'])}"
      ),
      (x1, max(20, y1 - 8)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (220, 120, 255),
      1,
      cv2.LINE_AA,
    )

  return frame
