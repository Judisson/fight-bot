import os

import cv2
import numpy as np

from src.utils.log import log

from src.visao.barras_vida import extrair_barras, obter_bboxes_vida

LIMIAR_COLUNA_ATIVA = 0.30

_CONTADOR_TENTATIVAS = 0

try:
  LOG_VIDA_CADA = max(1, int(os.getenv("BOT_LOG_VIDA_CADA", "30")))
except ValueError:
  LOG_VIDA_CADA = 30


def _extrair_faixa_central(roi):
  if roi.size == 0:
    return None

  altura, _, _ = roi.shape
  y1 = int(altura * 0.25)
  y2 = max(y1 + 1, int(altura * 0.75))
  return roi[y1:y2, :]


def _mascara_barra_por_cor(roi):
  faixa = _extrair_faixa_central(roi)
  if faixa is None:
    return None

  hsv = cv2.cvtColor(faixa, cv2.COLOR_BGR2HSV)
  # Mantem qualquer pixel nao escuro; remove fundo preto da HUD.
  mascara = cv2.inRange(hsv, (0, 0, 50), (179, 255, 255))
  return mascara


def _detectar_percentual_por_cor(roi):
  mascara = _mascara_barra_por_cor(roi)
  if mascara is None:
    return None, 0.0, 0, 0

  altura, largura = mascara.shape[:2]
  if altura == 0 or largura == 0:
    return None, 0.0, 0, 0

  atividade_por_coluna = np.sum(mascara > 0, axis=0)
  colunas_ativas = int(np.sum(atividade_por_coluna > altura * LIMIAR_COLUNA_ATIVA))
  colunas_escuras_total = int(np.sum(atividade_por_coluna == 0))

  # Escuridao relevante para perda de vida vem do final da barra (direita).
  zeros_finais_rev = (atividade_por_coluna == 0)[::-1]
  if not np.any(~zeros_finais_rev):
    colunas_escuras_finais = largura
  else:
    colunas_escuras_finais = int(np.argmax(~zeros_finais_rev))

  percentual = float((colunas_ativas / largura) * 100.0)
  confianca = min(1.0, max(0.0, colunas_ativas / max(1, largura)))
  return round(percentual, 1), confianca, colunas_escuras_total, colunas_escuras_finais


def obter_info_vida(frame):
  global _CONTADOR_TENTATIVAS

  barra_jogador, barra_inimigo = extrair_barras(frame)
  bbox_jogador, bbox_inimigo = obter_bboxes_vida(frame)

  (
    vida_jogador_pct,
    conf_jogador,
    colunas_escuras_jogador_total,
    colunas_escuras_jogador_finais,
  ) = _detectar_percentual_por_cor(barra_jogador)
  (
    vida_inimigo_pct,
    conf_inimigo,
    colunas_escuras_inimigo_total,
    colunas_escuras_inimigo_finais,
  ) = _detectar_percentual_por_cor(barra_inimigo)

  _CONTADOR_TENTATIVAS += 1
  if (_CONTADOR_TENTATIVAS % LOG_VIDA_CADA) == 0:
    log(
      f"[VIDA_COR #{_CONTADOR_TENTATIVAS}] "
      f"JOG bbox={bbox_jogador} valor={vida_jogador_pct} conf={conf_jogador:.3f} esc_fim={colunas_escuras_jogador_finais} | "
      f"INI bbox={bbox_inimigo} valor={vida_inimigo_pct} conf={conf_inimigo:.3f} esc_fim={colunas_escuras_inimigo_finais}"
    )

  return {
    "bbox_jogador": bbox_jogador,
    "bbox_inimigo": bbox_inimigo,
    "vida_jogador_pct": vida_jogador_pct,
    "vida_inimigo_pct": vida_inimigo_pct,
    "conf_jogador": conf_jogador,
    "conf_inimigo": conf_inimigo,
    "colunas_escuras_jogador_total": colunas_escuras_jogador_total,
    "colunas_escuras_jogador_finais": colunas_escuras_jogador_finais,
    "colunas_escuras_inimigo_total": colunas_escuras_inimigo_total,
    "colunas_escuras_inimigo_finais": colunas_escuras_inimigo_finais,
  }


def _formatar_pct(valor):
  if valor is None:
    return "?"
  return f"{valor:.1f}%"


def desenhar_info_vida(frame, info_vida):
  x1, y1, x2, y2 = info_vida["bbox_jogador"]
  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
  cv2.putText(
    frame,
    (
      f"Vida voce: {_formatar_pct(info_vida['vida_jogador_pct'])} "
      f"conf={info_vida['conf_jogador']:.2f}"
    ),
    (x1, max(20, y1 - 8)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (255, 220, 0),
    1,
    cv2.LINE_AA,
  )

  x1, y1, x2, y2 = info_vida["bbox_inimigo"]
  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 120), 2)
  cv2.putText(
    frame,
    (
      f"Vida inimigo: {_formatar_pct(info_vida['vida_inimigo_pct'])} "
      f"conf={info_vida['conf_inimigo']:.2f}"
    ),
    (x1, max(20, y1 - 8)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (255, 160, 0),
    1,
    cv2.LINE_AA,
  )

  return frame

