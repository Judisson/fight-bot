import os

import cv2

from src.utils.log import log
from src.visao.barras_poder import extrair_barras_poder, obter_bboxes_poder

LIMIAR_COLUNA_ATIVA = 0.45
NUM_BARRAS_ESPECIAL = 3

_CONTADOR_TENTATIVAS = 0

try:
  LOG_PODER_CADA = max(1, int(os.getenv("BOT_LOG_PODER_CADA", "30")))
except ValueError:
  LOG_PODER_CADA = 30

try:
  LIMIAR_SEGMENTO_CHEIO = float(os.getenv("BOT_LIMIAR_SEGMENTO_CHEIO", "60.0"))
except ValueError:
  LIMIAR_SEGMENTO_CHEIO = 60.0

try:
  H_MIN_PREENCHIDO = max(0, min(179, int(os.getenv("BOT_PODER_H_MIN", "8"))))
except ValueError:
  H_MIN_PREENCHIDO = 8

try:
  H_MAX_PREENCHIDO = max(0, min(179, int(os.getenv("BOT_PODER_H_MAX", "38"))))
except ValueError:
  H_MAX_PREENCHIDO = 38

try:
  S_MIN_PREENCHIDO = max(0, min(255, int(os.getenv("BOT_PODER_S_MIN", "90"))))
except ValueError:
  S_MIN_PREENCHIDO = 90

try:
  V_MIN_PREENCHIDO = max(0, min(255, int(os.getenv("BOT_PODER_V_MIN", "75"))))
except ValueError:
  V_MIN_PREENCHIDO = 75

try:
  MARGEM_X_SEGMENTO = float(os.getenv("BOT_PODER_MARGEM_X", "0.10"))
except ValueError:
  MARGEM_X_SEGMENTO = 0.10

try:
  MARGEM_Y_SEGMENTO = float(os.getenv("BOT_PODER_MARGEM_Y", "0.22"))
except ValueError:
  MARGEM_Y_SEGMENTO = 0.22

MARGEM_X_SEGMENTO = max(0.0, min(0.35, MARGEM_X_SEGMENTO))
MARGEM_Y_SEGMENTO = max(0.0, min(0.35, MARGEM_Y_SEGMENTO))


def _extrair_area_util(roi):
  if roi.size == 0:
    return None

  altura, largura, _ = roi.shape
  x1 = int(largura * MARGEM_X_SEGMENTO)
  x2 = max(x1 + 1, int(largura * (1.0 - MARGEM_X_SEGMENTO)))
  y1 = int(altura * MARGEM_Y_SEGMENTO)
  y2 = max(y1 + 1, int(altura * (1.0 - MARGEM_Y_SEGMENTO)))
  return roi[y1:y2, x1:x2]


def _mascara_barra_por_cor(roi):
  faixa = _extrair_area_util(roi)
  if faixa is None:
    return None

  hsv = cv2.cvtColor(faixa, cv2.COLOR_BGR2HSV)
  mascara = cv2.inRange(
    hsv,
    (H_MIN_PREENCHIDO, S_MIN_PREENCHIDO, V_MIN_PREENCHIDO),
    (H_MAX_PREENCHIDO, 255, 255),
  )
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
  mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
  return mascara


def _percentual_segmento(segmento):
  mascara = _mascara_barra_por_cor(segmento)
  if mascara is None:
    return None

  altura, largura = mascara.shape[:2]
  if altura == 0 or largura == 0:
    return None

  total_pixels = altura * largura
  ativos_totais = cv2.countNonZero(mascara)
  percentual_pixels = float((ativos_totais / total_pixels) * 100.0)

  colunas_ativas = 0
  for x in range(largura):
    coluna = mascara[:, x]
    ativos = cv2.countNonZero(coluna)
    if ativos > altura * LIMIAR_COLUNA_ATIVA:
      colunas_ativas += 1

  percentual_colunas = float((colunas_ativas / largura) * 100.0)

  # Combina cobertura de colunas com densidade real de pixels para evitar falso
  # positivo em barras "vazias" que ainda mantem bordas coloridas.
  percentual = min(percentual_colunas, min(100.0, percentual_pixels * 1.5))
  return round(percentual, 1)


def _separar_segmentos(roi, quantidade):
  if roi is None or roi.size == 0:
    return []

  largura = roi.shape[1]
  if largura < quantidade:
    return []

  segmentos = []
  largura_base = largura // quantidade
  for i in range(quantidade):
    x1 = i * largura_base
    if i == (quantidade - 1):
      x2 = largura
    else:
      x2 = (i + 1) * largura_base
    if x2 <= x1:
      continue
    segmentos.append(roi[:, x1:x2])
  return segmentos


def _detectar_nivel_especial(roi, ordem_segmentos):
  segmentos = _separar_segmentos(roi, NUM_BARRAS_ESPECIAL)
  if len(segmentos) != NUM_BARRAS_ESPECIAL:
    return None, [None] * NUM_BARRAS_ESPECIAL, None

  percentuais_ltr = [_percentual_segmento(seg) for seg in segmentos]
  percentuais_ordenados = []
  for indice in ordem_segmentos:
    if indice < 0 or indice >= len(percentuais_ltr):
      percentuais_ordenados.append(None)
    else:
      percentuais_ordenados.append(percentuais_ltr[indice])

  nivel = 0
  for percentual in percentuais_ordenados:
    if percentual is None or percentual < LIMIAR_SEGMENTO_CHEIO:
      break
    nivel += 1

  validos = [p for p in percentuais_ordenados if p is not None]
  percentual_total = None
  if validos:
    percentual_total = round(sum(validos) / len(validos), 1)

  return nivel, percentuais_ordenados, percentual_total


def obter_info_poder(frame, log_segmentos=False):
  global _CONTADOR_TENTATIVAS

  barra_jogador, barra_inimigo = extrair_barras_poder(frame)
  bbox_jogador, bbox_inimigo = obter_bboxes_poder(frame)

  # Player: E1 na direita, E3 na esquerda -> [2,1,0]
  nivel_jogador, segmentos_jogador_pct, poder_jogador_pct = _detectar_nivel_especial(
    barra_jogador,
    [2, 1, 0],
  )
  # Inimigo: E1 na esquerda, E3 na direita -> [0,1,2]
  nivel_inimigo, segmentos_inimigo_pct, poder_inimigo_pct = _detectar_nivel_especial(
    barra_inimigo,
    [0, 1, 2],
  )

  tem_especial_jogador = None if nivel_jogador is None else (nivel_jogador >= 1)
  tem_especial_inimigo = None if nivel_inimigo is None else (nivel_inimigo >= 1)

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
  x1, y1, x2, y2 = info_poder["bbox_jogador"]
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
