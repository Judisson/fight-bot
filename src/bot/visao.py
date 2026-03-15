import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

def _obter_workers_templates():
  valor = os.getenv("BOT_TEMPLATE_THREADS", "").strip()
  if valor:
    try:
      workers = int(valor)
      return max(1, min(workers, 16))
    except ValueError:
      pass

  cpu = os.cpu_count() or 4
  return max(2, min(cpu, 8))


_POOL_TEMPLATES = ThreadPoolExecutor(
  max_workers=_obter_workers_templates(),
  thread_name_prefix="tmpl",
)


def _obter_escala_visao():
  valor = os.getenv("BOT_VISAO_MATCH_SCALE", "0.5").strip()
  try:
    escala = float(valor)
  except ValueError:
    escala = 0.5

  # Evita escala nula e valores > 1.
  return max(0.1, min(1.0, escala))


def _frame_para_gray(frame):
  if frame is None:
    return None

  if len(frame.shape) == 2:
    return frame

  if len(frame.shape) == 3 and frame.shape[2] == 1:
    return frame[:, :, 0]

  return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _normalizar_roi(frame, roi):
  if roi is None:
    return None

  if not isinstance(roi, (tuple, list)) or len(roi) != 4:
    return None

  try:
    x1, y1, x2, y2 = [int(v) for v in roi]
  except (TypeError, ValueError):
    return None

  altura, largura = frame.shape[:2]
  x1 = max(0, min(largura, x1))
  x2 = max(0, min(largura, x2))
  y1 = max(0, min(altura, y1))
  y2 = max(0, min(altura, y2))
  if x2 <= x1 or y2 <= y1:
    return None

  return (x1, y1, x2, y2)


def _redimensionar_frame_gray(frame_gray, escala):
  if frame_gray is None or escala >= 0.999:
    return frame_gray

  altura, largura = frame_gray.shape[:2]
  nova_largura = max(1, int(round(largura * escala)))
  nova_altura = max(1, int(round(altura * escala)))
  if nova_largura == largura and nova_altura == altura:
    return frame_gray

  return cv2.resize(frame_gray, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)


def _escalar_roi(roi, escala, largura, altura):
  if roi is None or escala >= 0.999:
    return roi

  x1, y1, x2, y2 = roi
  sx1 = max(0, min(largura - 1, int(round(x1 * escala))))
  sy1 = max(0, min(altura - 1, int(round(y1 * escala))))
  sx2 = max(1, min(largura, int(round(x2 * escala))))
  sy2 = max(1, min(altura, int(round(y2 * escala))))

  if sx2 <= sx1:
    sx2 = min(largura, sx1 + 1)
  if sy2 <= sy1:
    sy2 = min(altura, sy1 + 1)

  return (sx1, sy1, sx2, sy2)


def carregar_template(caminho_template):
  imagem = cv2.imread(caminho_template, cv2.IMREAD_GRAYSCALE)
  if imagem is None:
    try:
      buffer = np.fromfile(caminho_template, dtype=np.uint8)
    except OSError:
      buffer = np.array([], dtype=np.uint8)
    if buffer.size > 0:
      imagem = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
  return imagem


def _carregar_template_escalado(caminho_template, escala):
  template = carregar_template(caminho_template)
  if template is None:
    return None
  if escala >= 0.999:
    return template

  altura, largura = template.shape[:2]
  nova_largura = max(1, int(round(largura * escala)))
  nova_altura = max(1, int(round(altura * escala)))
  if nova_largura == largura and nova_altura == altura:
    template_escalado = template
  else:
    template_escalado = cv2.resize(
      template,
      (nova_largura, nova_altura),
      interpolation=cv2.INTER_AREA,
    )
  return template_escalado


def _encontrar_template_no_gray(frame_gray, caminho_template, limiar=0.8, roi=None, escala=1.0):
  if frame_gray is None:
    return None

  offset_x = 0
  offset_y = 0
  frame_busca = frame_gray
  roi_normalizada = _normalizar_roi(frame_gray, roi)
  if roi is not None:
    if roi_normalizada is None:
      return None
    x1, y1, x2, y2 = roi_normalizada
    frame_busca = frame_gray[y1:y2, x1:x2]
    offset_x = x1
    offset_y = y1

  template = _carregar_template_escalado(caminho_template, escala)
  if template is None:
    return None
  if len(template.shape) == 3:
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  altura_template, largura_template = template.shape[:2]
  altura_busca, largura_busca = frame_busca.shape[:2]
  if altura_busca < altura_template or largura_busca < largura_template:
    return None

  resultado = cv2.matchTemplate(frame_busca, template, cv2.TM_CCOEFF_NORMED)
  _, valor_maximo, _, local_maximo = cv2.minMaxLoc(resultado)

  if valor_maximo >= limiar:
    centro_x = local_maximo[0] + largura_template // 2 + offset_x
    centro_y = local_maximo[1] + altura_template // 2 + offset_y
    if escala < 0.999:
      centro_x = int(round(centro_x / escala))
      centro_y = int(round(centro_y / escala))
    return {
      "x": centro_x,
      "y": centro_y,
      "confianca": valor_maximo,
    }

  return None


def encontrar_template(frame, caminho_template, limiar=0.8, roi=None):
  frame_gray = _frame_para_gray(frame)
  if frame_gray is None:
    return None
  escala = _obter_escala_visao()
  frame_match = _redimensionar_frame_gray(frame_gray, escala)
  roi_match = None
  if roi is not None:
    altura_m, largura_m = frame_match.shape[:2]
    roi_match = _escalar_roi(roi, escala, largura_m, altura_m)
  return _encontrar_template_no_gray(
    frame_match,
    caminho_template,
    limiar,
    roi_match,
    escala=escala,
  )


def encontrar_templates_em_paralelo(frame, consultas):
  if frame is None or not consultas:
    return {}
  frame_gray = _frame_para_gray(frame)
  if frame_gray is None:
    return {}
  escala = _obter_escala_visao()
  frame_match = _redimensionar_frame_gray(frame_gray, escala)
  altura_m, largura_m = frame_match.shape[:2]

  futuros = {}
  for item in consultas:
    roi = None
    if len(item) == 2:
      nome, caminho_template = item
      limiar = 0.8
    elif len(item) == 3:
      nome, caminho_template, terceiro = item
      if isinstance(terceiro, (tuple, list)) and len(terceiro) == 4:
        limiar = 0.8
        roi = _escalar_roi(terceiro, escala, largura_m, altura_m)
      else:
        limiar = terceiro
    else:
      nome, caminho_template, limiar, roi = item
      roi = _escalar_roi(roi, escala, largura_m, altura_m) if roi is not None else None

    futuro = _POOL_TEMPLATES.submit(
      _encontrar_template_no_gray,
      frame_match,
      caminho_template,
      limiar,
      roi,
      escala,
    )
    futuros[futuro] = nome

  resultados = {}
  for futuro in as_completed(futuros):
    nome = futuros[futuro]
    try:
      resultados[nome] = futuro.result()
    except Exception:
      resultados[nome] = None

  return resultados
