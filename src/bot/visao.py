import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

_CACHE_TEMPLATES = {}
_CACHE_LOCK = threading.Lock()


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


def carregar_template(caminho_template):
  with _CACHE_LOCK:
    if caminho_template not in _CACHE_TEMPLATES:
      imagem = cv2.imread(caminho_template)
      if imagem is None:
        try:
          buffer = np.fromfile(caminho_template, dtype=np.uint8)
        except OSError:
          buffer = np.array([], dtype=np.uint8)
        if buffer.size > 0:
          imagem = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
      _CACHE_TEMPLATES[caminho_template] = imagem
    return _CACHE_TEMPLATES[caminho_template]


def encontrar_template(frame, caminho_template, limiar=0.8):
  if frame is None:
    return None

  template = carregar_template(caminho_template)
  if template is None:
    return None

  resultado = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
  _, valor_maximo, _, local_maximo = cv2.minMaxLoc(resultado)

  if valor_maximo >= limiar:
    altura, largura = template.shape[:2]
    return {
      "x": local_maximo[0] + largura // 2,
      "y": local_maximo[1] + altura // 2,
      "confianca": valor_maximo,
    }

  return None


def encontrar_templates_em_paralelo(frame, consultas):
  if frame is None or not consultas:
    return {}

  futuros = {}
  for item in consultas:
    if len(item) == 2:
      nome, caminho_template = item
      limiar = 0.8
    else:
      nome, caminho_template, limiar = item

    futuro = _POOL_TEMPLATES.submit(encontrar_template, frame, caminho_template, limiar)
    futuros[futuro] = nome

  resultados = {}
  for futuro in as_completed(futuros):
    nome = futuros[futuro]
    try:
      resultados[nome] = futuro.result()
    except Exception:
      resultados[nome] = None

  return resultados
