import cv2

_CACHE_TEMPLATES = {}


def carregar_template(caminho_template):
  if caminho_template not in _CACHE_TEMPLATES:
    _CACHE_TEMPLATES[caminho_template] = cv2.imread(caminho_template)
  return _CACHE_TEMPLATES[caminho_template]


def encontrar_template(frame, caminho_template, limiar=0.8):
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
      "confianca": valor_maximo
    }

  return None
