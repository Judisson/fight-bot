import pygetwindow as gw


def encontrar_janela(titulo):

  janelas = gw.getWindowsWithTitle(titulo)

  if not janelas:
    return None

  return janelas[0]


def obter_bbox_janela(janela):

  return {
    "top": janela.top,
    "left": janela.left,
    "width": janela.width,
    "height": janela.height
  }
