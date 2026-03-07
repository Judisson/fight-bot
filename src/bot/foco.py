import pygetwindow as gw


def focar_jogo(titulo):
  janelas = gw.getWindowsWithTitle(titulo)

  if not janelas:
    return False

  janela = janelas[0]

  try:
    if janela.isMinimized:
      janela.restore()
    janela.activate()
    return True
  except Exception:
    return False
