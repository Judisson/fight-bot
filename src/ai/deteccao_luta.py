from src.bot.visao import encontrar_template


def esta_nocaute(frame):

  nocaute = encontrar_template(frame, "assets/derrota.png")

  if nocaute:
    return True

  return False


def esta_lutando(frame):

  luta = encontrar_template(frame, "assets/pausar-luta-button.png")

  if luta:
    return True

  return False
