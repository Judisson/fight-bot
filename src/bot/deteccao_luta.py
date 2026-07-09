import os

from src.bot.visao import encontrar_template, encontrar_templates_em_paralelo

TEMPLATE_DERROTA = "assets/derrota.png"
TEMPLATE_VITORIA = os.getenv("BOT_TEMPLATE_VITORIA", "assets/vitória.png").strip()
TEMPLATE_LUTA = "assets/pausar-luta-button.png"


def esta_nocaute(frame):
  nocaute = encontrar_template(frame, TEMPLATE_DERROTA)
  if nocaute:
    return True
  return False


def esta_vitoria(frame):
  vitoria = encontrar_template(frame, TEMPLATE_VITORIA)
  if vitoria:
    return True
  return False


def esta_lutando(frame):
  luta = encontrar_template(frame, TEMPLATE_LUTA)
  if luta:
    return True
  return False


def obter_estado_luta(frame):
  consultas = [
    ("nocaute", TEMPLATE_DERROTA),
    ("lutando", TEMPLATE_LUTA),
  ]
  if TEMPLATE_VITORIA:
    consultas.append(("vitoria", TEMPLATE_VITORIA))

  resultados = encontrar_templates_em_paralelo(
    frame,
    consultas,
  )
  return {
    "nocaute": resultados.get("nocaute") is not None,
    "vitoria": resultados.get("vitoria") is not None,
    "em_luta": resultados.get("lutando") is not None,
  }
