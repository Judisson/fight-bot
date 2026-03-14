import os


def logs_ativos():
  if os.getenv("BOT_SILENT", "0").strip() == "1":
    return False

  ambiente = os.getenv("BOT_ENV", "").strip().lower()
  if ambiente in {"prod", "producao", "production"}:
    return False

  return True


def log(*args, **kwargs):
  if not logs_ativos():
    return
  print(*args, **kwargs)
