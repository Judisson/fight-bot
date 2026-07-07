import os
import sys

# Ativa suporte ANSI (cores) nativo no CMD/PowerShell do Windows
if sys.platform == "win32":
  try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
  except Exception:
    pass


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

  sep = kwargs.get("sep", " ")
  texto = sep.join(str(arg) for arg in args)

  if "[REWARD_RT]" in texto:
    if "GRATIFICACAO" in texto:
      # Verde brilhante para gratificações
      print(f"\033[92m{texto}\033[0m", **{k: v for k, v in kwargs.items() if k != "sep"})
    elif "PUNICAO" in texto:
      # Vermelho brilhante para punições
      print(f"\033[91m{texto}\033[0m", **{k: v for k, v in kwargs.items() if k != "sep"})
    else:
      # Azul brilhante para outros logs de reward
      print(f"\033[94m{texto}\033[0m", **{k: v for k, v in kwargs.items() if k != "sep"})
  else:
    # Azul brilhante para todos os demais logs da aplicação
    print(f"\033[94m{texto}\033[0m", **{k: v for k, v in kwargs.items() if k != "sep"})
