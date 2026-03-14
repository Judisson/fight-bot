import os
import sys

from src.bot.arena import iniciar_arena
from src.bot.debug_simples import iniciar_debug_simples
from src.bot.treino import iniciar_treino
from src.calibracao.modo_calibracao import iniciar_calibracao
from src.calibracao.modo_calibracao_roi import iniciar_calibracao_roi_jogar_novamente

OPCOES_FPS = [
  ("120", "Mais estavel"),
  ("240", "Mais rapido (padrrao)"),
]

MODOS = [
  ("arena", "Fluxo completo da arena"),
  ("treino", "Treino da IA em combate"),
  ("debug", "Tela simples de debug com YOLO"),
  ("calibracao", "Ajustar posicoes de vida e poder"),
  ("calibracao_roi", "Calibrar ROI do jogar novamente"),
]


def limpar_tela():
  os.system("cls" if os.name == "nt" else "clear")


def ler_tecla():
  if os.name == "nt":
    import msvcrt

    tecla = msvcrt.getwch()
    if tecla in ("\x00", "\xe0"):
      especial = msvcrt.getwch()
      if especial == "H":
        return "up"
      if especial == "P":
        return "down"
      return "other"

    if tecla == "\r":
      return "enter"

    return "other"

  import termios
  import tty

  fd = sys.stdin.fileno()
  antigo = termios.tcgetattr(fd)

  try:
    tty.setraw(fd)
    tecla = sys.stdin.read(1)

    if tecla == "\x1b":
      sequencia = sys.stdin.read(2)
      if sequencia == "[A":
        return "up"
      if sequencia == "[B":
        return "down"
      return "other"

    if tecla in ("\r", "\n"):
      return "enter"

    return "other"
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, antigo)


def escolher_com_setas(titulo, opcoes):
  indice_atual = 0

  while True:
    limpar_tela()
    print("FightBot")
    print("")
    print(titulo)
    print("")

    for indice, (valor, descricao) in enumerate(opcoes):
      cursor = ">" if indice == indice_atual else " "
      print(f" {cursor} {valor:<8} {descricao}")

    print("")
    print("Atalhos: ↑/↓, Enter")

    tecla = ler_tecla()

    if tecla == "up":
      indice_atual = (indice_atual - 1) % len(opcoes)
    elif tecla == "down":
      indice_atual = (indice_atual + 1) % len(opcoes)
    elif tecla == "enter":
      return opcoes[indice_atual][0]


def main() -> None:
  fps_escolhido = escolher_com_setas(
    "Escolha o FPS alvo:",
    OPCOES_FPS,
  )

  os.environ["BOT_FPS"] = fps_escolhido

  modo = escolher_com_setas(
    "Escolha o modo de luta:",
    MODOS,
  )

  limpar_tela()
  print(f"Iniciando modo: {modo} | FPS alvo: {fps_escolhido}")
  print("")

  if modo == "arena":
    iniciar_arena()
    return

  if modo == "calibracao":
    iniciar_calibracao()
    return

  if modo == "calibracao_roi":
    iniciar_calibracao_roi_jogar_novamente()
    return

  if modo == "debug":
    iniciar_debug_simples()
    return

  iniciar_treino()


if __name__ == "__main__":
  main()
