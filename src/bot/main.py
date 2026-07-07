import os
import sys

from src.bot.arena import iniciar_arena
from src.bot.debug_simples import iniciar_debug_simples
from src.bot.treino import iniciar_treino
from src.bot.treino_observacao import iniciar_treino_observacao
from src.calibracao.modo_calibracao import iniciar_calibracao
from src.calibracao.modo_calibracao_roi import iniciar_calibracao_roi_jogar_novamente

OPCOES_FPS = [
  ("120", "Mais estavel"),
  ("240", "Mais rapido (padrrao)"),
]

MODOS = [
  ("arena", "Fluxo completo da arena"),
  ("treino", "Menu de treino"),
  ("debug", "Tela simples de debug com YOLO"),
  ("calibracao", "Ajustar posicoes de vida e poder"),
  ("calibracao_roi", "Calibrar ROIs de templates"),
]

TIPO_TREINO = [
  ("treino_normal", "Treino online IA Pixel + CNN + PPO"),
  ("treino_observacao", "Treino de observacao (sem interferir)"),
]

FOCO_TREINO = [
  ("completo", "Treino Completo (Melhorar todas as habilidades)"),
  ("aparar", "Só Aparar (Focado em bloquear/aparar ataques inimigos)"),
  ("defender", "Só Defender (Focado em sobrevivência: bloqueio e esquiva)"),
  ("combo", "Só Combo (Focado em sequências ofensivas com resets de esquiva/bloqueio)"),
  ("destreza", "Só Destreza (Focado em esquivar e dar destreza perfeita)"),
  ("assistido", "Treino Assistido (Você joga e a IA imita e aprende com você)"),
]


def limpar_tela():
  os.system("cls" if os.name == "nt" else "clear")


def ler_tecla():
  if os.name == "nt":
    import msvcrt

    tecla = msvcrt.getwch()
    if tecla == "\x03":
      raise KeyboardInterrupt

    if tecla in ("\x00", "\xe0"):
      especial = msvcrt.getwch()
      if especial == "H":
        return "up"
      if especial == "P":
        return "down"
      return "other"

    if tecla == "\r":
      return "enter"
    if tecla == "\x1b":
      return "back"
    if tecla.lower() == "q":
      return "back"

    return "other"

  import termios
  import tty

  fd = sys.stdin.fileno()
  antigo = termios.tcgetattr(fd)

  try:
    tty.setraw(fd)
    tecla = sys.stdin.read(1)
    if tecla == "\x03":
      raise KeyboardInterrupt

    if tecla == "\x1b":
      sequencia = sys.stdin.read(2)
      if sequencia == "[A":
        return "up"
      if sequencia == "[B":
        return "down"
      return "back"

    if tecla in ("\r", "\n"):
      return "enter"
    if tecla.lower() == "q":
      return "back"

    return "other"
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, antigo)


def escolher_com_setas(titulo, opcoes, permitir_voltar=False):
  indice_atual = 0

  while True:
    limpar_tela()
    print("FightBot")
    print("")
    print(titulo)
    print("")

    for indice, (valor, descricao) in enumerate(opcoes):
      cursor = ">" if indice == indice_atual else " "
      print(f" {cursor} {valor:<18} {descricao}")

    print("")
    if permitir_voltar:
      print("Atalhos: up/down, Enter, Esc/Q (voltar), Ctrl+C (encerrar)")
    else:
      print("Atalhos: up/down, Enter, Ctrl+C (encerrar)")

    tecla = ler_tecla()

    if tecla == "up":
      indice_atual = (indice_atual - 1) % len(opcoes)
    elif tecla == "down":
      indice_atual = (indice_atual + 1) % len(opcoes)
    elif tecla == "enter":
      return opcoes[indice_atual][0]
    elif tecla == "back" and permitir_voltar:
      return None


def main() -> None:
  while True:
    fps_escolhido = escolher_com_setas(
      "Escolha o FPS alvo:",
      OPCOES_FPS,
      permitir_voltar=False,
    )
    os.environ["BOT_FPS"] = fps_escolhido

    while True:
      modo = escolher_com_setas(
        "Escolha o modo de luta:",
        MODOS,
        permitir_voltar=True,
      )
      if modo is None:
        break

      limpar_tela()
      print(f"Iniciando modo: {modo} | FPS alvo: {fps_escolhido}")
      print("")

      if modo == "arena":
        iniciar_arena()
        return

      if modo == "treino":
        while True:
          tipo = escolher_com_setas(
            "Escolha o tipo de treino:",
            TIPO_TREINO,
            permitir_voltar=True,
          )
          if tipo is None:
            break
            
          foco = escolher_com_setas(
            "Escolha o foco do treino:",
            FOCO_TREINO,
            permitir_voltar=True,
          )
          if foco is None:
            continue
            
          limpar_tela()
          print(f"Iniciando treino: {tipo} ({foco}) | FPS alvo: {fps_escolhido}")
          print("")
          if tipo == "treino_observacao":
            iniciar_treino_observacao(modo_treino=foco)
            return
          iniciar_treino(modo_treino=foco)
          return
        continue

      if modo == "calibracao":
        iniciar_calibracao()
        return

      if modo == "calibracao_roi":
        iniciar_calibracao_roi_jogar_novamente()
        return

      if modo == "debug":
        iniciar_debug_simples()
        return

      iniciar_treino(modo_treino="completo")
      return


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("\nEncerrado pelo usuario.")
