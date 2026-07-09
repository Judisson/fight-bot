import os
import time
from pynput import keyboard
from src.bot.acoes_luta import (
  TECLA_ESQUIVA,
  TECLA_BLOQUEIO,
  TECLA_ESPECIAL,
  TECLA_ATAQUE_LEVE,
  TECLA_ATAQUE_MEDIO,
  ACAO_ESQUIVA,
  ACAO_BLOQUEIO,
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ESPECIAL,
)

# Mapeia teclas (em minúsculas) para seus nomes legíveis e constantes do bot
NOMES_TECLAS = {
  TECLA_ATAQUE_LEVE.lower().strip(): ("L", "ACAO_ATAQUE_LEVE"),
  TECLA_ATAQUE_MEDIO.lower().strip(): ("M", "ACAO_ATAQUE_MEDIO"),
  TECLA_ESQUIVA.lower().strip(): ("D", "ACAO_ESQUIVA"),
  TECLA_BLOQUEIO.lower().strip(): ("B", "ACAO_BLOQUEIO"),
  TECLA_ESPECIAL.lower().strip(): ("E", "ACAO_ESPECIAL"),
}


def iniciar_gravador_combo():
  while True:
    os.system("cls" if os.name == "nt" else "clear")
    print("==================================================")
    print("          GRAVADOR DE COMBOS DO FIGHTBOT         ")
    print("==================================================")
    print("Instruções:")
    print("  - Vá para o jogo/emulador.")
    print("  - Use as teclas normais que você configurou no bot:")
    print(f"      {TECLA_ATAQUE_LEVE.upper()} : Ataque Leve (L)")
    print(f"      {TECLA_ATAQUE_MEDIO.upper()} : Ataque Médio (M)")
    print(f"      {TECLA_ESQUIVA.upper()} : Destreza/Esquiva (D)")
    print(f"      {TECLA_BLOQUEIO.upper()} : Bloqueio (B)")
    print(f"      {TECLA_ESPECIAL.upper()} : Especial (E)")
    print("")
    print("  - Pressione BACKSPACE para LIMPAR a gravação atual.")
    print("  - Pressione ENTER ou ESC para ENCERRAR e gerar as linhas do combo.")
    print("==================================================")
    input("Pressione ENTER para iniciar a escuta do teclado...")
    print("\n>>> GRAVANDO! Digite o seu combo no teclado...")
    print("--------------------------------------------------")

    acoes_gravadas = []
    timestamps = []

    def on_press(key):
      nonlocal acoes_gravadas, timestamps
      
      # Trata teclas de comando especiais
      if key == keyboard.Key.backspace:
        acoes_gravadas.clear()
        timestamps.clear()
        print("  [LIMPO] Gravação de combo limpa! Digite o combo novamente.")
        return

      if key in (keyboard.Key.esc, keyboard.Key.enter):
        # Salva o tempo final para o delay da última ação
        agora = time.time()
        timestamps.append(agora)
        return False  # encerra o listener

      try:
        # Pynput retorna a tecla normal com key.char
        char = key.char.lower().strip() if key.char else None
        if char in NOMES_TECLAS:
          nome_amigavel, acao_const = NOMES_TECLAS[char]
          agora = time.time()
          acoes_gravadas.append((acao_const, nome_amigavel))
          timestamps.append(agora)
          print(f"  [REGISTRADO] Tecla '{char}' -> Ação: {nome_amigavel}")
      except AttributeError:
        pass

    # Inicia listener síncrono que bloqueia até retornar False no on_press
    with keyboard.Listener(on_press=on_press) as listener:
      listener.join()

    print("\n--------------------------------------------------")
    print("Gravação finalizada!")

    if not acoes_gravadas:
      print("Nenhuma tecla configurada foi detectada.")
    else:
      # Calcula os delays baseados nos timestamps dos pressionamentos
      delays = []
      for i in range(len(acoes_gravadas)):
        tempo_atual = timestamps[i]
        tempo_proximo = timestamps[i + 1]
        delay = tempo_proximo - tempo_atual
        delays.append(round(delay, 3))

      print("\n>>> CÓDIGO PYTHON FORMATADO PARA O SEU COMBO:")
      print("==================================================")
      print("self._acoes_combo = [")
      for (acao_const, nome_amigavel), delay in zip(acoes_gravadas, delays):
        print(f"  ({acao_const}, {delay:.3f}),")
      print("]")
      print("==================================================")

      print("\n>>> DIALETO DOS COMBOS (Exemplo visual):")
      dialeto_partes = []
      for (acao_const, nome_amigavel), delay in zip(acoes_gravadas, delays):
        dialeto_partes.append(f"{nome_amigavel}({int(delay * 1000)}ms)")
      print(" , ".join(dialeto_partes))
      print("==================================================")

    # Pergunta se o usuário quer gravar novamente ou encerrar
    escolha = input("\nDeseja gravar outro combo? (s/n): ").strip().lower()
    if escolha != 's':
      break
