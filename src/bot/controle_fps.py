import os
import time


def obter_fps_alvo():
  valor = os.getenv("BOT_FPS", "240").strip()
  try:
    fps = int(valor)
  except ValueError:
    fps = 240

  # Limite defensivo para evitar valores absurdos.
  return max(30, min(fps, 1000))


class ControleFPS:

  def __init__(self, fps_alvo):
    self.fps_alvo = fps_alvo
    self.intervalo = 1.0 / fps_alvo
    self._janela_inicio = time.perf_counter()
    self._frames_janela = 0

  def iniciar_ciclo(self):
    return time.perf_counter()

  def finalizar_ciclo(self, inicio_ciclo):
    decorrido = time.perf_counter() - inicio_ciclo
    restante = self.intervalo - decorrido

    if restante > 0:
      time.sleep(restante)

    self._frames_janela += 1
    agora = time.perf_counter()
    decorrido_janela = agora - self._janela_inicio

    if decorrido_janela >= 1.0:
      fps_real = self._frames_janela / decorrido_janela
      print(f"FPS real: {fps_real:.1f} | FPS alvo: {self.fps_alvo}")
      self._janela_inicio = agora
      self._frames_janela = 0
