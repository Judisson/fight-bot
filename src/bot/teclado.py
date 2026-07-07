import pydirectinput
import time

pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.0


def pressionar(tecla):

  pydirectinput.keyDown(tecla)
  pydirectinput.keyUp(tecla)


def segurar(tecla, duracao=0.1):

  pydirectinput.keyDown(tecla)
  time.sleep(duracao)
  pydirectinput.keyUp(tecla)
