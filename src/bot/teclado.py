import pydirectinput
import time

pydirectinput.FAILSAFE = False


def pressionar(tecla):

  pydirectinput.keyDown(tecla)
  pydirectinput.keyUp(tecla)


def segurar(tecla, duracao=0.1):

  pydirectinput.keyDown(tecla)
  time.sleep(duracao)
  pydirectinput.keyUp(tecla)
