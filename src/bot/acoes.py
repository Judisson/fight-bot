import pyautogui

from src.utils.log import log

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


def clicar(x, y):

  try:
    pyautogui.moveTo(x, y)
    pyautogui.click()
  except pyautogui.FailSafeException:
    log("Failsafe ativado - mouse foi para o canto!")
