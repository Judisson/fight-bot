import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


def clicar(x, y):

  try:
    pyautogui.moveTo(x, y)
    pyautogui.click()
  except pyautogui.FailSafeException:
    print("Failsafe ativado - mouse foi para o canto!")
