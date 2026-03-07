import mss
import numpy as np
import cv2

sct = mss.mss()


def capturar_tela(monitor):

  screenshot = sct.grab(monitor)

  frame = np.array(screenshot)

  # converter BGRA -> BGR
  frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

  return frame
