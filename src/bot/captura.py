import queue
import threading
import time

import cv2
import mss
import numpy as np

from src.utils.log import log

_SCT_LOCAL = threading.local()


def _obter_sct_thread():
  sct = getattr(_SCT_LOCAL, "instancia", None)
  if sct is None:
    sct = mss.mss()
    _SCT_LOCAL.instancia = sct
  return sct


def _resetar_sct_thread():
  sct = getattr(_SCT_LOCAL, "instancia", None)
  if sct is None:
    return

  try:
    sct.close()
  except Exception:
    pass

  _SCT_LOCAL.instancia = None


def capturar_tela(monitor):
  sct = _obter_sct_thread()
  screenshot = sct.grab(monitor)

  frame = np.array(screenshot)

  # converter BGRA -> BGR
  frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

  return frame


class CapturaAssincrona:

  def __init__(self, monitor):
    self.monitor = monitor
    self._queue = queue.Queue(maxsize=1)
    self._stop_event = threading.Event()
    self._thread = None
    self._ultimo_frame = None

  def iniciar(self):
    if self._thread is not None and self._thread.is_alive():
      return

    self._stop_event.clear()
    self._thread = threading.Thread(
      target=self._loop_captura,
      name="captura-assincrona",
      daemon=True,
    )
    self._thread.start()

  def parar(self):
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)

  def obter_frame(self, timeout=0.05):
    try:
      frame = self._queue.get(timeout=timeout)
      self._ultimo_frame = frame
      return frame
    except queue.Empty:
      return self._ultimo_frame

  def _loop_captura(self):
    while not self._stop_event.is_set():
      try:
        frame = capturar_tela(self.monitor)
      except Exception as erro:
        log(f"[CAPTURA] erro ao capturar frame: {erro}")
        _resetar_sct_thread()
        time.sleep(0.05)
        continue

      try:
        if self._queue.full():
          self._queue.get_nowait()
        self._queue.put_nowait(frame)
      except queue.Full:
        pass
