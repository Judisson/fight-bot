import queue
import threading

import cv2

NOME_JANELA = "Visao do Bot"

_janela_iniciada = False
_renderer_iniciado = False
_renderer_thread = None
_encerrar_solicitado = False

_frame_queue = queue.Queue(maxsize=1)
_stop_event = threading.Event()
_metricas_lock = threading.Lock()
_metricas = {
  "fps_real": 0.0,
  "fps_alvo": 0,
  "proc_ms": 0.0,
  "sleep_ms": 0.0,
}


def _iniciar_janela():
  global _janela_iniciada
  if _janela_iniciada:
    return

  cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(NOME_JANELA, 960, 540)
  _janela_iniciada = True


def atualizar_metricas(metricas):
  if metricas is None:
    return

  with _metricas_lock:
    _metricas["fps_real"] = float(metricas.get("fps_real", 0.0))
    _metricas["fps_alvo"] = int(metricas.get("fps_alvo", 0))
    _metricas["proc_ms"] = float(metricas.get("proc_ms", 0.0))
    _metricas["sleep_ms"] = float(metricas.get("sleep_ms", 0.0))


def _desenhar_overlay(frame_debug):
  with _metricas_lock:
    fps_real = _metricas["fps_real"]
    fps_alvo = _metricas["fps_alvo"]
    proc_ms = _metricas["proc_ms"]
    sleep_ms = _metricas["sleep_ms"]

  texto_fps = (
    f"FPS: {fps_real:.1f}/{fps_alvo} | "
    f"proc: {proc_ms:.2f}ms | sleep: {sleep_ms:.2f}ms"
  )
  cv2.putText(
    frame_debug,
    texto_fps,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 255),
    2,
  )


def _loop_render():
  global _encerrar_solicitado
  _iniciar_janela()

  ultimo_frame = None

  while not _stop_event.is_set():
    try:
      while True:
        ultimo_frame = _frame_queue.get_nowait()
    except queue.Empty:
      pass

    if ultimo_frame is not None:
      _desenhar_overlay(ultimo_frame)
      cv2.imshow(NOME_JANELA, ultimo_frame)

    if cv2.waitKey(1) == 27:
      _encerrar_solicitado = True
      _stop_event.set()
      break

  try:
    cv2.destroyWindow(NOME_JANELA)
  except Exception:
    pass


def _iniciar_renderer():
  global _renderer_iniciado, _renderer_thread

  if _renderer_iniciado:
    return

  _stop_event.clear()
  _renderer_thread = threading.Thread(target=_loop_render, name="debug-render", daemon=True)
  _renderer_thread.start()
  _renderer_iniciado = True


def exibir(frame):
  if frame is None:
    return

  _iniciar_renderer()

  if _encerrar_solicitado:
    raise KeyboardInterrupt("Encerrado pelo usuario (ESC).")

  try:
    if _frame_queue.full():
      _frame_queue.get_nowait()
    _frame_queue.put_nowait(frame)
  except queue.Full:
    pass
