import threading

import cv2

NOME_JANELA = "Visao do Bot"
NOME_JANELA_PIXEL = "Debug Pixel"
NOME_JANELA_PERCEPTRON = "Debug Perceptron"

_janelas_iniciadas = {}
_encerrar_solicitado = False
_metricas_lock = threading.Lock()
_metricas = {
  "fps_real": 0.0,
  "fps_alvo": 0,
  "proc_ms": 0.0,
  "sleep_ms": 0.0,
}


def _obter_tamanho_janela(nome_janela):
  if nome_janela == NOME_JANELA:
    return (960, 540)
  if nome_janela == NOME_JANELA_PIXEL:
    return (760, 520)
  if nome_janela == NOME_JANELA_PERCEPTRON:
    return (560, 520)
  return (960, 540)


def _iniciar_janela(nome_janela):
  if _janelas_iniciadas.get(nome_janela):
    return

  cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
  largura, altura = _obter_tamanho_janela(nome_janela)
  cv2.resizeWindow(nome_janela, int(largura), int(altura))
  _janelas_iniciadas[nome_janela] = True


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


def exibir_multiplas(janelas, overlay_metricas_em=None):
  global _encerrar_solicitado
  if _encerrar_solicitado:
    raise KeyboardInterrupt("Encerrado pelo usuario (ESC).")

  if not janelas:
    return None

  if overlay_metricas_em is None:
    overlay_metricas_em = set()
  else:
    overlay_metricas_em = set(overlay_metricas_em)

  for nome_janela, frame in janelas.items():
    if frame is None:
      continue
    _iniciar_janela(nome_janela)
    frame_exibir = frame.copy()
    if nome_janela in overlay_metricas_em:
      _desenhar_overlay(frame_exibir)
    cv2.imshow(nome_janela, frame_exibir)

  tecla = cv2.waitKey(1) & 0xFF
  if tecla == 27:
    _encerrar_solicitado = True
    raise KeyboardInterrupt("Encerrado pelo usuario (ESC).")
  if tecla == 255:
    return None

  try:
    return chr(tecla).lower()
  except Exception:
    return None


def exibir(frame):
  if frame is None:
    return None

  return exibir_multiplas(
    {
      NOME_JANELA: frame,
    },
    overlay_metricas_em={NOME_JANELA},
  )
