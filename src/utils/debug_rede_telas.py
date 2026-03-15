import cv2
import numpy as np

from src.ai.acoes import nome_acao


def gerar_tela_perceptron(cerebro, titulo="PPO PIXEL PERCEPTRON", linha_status="", tecla_snapshot="P"):
  info = cerebro.obter_debug_rede()
  frame = np.zeros((500, 540, 3), dtype=np.uint8)
  h, w = frame.shape[:2]

  cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (120, 120, 120), thickness=1)
  cv2.putText(
    frame,
    titulo,
    (12, 24),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.62,
    (240, 240, 240),
    1,
    cv2.LINE_AA,
  )
  cv2.putText(
    frame,
    f"snapshot={str(tecla_snapshot).upper()}",
    (w - 170, 24),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.48,
    (200, 200, 200),
    1,
    cv2.LINE_AA,
  )

  linha_y = 54
  if linha_status:
    cv2.putText(
      frame,
      linha_status,
      (12, linha_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.46,
      (210, 210, 210),
      1,
      cv2.LINE_AA,
    )
    linha_y += 24

  cv2.putText(
    frame,
    f"acao: {info.get('acao_nome')} ({info.get('acao')})",
    (12, linha_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.50,
    (0, 220, 255),
    1,
    cv2.LINE_AA,
  )
  linha_y += 24

  cv2.putText(
    frame,
    f"V(s): {info.get('valor', 0.0):.3f} | H: {info.get('entropia', 0.0):.3f}",
    (12, linha_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.46,
    (180, 220, 180),
    1,
    cv2.LINE_AA,
  )
  linha_y += 20

  cv2.putText(
    frame,
    (
      f"updates: {info.get('updates', 0)} | passos: {info.get('passos', 0)} | "
      f"buffer: {info.get('buffer', 0)}/{info.get('rollout_size', 0)}"
    ),
    (12, linha_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.42,
    (190, 190, 190),
    1,
    cv2.LINE_AA,
  )
  linha_y += 20

  cv2.putText(
    frame,
    f"loss pi={info.get('policy_loss', 0.0):.4f} | v={info.get('value_loss', 0.0):.4f}",
    (12, linha_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.42,
    (220, 200, 160),
    1,
    cv2.LINE_AA,
  )
  linha_y += 20

  cv2.putText(
    frame,
    f"movimento={info.get('movimento_score', 0.0):.4f}",
    (12, linha_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.42,
    (120, 220, 255),
    1,
    cv2.LINE_AA,
  )

  probs = info.get("probs", {})
  bar_x = 12
  bar_y = linha_y + 16
  bar_h = 14
  bar_w_max = 250

  for acao, prob in sorted(probs.items(), key=lambda item: int(item[0])):
    nome = nome_acao(int(acao))
    cv2.putText(
      frame,
      f"{nome[:13]:<13}",
      (bar_x, bar_y + 11),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.38,
      (230, 230, 230),
      1,
      cv2.LINE_AA,
    )
    x_bar = bar_x + 130
    largura = int(max(0.0, min(1.0, float(prob))) * bar_w_max)
    cv2.rectangle(frame, (x_bar, bar_y), (x_bar + bar_w_max, bar_y + bar_h), (60, 60, 60), thickness=-1)
    cv2.rectangle(frame, (x_bar, bar_y), (x_bar + largura, bar_y + bar_h), (0, 180, 255), thickness=-1)
    cv2.putText(
      frame,
      f"{float(prob) * 100.0:5.1f}%",
      (x_bar + bar_w_max + 6, bar_y + 11),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.35,
      (220, 220, 220),
      1,
      cv2.LINE_AA,
    )
    bar_y += 18
    if bar_y + bar_h >= h - 84:
      break

  return frame


def _to_bgr_escala(frame_gray, largura, altura):
  img = np.clip(frame_gray * 255.0, 0, 255).astype(np.uint8)
  img = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_NEAREST)
  return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def gerar_tela_pixel(cerebro, titulo="PPO PIXEL INPUT (128x128 x4)"):
  debug_pixel = cerebro.obter_debug_pixel()
  stack = debug_pixel.get("stack")
  diff = debug_pixel.get("frame_diff")

  frame = np.zeros((500, 760, 3), dtype=np.uint8)
  cv2.rectangle(frame, (0, 0), (759, 499), (120, 120, 120), thickness=1)
  cv2.putText(
    frame,
    titulo,
    (12, 24),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.62,
    (240, 240, 240),
    1,
    cv2.LINE_AA,
  )

  if not isinstance(stack, np.ndarray) or stack.shape != (4, 128, 128):
    cv2.putText(
      frame,
      "Sem stack de pixel ainda.",
      (12, 56),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.52,
      (210, 210, 210),
      1,
      cv2.LINE_AA,
    )
    return frame

  tam = 190
  posicoes = [
    (12, 40, "t-3"),
    (212, 40, "t-2"),
    (12, 240, "t-1"),
    (212, 240, "t"),
  ]
  for idx, (x, y, label) in enumerate(posicoes):
    tile = _to_bgr_escala(stack[idx], tam, tam)
    frame[y : y + tam, x : x + tam] = tile
    cv2.putText(
      frame,
      label,
      (x + 4, y + 16),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.46,
      (0, 255, 255),
      1,
      cv2.LINE_AA,
    )

  if isinstance(diff, np.ndarray) and diff.size > 0:
    try:
      diff_uint8 = diff.astype(np.uint8)
      diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_TURBO)
      diff_color = cv2.resize(diff_color, (300, 300), interpolation=cv2.INTER_AREA)
      x = 440
      y = 110
      frame[y : y + 300, x : x + 300] = diff_color
      cv2.putText(
        frame,
        "frame_diff (movimento)",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
      )
    except Exception:
      pass

  return frame
