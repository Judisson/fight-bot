import cv2

NOME_JANELA = "Visao do Bot"

cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOME_JANELA, 960, 540)

mouse_x = 0
mouse_y = 0

def mouse_event(event, x, y, flags, param):
  global mouse_x, mouse_y
  mouse_x = x
  mouse_y = y

cv2.setMouseCallback(NOME_JANELA, mouse_event)

def exibir(frame):
  frame_reduzido = cv2.resize(frame, (960, 540))

  texto = f"X:{mouse_x} Y:{mouse_y}"

  cv2.putText(
    frame_reduzido,
    texto,
    (10,30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0,255,0),
    2
  )

  cv2.imshow(NOME_JANELA, frame_reduzido)

  # waitKey(1) para nao limitar o loop a ~30 FPS.
  if cv2.waitKey(1) == 27:
    raise KeyboardInterrupt("Encerrado pelo usuario (ESC).")
