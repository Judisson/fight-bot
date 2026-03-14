import time

import cv2

from src.bot.captura import capturar_tela
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.rois import ROIS_PADRAO, obter_env_roi, obter_roi, obter_template_roi
from src.bot.visao import encontrar_template
from src.utils.log import log

TITULO_JOGO = "Champions"
NOME_JANELA = "Calibracao ROI (Permanente)"


def _obter_monitor_jogo():
  log("Procurando janela do jogo...")
  janela = None

  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(0.5)

  log("Janela encontrada!")

  if focar_jogo(TITULO_JOGO):
    log("Janela focada automaticamente.")
    time.sleep(0.5)
  else:
    log("Nao foi possivel focar automaticamente a janela.")

  return obter_bbox_janela(janela)


def _clamp_roi(roi, largura, altura):
  x1, y1, x2, y2 = roi
  x1 = max(0, min(largura - 1, x1))
  x2 = max(1, min(largura, x2))
  y1 = max(0, min(altura - 1, y1))
  y2 = max(1, min(altura, y2))

  if x2 <= x1:
    x2 = min(largura, x1 + 1)
  if y2 <= y1:
    y2 = min(altura, y1 + 1)

  return (x1, y1, x2, y2)


def _aplicar_tecla_config(tecla, roi, passo):
  x1, y1, x2, y2 = roi

  if tecla == ord("q"):
    y1 -= passo
  elif tecla == ord("a"):
    y1 += passo
  elif tecla == ord("w"):
    y2 -= passo
  elif tecla == ord("s"):
    y2 += passo
  elif tecla == ord("e"):
    x1 -= passo
  elif tecla == ord("d"):
    x1 += passo
  elif tecla == ord("r"):
    x2 -= passo
  elif tecla == ord("f"):
    x2 += passo
  else:
    return roi, False

  return (x1, y1, x2, y2), True


def iniciar_calibracao_roi_jogar_novamente():
  monitor = _obter_monitor_jogo()

  alvos = [nome for nome in ROIS_PADRAO.keys() if obter_template_roi(nome)]
  if not alvos:
    log("Nenhum alvo de ROI configurado para calibracao.")
    return

  rois = {nome: obter_roi(nome) for nome in alvos}
  indice_alvo = 0
  passo = 5

  log("Calibracao ROI permanente iniciada.")
  log("N/P troca alvo | Q/A y1 | W/S y2 | E/D x1 | R/F x2")
  log("1 passo=1 | 5 passo=5 | 9 passo=20 | C copiar alvo | V listar todos | ESC sair")

  cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(NOME_JANELA, 1280, 720)

  while True:
    frame = capturar_tela(monitor)
    altura, largura = frame.shape[:2]

    alvo = alvos[indice_alvo]
    template_atual = obter_template_roi(alvo)
    env_atual = obter_env_roi(alvo)

    roi = _clamp_roi(rois[alvo], largura, altura)
    rois[alvo] = roi
    x1, y1, x2, y2 = roi

    deteccao = encontrar_template(frame, template_atual, limiar=0.8, roi=roi)

    cor_roi = (0, 255, 255)
    if deteccao is not None:
      cor_roi = (0, 255, 0)
      cv2.circle(frame, (deteccao["x"], deteccao["y"]), 8, (0, 255, 0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_roi, 2)

    cv2.putText(
      frame,
      f"Alvo: {alvo} | Env: {env_atual}",
      (20, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.62,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
    )
    cv2.putText(
      frame,
      f"ROI (x1,y1,x2,y2): {x1},{y1},{x2},{y2}",
      (20, 58),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.58,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )
    cv2.putText(
      frame,
      f"Tam: {x2 - x1}x{y2 - y1} | passo={passo}",
      (20, 84),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )

    if deteccao is not None:
      cv2.putText(
        frame,
        f"Match: conf={deteccao['confianca']:.3f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
      )
    else:
      cv2.putText(
        frame,
        "Match: nao encontrado no ROI",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
      )

    cv2.putText(
      frame,
      "N/P alvo | Q/A y1 W/S y2 E/D x1 R/F x2 | 1/5/9 passo | C copiar | V listar | ESC sair",
      (20, frame.shape[0] - 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
    )

    cv2.imshow(NOME_JANELA, frame)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
      break
    if tecla == ord("n"):
      indice_alvo = (indice_alvo + 1) % len(alvos)
      continue
    if tecla == ord("p"):
      indice_alvo = (indice_alvo - 1) % len(alvos)
      continue
    if tecla == ord("1"):
      passo = 1
      continue
    if tecla == ord("5"):
      passo = 5
      continue
    if tecla == ord("9"):
      passo = 20
      continue
    if tecla == ord("c"):
      log(f"{env_atual}={x1},{y1},{x2},{y2}")
      continue
    if tecla == ord("v"):
      for nome in alvos:
        rx1, ry1, rx2, ry2 = rois[nome]
        log(f"{obter_env_roi(nome)}={rx1},{ry1},{rx2},{ry2}")
      continue

    novo_roi, alterou = _aplicar_tecla_config(tecla, roi, passo)
    if alterou:
      rois[alvo] = _clamp_roi(novo_roi, largura, altura)

  cv2.destroyWindow(NOME_JANELA)
