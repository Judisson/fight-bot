import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

from src.utils.log import log


try:
  from ultralytics import YOLO
except ImportError:
  YOLO = None

COR_ROXO = (255, 0, 255)  # BGR
CAMINHO_MODELO_PADRAO = Path("src/modelos/personagens.pt")

# Pes por borda (fallback)
CHAO_Y_INICIO_REL = 0.62
CHAO_Y_FIM_REL = 0.90
CANNY_1 = 50
CANNY_2 = 150
SOBEL_LIMIAR = 35
LIMIAR_COLUNA_ATIVA_REL = 0.10
SEGMENTO_LARGURA_MIN = 5
SEGMENTO_DIST_MERGE = 70
SEGMENTO_PIXELS_MIN = 110

# Fallbacks
AREA_MINIMA_HEROI = 8000
ASPECTO_MINIMO = 1.2
LIMIAR_DIFF = 25


class RastreadorPersonagens:

  def __init__(self, usar_yolo=True, caminho_modelo=None, confianca=0.30, somente_yolo=True):
    self._frame_anterior = None
    self._ultimo_resultado = None

    self._modelo = None
    self._confianca = confianca
    self._somente_yolo = somente_yolo

    self._device = "cpu"
    self._usa_gpu = False
    self._nome_gpu = ""
    self._yolo_imgsz = int(os.getenv("BOT_YOLO_IMGSZ", "416"))
    self._yolo_assincrono = os.getenv("BOT_YOLO_ASYNC", "1").strip() == "1"
    self._fila_frames_yolo = queue.Queue(maxsize=1)
    self._lock_resultado_yolo = threading.Lock()
    self._resultado_yolo_recente = None
    self._stop_event_yolo = threading.Event()
    self._thread_yolo = None

    caminho = self._resolver_caminho_modelo(caminho_modelo)

    if usar_yolo and YOLO is not None:
      try:
        self._modelo = YOLO(str(caminho)).to("cuda")
        self._configurar_device_yolo()
        log(f"Rastreador personagens: YOLO carregado ({caminho})")
        log(
          "Rastreador personagens: YOLO em "
          f"{'GPU' if self._usa_gpu else 'CPU'} (device={self._device}, imgsz={self._yolo_imgsz})"
        )
      except Exception:
        self._modelo = None
        log(f"Rastreador personagens: falha ao carregar YOLO ({caminho}), seguindo sem ele.")
    elif usar_yolo:
      log("Rastreador personagens: ultralytics nao instalado, YOLO desativado.")

    if self._modelo is not None:
      log("Rastreador personagens: modo principal = YOLO treinado.")
      if self._yolo_assincrono:
        self._iniciar_worker_yolo()
        log("Rastreador personagens: YOLO assincrono ativo.")
    else:
      log("Rastreador personagens: modo principal = pes_edge (fallback).")

  @staticmethod
  def _resolver_caminho_modelo(caminho_modelo):
    caminho_env = os.getenv("BOT_YOLO_MODEL", "").strip()

    if caminho_modelo:
      return Path(caminho_modelo)
    if caminho_env:
      return Path(caminho_env)
    return CAMINHO_MODELO_PADRAO

  def _iniciar_worker_yolo(self):
    if self._thread_yolo is not None and self._thread_yolo.is_alive():
      return

    self._stop_event_yolo.clear()
    self._thread_yolo = threading.Thread(
      target=self._loop_yolo_assincrono,
      name="yolo-assincrono",
      daemon=True,
    )
    self._thread_yolo.start()

  def _enfileirar_frame_yolo(self, frame):
    if frame is None:
      return

    frame_copia = frame.copy()
    try:
      if self._fila_frames_yolo.full():
        self._fila_frames_yolo.get_nowait()
      self._fila_frames_yolo.put_nowait(frame_copia)
    except queue.Full:
      pass

  def _loop_yolo_assincrono(self):
    while not self._stop_event_yolo.is_set():
      try:
        frame = self._fila_frames_yolo.get(timeout=0.05)
      except queue.Empty:
        continue

      info_yolo = self._detectar_por_yolo_arena(frame)
      if info_yolo is None:
        continue

      with self._lock_resultado_yolo:
        self._resultado_yolo_recente = info_yolo

  def reset(self):
    self._frame_anterior = None
    self._ultimo_resultado = None
    with self._lock_resultado_yolo:
      self._resultado_yolo_recente = None
    while not self._fila_frames_yolo.empty():
      try:
        self._fila_frames_yolo.get_nowait()
      except queue.Empty:
        break

  def detectar(self, frame):
    if self._somente_yolo and self._modelo is None:
      return self._ultimo_resultado

    # YOLO sempre primeiro quando disponivel.
    if self._modelo is not None:
      if self._yolo_assincrono:
        self._enfileirar_frame_yolo(frame)
        with self._lock_resultado_yolo:
          info_yolo = self._resultado_yolo_recente
      else:
        info_yolo = self._detectar_por_yolo_arena(frame)

      if info_yolo is not None:
        self._ultimo_resultado = info_yolo
        return info_yolo
      if self._somente_yolo:
        return self._ultimo_resultado

    info_pes = self._detectar_por_pes_bordas(frame)
    if info_pes is not None:
      self._ultimo_resultado = info_pes
      return info_pes

    info_movimento = self._detectar_por_movimento(frame)
    if info_movimento is not None:
      self._ultimo_resultado = info_movimento
      return info_movimento

    return self._ultimo_resultado

  def _configurar_device_yolo(self):
    try:
      import torch
    except Exception:
      self._device = "cpu"
      self._usa_gpu = False
      self._nome_gpu = ""
      return

    if torch.cuda.is_available():
      self._device = 0
      self._usa_gpu = True
      try:
        self._nome_gpu = torch.cuda.get_device_name(0)
      except Exception:
        self._nome_gpu = ""
      return

    self._device = "cpu"
    self._usa_gpu = False
    self._nome_gpu = ""

  def obter_info_yolo(self):
    return {
      "modelo_carregado": self._modelo is not None,
      "usa_gpu": self._usa_gpu,
      "device": self._device,
      "nome_gpu": self._nome_gpu,
      "assincrono": self._yolo_assincrono,
      "imgsz": self._yolo_imgsz,
    }

  def encerrar(self):
    self._stop_event_yolo.set()
    if self._thread_yolo is not None:
      self._thread_yolo.join(timeout=1.0)

  def _recortar_faixa_chao(self, frame):
    altura = frame.shape[0]
    y_inicio = int(altura * CHAO_Y_INICIO_REL)
    y_fim = int(altura * CHAO_Y_FIM_REL)
    if y_fim <= y_inicio:
      return None, None, None

    roi_chao = frame[y_inicio:y_fim, :]
    return roi_chao, y_inicio, y_fim

  def _recortar_arena(self, frame):
    altura = frame.shape[0]
    y_offset = int(altura * 0.25)
    arena = frame[y_offset:, :]
    return arena, y_offset
  def _detectar_por_yolo_arena(self, frame):
    altura_frame, largura_frame, _ = frame.shape
    arena, y_offset = self._recortar_arena(frame)
    if arena.size == 0:
      return None

    try:
      resultados = self._modelo.predict(
        source=arena,
        conf=self._confianca,
        device=self._device,
        imgsz=self._yolo_imgsz,
        half=self._usa_gpu,
        max_det=10,
        verbose=False,
      )
    except Exception:
      return None

    candidatos = []

    for resultado in resultados:
      caixas = getattr(resultado, "boxes", None)
      if caixas is None:
        continue

      for box in caixas:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        conf = float(box.conf[0])
        candidatos.append((x1, y1 + y_offset, w, h, conf))

    info = self._montar_info_personagens(candidatos, largura_frame, altura_frame)
    if info is not None:
      info["fonte"] = "yolo"
    return info

  def _detectar_por_pes_bordas(self, frame):
    altura, largura, _ = frame.shape
    roi_chao, y_inicio, y_fim = self._recortar_faixa_chao(frame)
    if roi_chao is None or roi_chao.size == 0:
      return None

    gray = cv2.cvtColor(roi_chao, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, CANNY_1, CANNY_2)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    _, sobel_bin = cv2.threshold(sobel, SOBEL_LIMIAR, 255, cv2.THRESH_BINARY)

    mascara = cv2.bitwise_and(edges, sobel_bin)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=1)
    mascara = cv2.dilate(mascara, kernel, iterations=1)

    atividade_coluna = np.sum(mascara > 0, axis=0)
    limiar_coluna = max(4, int(roi_chao.shape[0] * LIMIAR_COLUNA_ATIVA_REL))
    colunas_ativas = atividade_coluna >= limiar_coluna

    segmentos = _extrair_segmentos_ativos(colunas_ativas, SEGMENTO_LARGURA_MIN)
    grupos = _agrupar_segmentos_pes(
      segmentos,
      mascara,
      y_inicio,
      SEGMENTO_PIXELS_MIN,
      SEGMENTO_DIST_MERGE,
    )

    if len(grupos) < 2:
      return None

    info = self._montar_info_por_pes(grupos, largura)
    if info is None:
      return None

    info["fonte"] = "pes_edge"
    info["debug_chao"] = (y_inicio, y_fim)
    info["candidatos_validos"] = len(grupos)
    return info

  def _montar_info_por_pes(self, grupos, largura):
    grupos_ordenados = sorted(grupos, key=lambda g: g["cx"])
    meio = largura // 2

    esquerda = [g for g in grupos_ordenados if g["cx"] < meio]
    direita = [g for g in grupos_ordenados if g["cx"] >= meio]

    if esquerda and direita:
      jogador = self._escolher_grupo_por_historico(esquerda, "esquerda")
      inimigo = self._escolher_grupo_por_historico(direita, "direita")
    else:
      if len(grupos_ordenados) < 2:
        return None
      jogador = grupos_ordenados[0]
      inimigo = grupos_ordenados[-1]

    cx_jogador = jogador["cx"]
    cy_jogador = jogador["cy"]
    cx_inimigo = inimigo["cx"]
    cy_inimigo = inimigo["cy"]

    distancia = abs(cx_inimigo - cx_jogador)
    distancia_norm = float(distancia / max(1, largura))

    return {
      "jogador_bbox": (jogador["x"], jogador["y"], jogador["w"], jogador["h"]),
      "inimigo_bbox": (inimigo["x"], inimigo["y"], inimigo["w"], inimigo["h"]),
      "jogador_centro": (cx_jogador, cy_jogador),
      "inimigo_centro": (cx_inimigo, cy_inimigo),
      "distancia_px": distancia,
      "distancia_norm": distancia_norm,
    }

  def _escolher_grupo_por_historico(self, grupos, lado):
    if self._ultimo_resultado is None:
      return max(grupos, key=lambda g: g["score"])

    centro_ref = (
      self._ultimo_resultado["jogador_centro"]
      if lado == "esquerda"
      else self._ultimo_resultado["inimigo_centro"]
    )

    melhor = None
    melhor_score = None

    for grupo in grupos:
      distancia = abs(grupo["cx"] - centro_ref[0]) + abs(grupo["cy"] - centro_ref[1])
      score = distancia - (grupo["score"] * 0.015)

      if melhor is None or score < melhor_score:
        melhor = grupo
        melhor_score = score

    return melhor

  def _detectar_por_movimento(self, frame):
    altura, largura, _ = frame.shape
    arena, y_offset = self._recortar_arena(frame)

    if self._frame_anterior is None or self._frame_anterior.shape != arena.shape:
      self._frame_anterior = arena.copy()
      return None

    diff = cv2.absdiff(arena, self._frame_anterior)
    self._frame_anterior = arena.copy()

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mascara = cv2.threshold(gray, LIMIAR_DIFF, 255, cv2.THRESH_BINARY)
    mascara = cv2.dilate(mascara, None, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for contorno in contornos:
      x, y, w, h = cv2.boundingRect(contorno)
      area = w * h
      if area < AREA_MINIMA_HEROI:
        continue

      y_global = y + y_offset
      candidatos.append((x, y_global, w, h, 0.7))

    info = self._montar_info_personagens(candidatos, largura, altura)
    if info is not None:
      info["fonte"] = "movimento"
    return info

  def _montar_info_personagens(self, candidatos, largura, altura):
    filtrados = self._filtrar_candidatos(candidatos, largura, altura)
    if len(filtrados) < 2:
      return None

    jogador, inimigo = self._selecionar_dupla(filtrados, largura)
    if jogador is None or inimigo is None:
      return None

    cx_jogador = jogador[5]
    cy_jogador = jogador[6]
    cx_inimigo = inimigo[5]
    cy_inimigo = inimigo[6]

    distancia = abs(cx_inimigo - cx_jogador)
    distancia_norm = float(distancia / max(1, largura))

    return {
      "jogador_bbox": (jogador[0], jogador[1], jogador[2], jogador[3]),
      "inimigo_bbox": (inimigo[0], inimigo[1], inimigo[2], inimigo[3]),
      "jogador_centro": (cx_jogador, cy_jogador),
      "inimigo_centro": (cx_inimigo, cy_inimigo),
      "distancia_px": distancia,
      "distancia_norm": distancia_norm,
      "candidatos_validos": len(filtrados),
    }

  def _filtrar_candidatos(self, candidatos, largura, altura):
    area_frame = largura * altura
    area_max = int(area_frame * 0.22)

    filtrados = []
    for x, y, w, h, conf in candidatos:
      area = w * h
      if area < AREA_MINIMA_HEROI or area > area_max:
        continue

      cx = x + w // 2
      cy = y + h // 2
      aspecto = h / max(1, w)

      if aspecto < ASPECTO_MINIMO or aspecto > 5.0:
        continue
      if cx < largura * 0.04 or cx > largura * 0.96:
        continue
      if cy < altura * 0.30 or cy > altura * 0.96:
        continue

      filtrados.append((x, y, w, h, conf, cx, cy))

    return filtrados

  def _selecionar_dupla(self, candidatos, largura):
    meio = largura // 2
    esquerda = [c for c in candidatos if c[5] < meio]
    direita = [c for c in candidatos if c[5] >= meio]

    if esquerda and direita:
      jogador = self._escolher_por_historico(esquerda, lado="esquerda")
      inimigo = self._escolher_por_historico(direita, lado="direita")
      return jogador, inimigo

    candidatos_ordenados = sorted(candidatos, key=lambda c: c[5])
    return candidatos_ordenados[0], candidatos_ordenados[-1]

  def _escolher_por_historico(self, candidatos, lado):
    if self._ultimo_resultado is None:
      return max(candidatos, key=lambda c: c[4])

    centro_ref = (
      self._ultimo_resultado["jogador_centro"]
      if lado == "esquerda"
      else self._ultimo_resultado["inimigo_centro"]
    )

    melhor = None
    melhor_score = None

    for candidato in candidatos:
      cx = candidato[5]
      cy = candidato[6]
      conf = candidato[4]

      distancia = abs(cx - centro_ref[0]) + abs(cy - centro_ref[1])
      score = distancia - (conf * 120)

      if melhor is None or score < melhor_score:
        melhor = candidato
        melhor_score = score

    return melhor


def _extrair_segmentos_ativos(colunas_ativas, largura_minima):
  segmentos = []
  inicio = None

  for i, ativo in enumerate(colunas_ativas):
    if ativo and inicio is None:
      inicio = i
    elif not ativo and inicio is not None:
      fim = i - 1
      if (fim - inicio + 1) >= largura_minima:
        segmentos.append((inicio, fim))
      inicio = None

  if inicio is not None:
    fim = len(colunas_ativas) - 1
    if (fim - inicio + 1) >= largura_minima:
      segmentos.append((inicio, fim))

  return segmentos


def _agrupar_segmentos_pes(segmentos, mascara, y_offset, pixels_minimos, merge_dist):
  grupos_base = []

  for x1, x2 in segmentos:
    faixa = mascara[:, x1 : x2 + 1]
    ys, _ = np.where(faixa > 0)

    if ys.size < pixels_minimos:
      continue

    y1 = int(ys.min()) + y_offset
    y2 = int(ys.max()) + y_offset

    grupos_base.append(
      {
        "x1": int(x1),
        "x2": int(x2),
        "y1": y1,
        "y2": y2,
        "score": int(ys.size),
      }
    )

  if not grupos_base:
    return []

  grupos_base.sort(key=lambda g: (g["x1"] + g["x2"]) // 2)

  grupos_unificados = []
  atual = grupos_base[0]

  for grupo in grupos_base[1:]:
    cx_atual = (atual["x1"] + atual["x2"]) // 2
    cx_novo = (grupo["x1"] + grupo["x2"]) // 2

    if abs(cx_novo - cx_atual) <= merge_dist:
      atual["x1"] = min(atual["x1"], grupo["x1"])
      atual["x2"] = max(atual["x2"], grupo["x2"])
      atual["y1"] = min(atual["y1"], grupo["y1"])
      atual["y2"] = max(atual["y2"], grupo["y2"])
      atual["score"] += grupo["score"]
    else:
      grupos_unificados.append(atual)
      atual = grupo

  grupos_unificados.append(atual)

  resultado = []
  for g in grupos_unificados:
    x = g["x1"]
    y = g["y1"]
    w = max(1, g["x2"] - g["x1"] + 1)
    h = max(1, g["y2"] - g["y1"] + 1)
    cx = x + w // 2
    cy = y + h // 2

    if h < 8:
      continue

    resultado.append(
      {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "score": g["score"],
      }
    )

  return resultado


def desenhar_info_personagens(frame, info_personagens):
  if not info_personagens:
    return frame

  x, y, w, h = info_personagens["jogador_bbox"]
  cv2.rectangle(frame, (x, y), (x + w, y + h), COR_ROXO, 2)

  x, y, w, h = info_personagens["inimigo_bbox"]
  cv2.rectangle(frame, (x, y), (x + w, y + h), COR_ROXO, 2)

  p1 = info_personagens["jogador_centro"]
  p2 = info_personagens["inimigo_centro"]
  cv2.line(frame, p1, p2, COR_ROXO, 2)

  mx = (p1[0] + p2[0]) // 2
  my = (p1[1] + p2[1]) // 2
  cv2.putText(
    frame,
    f"{info_personagens['distancia_px']}px",
    (mx - 32, my - 12),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    COR_ROXO,
    2,
    cv2.LINE_AA,
  )

  if "debug_chao" in info_personagens:
    y1, y2 = info_personagens["debug_chao"]
    cv2.line(frame, (0, y1), (frame.shape[1] - 1, y1), COR_ROXO, 1)
    cv2.line(frame, (0, y2), (frame.shape[1] - 1, y2), COR_ROXO, 1)

  fonte = info_personagens.get("fonte", "?")
  candidatos = info_personagens.get("candidatos_validos", 0)
  cv2.putText(
    frame,
    f"Rastreador: {fonte} | validos={candidatos}",
    (20, 82),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (255, 255, 255),
    1,
    cv2.LINE_AA,
  )

  return frame







