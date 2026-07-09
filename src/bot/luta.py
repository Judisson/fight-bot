import os
import time
import cv2
import numpy as np

from src.bot.acoes_luta import (
  executar_acao,
  limpar_fila_acoes,
  obter_acao_ativa,
  ACAO_ESPERAR,
  ACAO_ESQUIVA,
  ACAO_BLOQUEIO,
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_ESPECIAL,
  ACAO_BLOQUEIO_3S,
  nome_acao,
)
from src.bot.poder import obter_info_poder
from src.bot.vida import obter_info_vida
from src.bot.deteccao_luta import obter_estado_luta
from src.visao.hud_debug import desenhar_info_hud
from src.utils.visao_debug import exibir
from src.utils.log import log


class Luta:

  def __init__(self, exibir_debug=True, modo_treino="treino"):
    self.exibir_debug = exibir_debug
    self.rastreador = None
    
    if self.exibir_debug:
      from src.visao.personagens import RastreadorPersonagens
      self.rastreador = RastreadorPersonagens(
        usar_yolo=True,
        caminho_modelo="src/modelos/personagens.pt",
        somente_yolo=True,
      )

    self._slow_motion = False
    self._debug_downscale = 2
    try:
      self._debug_downscale = max(1, int(os.getenv("BOT_DEBUG_DOWNSCALE", "2")))
    except ValueError:
      self._debug_downscale = 2

    # Variáveis de Estado
    self.estado = "BUSCANDO_LUTA"
    self.episodio = 0
    self.resetar()

  def resetar(self):
    self._acoes_combo = []
    self._proxima_acao_ts = 0.0
    self._bloqueio_especial_ate = 0.0
    self._inimigo_tem_especial_confirmado = False
    self._frames_especial_inimigo_ativo = 0
    self._frames_especial_inimigo_inativo = 0
    self._acao_anterior = ACAO_ESPERAR
    self._ultima_recompensa = 0.0
    self.vida_jogador_anterior = None
    limpar_fila_acoes()

  def processar_frame(self, frame):
    # 1. Detectar o estado da luta
    estado_luta = obter_estado_luta(frame)
    em_luta = estado_luta["em_luta"]

    terminal = None
    if estado_luta["nocaute"]:
      terminal = "ko"
    elif estado_luta.get("vitoria"):
      terminal = "vitoria"

    info_vida = obter_info_vida(frame)
    vida_jogador = info_vida.get("vida_jogador_pct")

    if terminal is not None:
      log(f"Terminal detectado: {terminal.upper()} | episodio={self.episodio}")
      self.episodio += 1
      self.resetar()
      if self.exibir_debug:
        self._exibir_telas_debug(frame, info_vida=info_vida)
      time.sleep(2)
      return {
        "terminal": terminal,
        "em_luta": False,
      }

    if not em_luta:
      self.resetar()
      if self.exibir_debug:
        self._exibir_telas_debug(frame, info_vida=info_vida)
      return {
        "terminal": None,
        "em_luta": False,
      }

    # 2. Obter informações visuais
    distancia_px = None
    info_personagens = None
    if self.rastreador is not None:
      info_personagens = self.rastreador.detectar(frame)
      if info_personagens:
        distancia_px = info_personagens.get("distancia_px")

    info_poder = obter_info_poder(frame, log_segmentos=False)
    tem_especial_jogador = bool(info_poder.get("tem_especial_jogador"))
    tem_especial_inimigo = bool(info_poder.get("tem_especial_inimigo"))
    nivel_especial_inimigo = int(info_poder.get("nivel_especial_inimigo", 0))

    # 3. Detectar se o oponente usou/soltou especial com debounce
    if tem_especial_inimigo:
      self._frames_especial_inimigo_ativo += 1
      self._frames_especial_inimigo_inativo = 0
      if self._frames_especial_inimigo_ativo >= 5:
        if not self._inimigo_tem_especial_confirmado:
          self._inimigo_tem_especial_confirmado = True
          log("[ESPECIAL] Inimigo TEM especial (confirmado)")
    else:
      self._frames_especial_inimigo_inativo += 1
      self._frames_especial_inimigo_ativo = 0
      if self._frames_especial_inimigo_inativo >= 5:
        if self._inimigo_tem_especial_confirmado:
          self._inimigo_tem_especial_confirmado = False
          log("[ESPECIAL DETECTADO] Inimigo soltou especial! Entrando em defesa absoluta.")
          self._acoes_combo = []
          limpar_fila_acoes()
          self._bloqueio_especial_ate = time.time() + 3.0
          executar_acao(ACAO_BLOQUEIO_3S)
          self._acao_anterior = ACAO_BLOQUEIO_3S

    # 4. Se estiver sob defesa de especial inimigo
    agora = time.time()
    if agora < self._bloqueio_especial_ate:
      if vida_jogador is not None:
        self.vida_jogador_anterior = vida_jogador
      if self.exibir_debug:
        self._exibir_telas_debug(
          frame,
          info_vida=info_vida,
          info_poder=info_poder,
          info_personagens=info_personagens
        )
      return {
        "terminal": None,
        "em_luta": True,
      }

    # 5. Detectar se acabamos de apanhar
    if (
      vida_jogador is not None
      and self.vida_jogador_anterior is not None
      and vida_jogador < self.vida_jogador_anterior
    ):
      log(f"[DANOTOMADO] Apanhamos! Vida caiu de {self.vida_jogador_anterior:.1f}% para {vida_jogador:.1f}%. Reposicionando com Destreza, Destreza.")
      self._acoes_combo = [
        (ACAO_ESQUIVA, 0.300),
        (ACAO_ESQUIVA, 0.300),
      ]
      limpar_fila_acoes()
      acao_para_executar, delay = self._acoes_combo.pop(0)
      executar_acao(acao_para_executar)
      self._acao_anterior = acao_para_executar
      self._proxima_acao_ts = agora + delay

    # 6. Processar fila de combos
    if self._acoes_combo:
      if agora >= self._proxima_acao_ts:
        acao_para_executar, delay = self._acoes_combo.pop(0)
        executar_acao(acao_para_executar)
        self._acao_anterior = acao_para_executar
        self._proxima_acao_ts = agora + delay
    else:
      # Se a fila de combos estiver vazia, decidimos qual combo programar
      # if tem_especial_jogador:
      #   ...
      if distancia_px is None or distancia_px > 500:
        log(f"[COMBO DISTANCIA > 250] dist={distancia_px}px. Iniciando M, L, L, L, D, B")
        self._acoes_combo = [
          (ACAO_ATAQUE_MEDIO, 0.604),
          (ACAO_ATAQUE_LEVE, 0.616),
          (ACAO_ATAQUE_LEVE, 0.620),
          (ACAO_ATAQUE_LEVE, 0.633),
          (ACAO_ESQUIVA, 0.740), # D(300ms)
          (ACAO_BLOQUEIO, 0.506), # B(300ms)
        ]
      else:
        # Se distância for <= 300px:
        # L(150ms), M(200ms), L(150ms), L(150ms), D(300ms), B(300ms)
        log(f"[COMBO DISTANCIA <= 250] dist={distancia_px}px. Iniciando L, M, L, L, D, B")
        self._acoes_combo = [
          (ACAO_ATAQUE_LEVE, 0.634),
          (ACAO_ATAQUE_MEDIO, 0.590),
          (ACAO_ATAQUE_LEVE, 0.620),
          (ACAO_ATAQUE_LEVE, 0.633),
          (ACAO_ESQUIVA, 0.740), # D(300ms)
          (ACAO_BLOQUEIO, 0.506), # B(300ms)
        ]

      # Dispara a primeira ação do combo imediatamente
      if self._acoes_combo:
        acao_para_executar, delay = self._acoes_combo.pop(0)
        executar_acao(acao_para_executar)
        self._acao_anterior = acao_para_executar
        self._proxima_acao_ts = agora + delay

    if vida_jogador is not None:
      self.vida_jogador_anterior = vida_jogador

    if self.exibir_debug:
      self._exibir_telas_debug(
        frame,
        info_vida=info_vida,
        info_poder=info_poder,
        info_personagens=info_personagens
      )

    if self._slow_motion:
      time.sleep(0.15)

    return {
      "terminal": None,
      "em_luta": True,
    }

  def _desenhar_evento_debug(self, frame):
    # Se estiver em bloqueio especial, desenha mensagem
    if time.time() < self._bloqueio_especial_ate:
      cv2.putText(
        frame,
        "DEFESA ESPECIAL ATIVA (3s)",
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
      )
    return frame

  def _exibir_telas_debug(self, frame, info_vida=None, info_poder=None, info_personagens=None):
    frame_desenho = frame.copy()
    frame_desenho = desenhar_info_hud(
      frame_desenho,
      info_vida=info_vida,
      info_poder=info_poder,
      info_personagens=info_personagens,
      acao_atual=self._acao_anterior,
      slow_motion=self._slow_motion,
    )
    frame_desenho = self._desenhar_evento_debug(frame_desenho)
    frame_desenho = self._preparar_frame_debug(frame_desenho)

    tecla_debug = exibir(frame_desenho)
    self._processar_tecla_debug(tecla_debug)

  def _processar_tecla_debug(self, tecla_debug):
    if not tecla_debug:
      return
    tecla_char = str(tecla_debug).lower()
    if tecla_char == "l":
      self._slow_motion = not self._slow_motion
      log(f"[DEBUG] Camera lenta: {'ATIVA' if self._slow_motion else 'DESATIVADA'}")

  def _preparar_frame_debug(self, frame):
    if frame is None:
      return None
    if self._debug_downscale <= 1:
      return frame
    return frame[:: self._debug_downscale, :: self._debug_downscale].copy()
