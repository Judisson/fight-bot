import json
import os
import time
from pathlib import Path

import cv2

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_DEFENDER,
  ACAO_DESTREZA,
  ACAO_ESPERAR,
  nome_acao,
)
from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import obter_estado_luta
from src.ai.modo_treino import obter_acoes_permitidas, resolver_modo_treino
from src.ai.recompensa import PESO_KO, PESO_VITORIA, obter_recompensa
from src.bot.acoes_luta import executar_acao, obter_acao_em_execucao
from src.bot.vida import desenhar_info_vida, obter_info_vida
from src.utils.log import log
from src.utils.visao_debug import exibir

CAMINHO_LOG_TREINO = Path("data/treino_log.jsonl")


class Luta:

  def __init__(self, exibir_debug=True, modo_treino="treino"):
    self.exibir_debug = exibir_debug
    self.modo_treino = resolver_modo_treino(modo_treino)
    self.cerebro = CerebroIA(
      acoes_permitidas=obter_acoes_permitidas(self.modo_treino),
      modo_treino=self.modo_treino,
    )
    self.estado = "BUSCANDO_LUTA"
    self.episodio = 0
    self.vida_jogador_anterior = None
    self.vida_inimigo_anterior = None

    self._estado_anterior = None
    self._acao_anterior = None
    self._acao_recompensada_anterior = None
    self._ultima_recompensa = 0.0

    try:
      self._log_luta_cada = max(1, int(os.getenv("BOT_LOG_LUTA_CADA", "30")))
    except ValueError:
      self._log_luta_cada = 30
    try:
      self._debug_downscale = max(1, int(os.getenv("BOT_DEBUG_DOWNSCALE", "2")))
    except ValueError:
      self._debug_downscale = 2
    try:
      self._janela_dano_recente = max(1, int(os.getenv("BOT_DANO_RECENTE_FRAMES", "8")))
    except ValueError:
      self._janela_dano_recente = 8

    self._contador_frames = 0
    self._mensagem_evento_debug = ""
    self._mensagem_evento_debug_restante = 0
    self._frames_dano_recente = 0
    self._frames_esperando_consecutivos = 0

    self._iniciar_metricas_episodio()
    log(f"Modo de treino ativo: {self.modo_treino.value}")

  def processar_frame(self, frame):
    estado_luta = obter_estado_luta(frame)
    em_luta = estado_luta["em_luta"]

    terminal = None
    terminal_recompensa = None
    if estado_luta["nocaute"]:
      terminal = "ko"
      terminal_recompensa = PESO_KO
    elif estado_luta.get("vitoria"):
      terminal = "vitoria"
      terminal_recompensa = PESO_VITORIA

    if terminal is not None:
      if self._estado_anterior is not None and self._acao_anterior is not None:
        self.cerebro.aprender(
          self._estado_anterior,
          self._acao_anterior,
          terminal_recompensa,
          proximo_estado=None,
          terminal=True,
        )
      self._recompensa_total_episodio += terminal_recompensa
      self.episodio += 1
      self._registrar_log_episodio(terminal)
      log(f"Terminal detectado: {terminal.upper()} | episodio={self.episodio}")
      self._resetar_contexto_pos_episodio()
      if self.exibir_debug:
        exibir(self._preparar_frame_debug(frame))
      time.sleep(2)
      return {
        "terminal": terminal,
        "em_luta": False,
      }

    if not em_luta:
      self.cerebro.reset_observacao()
      self._estado_anterior = None
      self._acao_anterior = None
      self._acao_recompensada_anterior = None
      self._mensagem_evento_debug = ""
      self._mensagem_evento_debug_restante = 0
      self._frames_dano_recente = 0
      self._frames_esperando_consecutivos = 0
      if self.exibir_debug:
        exibir(self._preparar_frame_debug(frame))
      return {
        "terminal": None,
        "em_luta": False,
      }

    info_vida = obter_info_vida(frame)
    sinais_estado = self._detectar_sinais_estado(info_vida)
    estado_atual = self.cerebro.obter_estado(
      frame=frame,
      info_vida=info_vida,
      sinais=sinais_estado,
    )

    if self.estado != "LUTANDO":
      log("Luta comecou")
      self.estado = "LUTANDO"

    info_recompensa = {}
    if self._estado_anterior is not None and self._acao_anterior is not None:
      destreza_consecutiva = (
        self._acao_anterior == ACAO_DESTREZA
        and self._acao_recompensada_anterior == ACAO_DESTREZA
      )
      bloqueio_consecutivo = (
        self._acao_anterior == ACAO_DEFENDER
        and self._acao_recompensada_anterior == ACAO_DEFENDER
      )
      recompensa, _, info_recompensa = obter_recompensa(
        frame,
        acao_atual=self._acao_anterior,
        inimigo_atacando=bool(sinais_estado.get("inimigo_atacando")),
        vida_jogador_atual=info_vida["vida_jogador_pct"],
        vida_jogador_anterior=self.vida_jogador_anterior,
        vida_inimigo_atual=info_vida["vida_inimigo_pct"],
        vida_inimigo_anterior=self.vida_inimigo_anterior,
        colunas_escuras_jogador_finais=info_vida["colunas_escuras_jogador_finais"],
        colunas_escuras_inimigo_finais=info_vida["colunas_escuras_inimigo_finais"],
        nocaute_detectado=False,
        vitoria_detectada=False,
        destreza_consecutiva=destreza_consecutiva,
        bloqueio_consecutivo=bloqueio_consecutivo,
        modo_treino=self.modo_treino,
        tempo_parado_frames=(
          self._frames_esperando_consecutivos
          if self._acao_anterior == ACAO_ESPERAR else 0
        ),
      )
      self._acao_recompensada_anterior = self._acao_anterior
      self._ultima_recompensa = recompensa
      self._recompensa_total_episodio += recompensa

      if info_recompensa.get("destreza_tentada"):
        self._destreza_tentada_episodio += 1
      if info_recompensa.get("destreza_perfeita"):
        self._destreza_perfeita_episodio += 1
      if info_recompensa.get("destreza_errada"):
        self._destreza_errada_episodio += 1
      if info_recompensa.get("bloqueio_tentado"):
        self._bloqueio_tentado_episodio += 1
      if info_recompensa.get("aparar_perfeito"):
        self._aparar_perfeito_episodio += 1
      if info_recompensa.get("bloqueio_errado"):
        self._bloqueio_errado_episodio += 1
      if self._acao_anterior == ACAO_ATAQUE_LEVE:
        self._ataque_leve_tentado_episodio += 1
      if self._acao_anterior == ACAO_ATAQUE_MEDIO:
        self._ataque_medio_tentado_episodio += 1
      if info_recompensa.get("tomou_dano"):
        self._dano_tomado_episodio += float(info_recompensa.get("tomou_dano_delta", 0.0))
      if info_recompensa.get("causou_dano"):
        self._dano_inimigo_episodio += float(info_recompensa.get("dano_inimigo_delta", 0.0))

      self.cerebro.aprender(
        self._estado_anterior,
        self._acao_anterior,
        recompensa,
        proximo_estado=estado_atual,
        terminal=False,
      )
      self._atualizar_evento_debug(info_recompensa)

    acao = self.cerebro.escolher_acao(estado_atual)
    acao_executada = executar_acao(acao)
    acao_registrada = acao if acao_executada else obter_acao_em_execucao()

    self._estado_anterior = estado_atual
    self._acao_anterior = acao_registrada
    if acao_registrada == ACAO_ESPERAR:
      self._frames_esperando_consecutivos += 1
    else:
      self._frames_esperando_consecutivos = 0

    if self.exibir_debug:
      frame_debug = desenhar_info_vida(frame.copy(), info_vida)
      frame_debug = self._desenhar_evento_debug(frame_debug)
      frame_debug = self._desenhar_debug_rede(frame_debug)
      exibir(self._preparar_frame_debug(frame_debug))

    self._contador_frames += 1
    if (self._contador_frames % self._log_luta_cada) == 0:
      log("ACAO:", nome_acao(acao_registrada), f"({acao_registrada})", "RECOMPENSA:", self._ultima_recompensa)
      log(
        "VIDA_VOCE:",
        self._formatar_pct(info_vida["vida_jogador_pct"]),
        self._barra_vida_texto(info_vida["vida_jogador_pct"]),
      )
      log(
        "VIDA_INIMIGO:",
        self._formatar_pct(info_vida["vida_inimigo_pct"]),
        self._barra_vida_texto(info_vida["vida_inimigo_pct"]),
      )

    self.vida_jogador_anterior = info_vida["vida_jogador_pct"]
    self.vida_inimigo_anterior = info_vida["vida_inimigo_pct"]

    return {
      "terminal": None,
      "em_luta": True,
    }

  def _detectar_sinais_estado(self, info_vida):
    vida_jogador = info_vida.get("vida_jogador_pct")

    inimigo_atacando = False
    if (
      vida_jogador is not None
      and self.vida_jogador_anterior is not None
      and vida_jogador < self.vida_jogador_anterior
      and info_vida.get("colunas_escuras_jogador_finais", 0) >= 1
    ):
      inimigo_atacando = True
      self._frames_dano_recente = self._janela_dano_recente
    elif self._frames_dano_recente > 0:
      self._frames_dano_recente -= 1

    tomou_dano_recente = self._frames_dano_recente > 0

    return {
      "inimigo_atacando": inimigo_atacando,
      "tomou_dano_recente": tomou_dano_recente,
    }

  def _atualizar_evento_debug(self, info_recompensa):
    if info_recompensa.get("destreza_perfeita"):
      self._mensagem_evento_debug = "DESTREZA PERFEITA +6"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("aparar_perfeito"):
      self._mensagem_evento_debug = "APARAR PERFEITO +3"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("causou_dano"):
      self._mensagem_evento_debug = "DANO INIMIGO +"
      self._mensagem_evento_debug_restante = 16
      return
    if info_recompensa.get("tomou_dano"):
      self._mensagem_evento_debug = "TOMOU DANO -5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("destreza_longe_ruim"):
      self._mensagem_evento_debug = "DESTREZA LONGE"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("spam_destreza"):
      self._mensagem_evento_debug = "SPAM DESTREZA"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("spam_bloqueio"):
      self._mensagem_evento_debug = "SPAM BLOQUEIO"
      self._mensagem_evento_debug_restante = 20
      return

    if self._mensagem_evento_debug_restante > 0:
      self._mensagem_evento_debug_restante -= 1

  def _registrar_log_episodio(self, resultado):
    duracao = max(0.0, time.time() - self._episodio_inicio_ts)
    debug_rede = self.cerebro.obter_debug_rede()
    registro = {
      "modelo": "pixel_cnn_ppo",
      "fase": self.modo_treino.value,
      "episodio": self.episodio,
      "resultado": resultado,
      "recompensa_total": round(self._recompensa_total_episodio, 3),
      "duracao_segundos": round(duracao, 3),
      "tempo_sobrevivencia": round(duracao, 3),
      "destreza_tentada": int(self._destreza_tentada_episodio),
      "destreza_perfeita": int(self._destreza_perfeita_episodio),
      "destreza_errada": int(self._destreza_errada_episodio),
      "bloqueio_tentado": int(self._bloqueio_tentado_episodio),
      "aparar_perfeito": int(self._aparar_perfeito_episodio),
      "bloqueio_errado": int(self._bloqueio_errado_episodio),
      "ataque_leve_tentado": int(self._ataque_leve_tentado_episodio),
      "ataque_medio_tentado": int(self._ataque_medio_tentado_episodio),
      "dano_tomado": round(self._dano_tomado_episodio, 3),
      "dano_inimigo": round(self._dano_inimigo_episodio, 3),
      "ppo_updates": int(debug_rede.get("updates", 0)),
      "ppo_passos": int(debug_rede.get("passos", 0)),
    }
    try:
      CAMINHO_LOG_TREINO.parent.mkdir(parents=True, exist_ok=True)
      with CAMINHO_LOG_TREINO.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError:
      log("Falha ao registrar log de treino por episodio.")

  def _iniciar_metricas_episodio(self):
    self._episodio_inicio_ts = time.time()
    self._recompensa_total_episodio = 0.0
    self._destreza_tentada_episodio = 0
    self._destreza_perfeita_episodio = 0
    self._destreza_errada_episodio = 0
    self._bloqueio_tentado_episodio = 0
    self._aparar_perfeito_episodio = 0
    self._bloqueio_errado_episodio = 0
    self._ataque_leve_tentado_episodio = 0
    self._ataque_medio_tentado_episodio = 0
    self._dano_tomado_episodio = 0.0
    self._dano_inimigo_episodio = 0.0

  def _resetar_contexto_pos_episodio(self):
    self.estado = "BUSCANDO_LUTA"
    self.vida_jogador_anterior = None
    self.vida_inimigo_anterior = None
    self._estado_anterior = None
    self._acao_anterior = None
    self._acao_recompensada_anterior = None
    self._ultima_recompensa = 0.0
    self._contador_frames = 0
    self._mensagem_evento_debug = ""
    self._mensagem_evento_debug_restante = 0
    self._frames_dano_recente = 0
    self._frames_esperando_consecutivos = 0
    self.cerebro.reset_observacao()
    self._iniciar_metricas_episodio()

  def _desenhar_evento_debug(self, frame):
    if self._mensagem_evento_debug_restante <= 0 or not self._mensagem_evento_debug:
      return frame

    cv2.putText(
      frame,
      self._mensagem_evento_debug,
      (20, 108),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (0, 255, 255),
      2,
      cv2.LINE_AA,
    )
    return frame

  def _desenhar_debug_rede(self, frame):
    info = self.cerebro.obter_debug_rede()
    painel_largura = 390
    painel_altura = 250
    margem = 10
    x0 = max(margem, frame.shape[1] - painel_largura - margem)
    y0 = margem
    x1 = min(frame.shape[1] - margem, x0 + painel_largura)
    y1 = min(frame.shape[0] - margem, y0 + painel_altura)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 30), thickness=-1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), thickness=1)

    cv2.putText(
      frame,
      "PPO PIXEL DEBUG",
      (x0 + 10, y0 + 20),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (240, 240, 240),
      1,
      cv2.LINE_AA,
    )

    linha_y = y0 + 42
    cv2.putText(
      frame,
      f"acao: {info.get('acao_nome')} ({info.get('acao')})",
      (x0 + 10, linha_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.48,
      (0, 220, 255),
      1,
      cv2.LINE_AA,
    )
    linha_y += 20
    cv2.putText(
      frame,
      (
        f"V(s): {info.get('valor', 0.0):.3f} | "
        f"H: {info.get('entropia', 0.0):.3f}"
      ),
      (x0 + 10, linha_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.45,
      (180, 220, 180),
      1,
      cv2.LINE_AA,
    )
    linha_y += 18
    cv2.putText(
      frame,
      (
        f"updates: {info.get('updates', 0)} | passos: {info.get('passos', 0)} | "
        f"buffer: {info.get('buffer', 0)}/{info.get('rollout_size', 0)}"
      ),
      (x0 + 10, linha_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (190, 190, 190),
      1,
      cv2.LINE_AA,
    )

    linha_y += 20
    cv2.putText(
      frame,
      (
        f"loss pi={info.get('policy_loss', 0.0):.4f} "
        f"v={info.get('value_loss', 0.0):.4f}"
      ),
      (x0 + 10, linha_y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.42,
      (220, 200, 160),
      1,
      cv2.LINE_AA,
    )

    probs = info.get("probs", {})
    bar_x = x0 + 10
    bar_y = linha_y + 20
    bar_h = 16
    bar_w_max = max(40, painel_largura - 170)

    for acao, prob in sorted(probs.items(), key=lambda item: int(item[0])):
      nome = nome_acao(int(acao))
      cv2.putText(
        frame,
        f"{nome[:13]:<13}",
        (bar_x, bar_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
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
        (x_bar + bar_w_max + 6, bar_y + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
      )
      bar_y += 20
      if bar_y + bar_h >= y1 - 5:
        break

    return frame

  def _preparar_frame_debug(self, frame):
    if frame is None:
      return None
    if self._debug_downscale <= 1:
      return frame
    return frame[:: self._debug_downscale, :: self._debug_downscale].copy()

  @staticmethod
  def _formatar_pct(valor):
    if valor is None:
      return "?"
    return f"{valor:.1f}%"

  @staticmethod
  def _barra_vida_texto(valor, largura=20):
    if valor is None:
      return "?" * largura

    preenchidos = max(0, min(largura, int(round((valor / 100.0) * largura))))
    vazios = largura - preenchidos
    return ("#" * preenchidos) + ("-" * vazios)
