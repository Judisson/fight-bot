import json
import os
import time
from pathlib import Path

import cv2

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENDER, ACAO_DESTREZA, ACAO_ESPERAR
from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import obter_estado_luta
from src.ai.modo_treino import obter_acoes_permitidas, resolver_modo_treino
from src.ai.recompensa import PESO_KO, PESO_VITORIA, obter_recompensa
from src.bot.acoes_luta import executar_acao, obter_acao_em_execucao
from src.bot.vida import desenhar_info_vida, obter_info_vida
from src.utils.log import log
from src.utils.visao_debug import exibir
from src.visao.personagens import RastreadorPersonagens, desenhar_info_personagens

CAMINHO_LOG_TREINO = Path("data/treino_log.jsonl")


class Luta:

  def __init__(self, exibir_debug=True, modo_treino="completo"):
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

    self.rastreador_personagens = RastreadorPersonagens(
      usar_yolo=True,
      caminho_modelo="src/modelos/personagens.pt",
      somente_yolo=True,
    )

    self._iniciar_metricas_episodio()
    log(f"Modo de treino ativo: {self.modo_treino.value}")

    # FASE 2 (PENDENTE): introduzir sinais de janela segura para combo.
    # FASE 3 (PENDENTE): voltar com postura ofensiva/defensiva e especial.
    # ETAPA 2 (PENDENTE): trocar Q-table por DQN mantendo este extrator.

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
      self.rastreador_personagens.reset()
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
    info_personagens = self.rastreador_personagens.detectar(frame)
    sinais_estado = self._detectar_sinais_estado(info_vida)
    estado_atual = self.cerebro.obter_estado(
      info_personagens=info_personagens,
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
      combo_apos_destreza = (
        self._acao_anterior == ACAO_COMBO_SEGURO
        and self._acao_recompensada_anterior == ACAO_DESTREZA
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
        combo_apos_destreza=combo_apos_destreza,
        distancia_px=info_personagens.get("distancia_px") if info_personagens else None,
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
      if info_recompensa.get("combo_tentado"):
        self._combo_tentado_episodio += 1
      if info_recompensa.get("combo_apos_destreza"):
        self._combo_apos_destreza_episodio += 1
      if info_recompensa.get("tomou_dano"):
        self._dano_tomado_episodio += float(info_recompensa.get("tomou_dano_delta", 0.0))

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
      frame_debug = desenhar_info_personagens(frame_debug, info_personagens)
      frame_debug = self._desenhar_evento_debug(frame_debug)
      exibir(self._preparar_frame_debug(frame_debug))

    self._contador_frames += 1
    if (self._contador_frames % self._log_luta_cada) == 0:
      log("ESTADO:", estado_atual)
      log("ACAO:", acao_registrada, "RECOMPENSA:", self._ultima_recompensa)
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
      self._mensagem_evento_debug = "APARAR PERFEITO +6"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("destreza_quase"):
      self._mensagem_evento_debug = "DESTREZA QUASE +2"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("bloqueio_quase"):
      self._mensagem_evento_debug = "BLOQUEIO QUASE +2"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("combo_apos_destreza"):
      self._mensagem_evento_debug = "COMBO POS DESTREZA +4"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("spam_destreza"):
      self._mensagem_evento_debug = "SPAM DESTREZA -0.5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("spam_bloqueio"):
      self._mensagem_evento_debug = "SPAM BLOQUEIO -0.5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("destreza_longe_ruim"):
      self._mensagem_evento_debug = "DESTREZA LONGE -1.5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("tomou_dano"):
      self._mensagem_evento_debug = "TOMOU DANO"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("nao_reagiu_ataque"):
      self._mensagem_evento_debug = "SEM REACAO -0.2"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("tentou_reagir_ataque"):
      self._mensagem_evento_debug = "TENTOU REAGIR +0.5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("sobreviveu_perigo"):
      self._mensagem_evento_debug = "SOBREVIVEU JANELA PERIGOSA +0.2"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("acao_destreza") and not info_recompensa.get("destreza_perfeita"):
      self._mensagem_evento_debug = "DESTREZA SEM PERIGO -1"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("acao_bloqueio") and not info_recompensa.get("aparar_perfeito"):
      self._mensagem_evento_debug = "BLOQUEIO SEM PERIGO -1"
      self._mensagem_evento_debug_restante = 20
      return

    if self._mensagem_evento_debug_restante > 0:
      self._mensagem_evento_debug_restante -= 1

  def _registrar_log_episodio(self, resultado):
    duracao = max(0.0, time.time() - self._episodio_inicio_ts)
    registro = {
      "etapa": 1,
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
      "combo_tentado": int(self._combo_tentado_episodio),
      "combo_apos_destreza": int(self._combo_apos_destreza_episodio),
      "dano_tomado": round(self._dano_tomado_episodio, 3),
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
    self._combo_tentado_episodio = 0
    self._combo_apos_destreza_episodio = 0
    self._dano_tomado_episodio = 0.0

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
    self.rastreador_personagens.reset()
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
      (255, 0, 255),
      2,
      cv2.LINE_AA,
    )
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
