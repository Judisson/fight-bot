import json
import os
import time
from pathlib import Path

import cv2

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_BLOQUEIO,
  ACAO_ESPERAR,
  ACAO_ESQUIVA,
  nome_acao,
)
from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import obter_estado_luta
from src.ai.modo_treino import obter_acoes_permitidas, resolver_modo_treino
from src.ai.recompensa import (
  PESO_KO,
  PESO_VITORIA,
  obter_recompensa,
)
from src.bot.acoes_luta import executar_acao
from src.bot.poder import obter_info_poder
from src.bot.vida import desenhar_info_vida, obter_info_vida
from src.utils.debug_rede_telas import gerar_tela_perceptron, gerar_tela_pixel
from src.utils.log import log
from src.utils.visao_debug import (
  NOME_JANELA,
  NOME_JANELA_PERCEPTRON,
  NOME_JANELA_PIXEL,
  exibir_multiplas,
)

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
    try:
      self._qtd_filtros_snapshot = max(1, int(os.getenv("BOT_DEBUG_SNAPSHOT_FILTROS", "3")))
    except ValueError:
      self._qtd_filtros_snapshot = 3
    try:
      self._log_reward_rt_cada = max(1, int(os.getenv("BOT_LOG_REWARD_RT_CADA", "1")))
    except ValueError:
      self._log_reward_rt_cada = 1
    self._frames_por_tempo = self._ler_int_env("BOT_FRAMES_POR_TEMPO", 4)
    self._janela_punicao_sem_decisao_tempos = self._ler_int_env(
      "BOT_PUNICAO_SEM_DECISAO_TEMPOS",
      4,
    )
    self._janela_pontuacao_combo_tempos = self._ler_int_env(
      "BOT_PONTUACAO_COMBO_TEMPOS",
      2,
    )
    self._janela_punicao_sem_decisao_frames = (
      self._janela_punicao_sem_decisao_tempos * self._frames_por_tempo
    )
    self._janela_pontuacao_combo_frames = (
      self._janela_pontuacao_combo_tempos * self._frames_por_tempo
    )
    # Compatibilidade com envs legadas em frames.
    valor_legacy_punicao = os.getenv("BOT_PUNICAO_SEM_DECISAO_FRAMES", "").strip()
    if valor_legacy_punicao:
      self._janela_punicao_sem_decisao_frames = self._ler_int_env(
        "BOT_PUNICAO_SEM_DECISAO_FRAMES",
        self._janela_punicao_sem_decisao_frames,
      )
    valor_legacy_combo = os.getenv("BOT_PONTUACAO_COMBO_FRAMES", "").strip()
    if valor_legacy_combo:
      self._janela_pontuacao_combo_frames = self._ler_int_env(
        "BOT_PONTUACAO_COMBO_FRAMES",
        self._janela_pontuacao_combo_frames,
      )
    self._log_reward_rt = os.getenv("BOT_LOG_REWARD_RT", "1").strip() == "1"
    tecla_snapshot = (os.getenv("BOT_DEBUG_SNAPSHOT_KEY", "p").strip() or "p")[0]
    self._tecla_snapshot_filtro = tecla_snapshot.lower()

    self._contador_frames = 0
    self._mensagem_evento_debug = ""
    self._mensagem_evento_debug_restante = 0
    self._frames_dano_recente = 0
    self._frames_esperando_consecutivos = 0
    self._frames_ameaca_sem_decisao = 0
    self._frames_combo_intervalo = 0
    self._ts_frame_anterior = None
    self._contador_reward_rt = 0
    self._espera_sem_pause_logada = False
    self._acoes_fila_cheia_episodio = 0
    self._ultimo_status_especial = None

    self._iniciar_metricas_episodio()
    log(f"Modo de treino ativo: {self.modo_treino.value}")
    if self._log_reward_rt:
      log(f"Logs de recompensa em tempo real ativos (cada={self._log_reward_rt_cada} frame).")
    log(
      "Punicao por sem decisao em ameaca:",
      (
        f"janela={self._janela_punicao_sem_decisao_tempos} tempos "
        f"({self._janela_punicao_sem_decisao_frames} frames)"
      ),
    )
    log(
      "Pontuacao de combo:",
      (
        f"janela={self._janela_pontuacao_combo_tempos} tempos "
        f"({self._janela_pontuacao_combo_frames} frames) (+5/-1)"
      ),
    )
    if self.exibir_debug:
      log(
        f"Debug filtro ativo. Pressione '{self._tecla_snapshot_filtro.upper()}' "
        "na janela de debug para imprimir snapshot no console."
      )

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
        self._exibir_telas_debug(frame, info_vida=None, status_perceptron=f"estado={terminal}")
      time.sleep(2)
      return {
        "terminal": terminal,
        "em_luta": False,
      }

    if not em_luta:
      if not self._espera_sem_pause_logada:
        self._espera_sem_pause_logada = True
      self.cerebro.reset_observacao()
      self._estado_anterior = None
      self._acao_anterior = None
      self._acao_recompensada_anterior = None
      self._mensagem_evento_debug = ""
      self._mensagem_evento_debug_restante = 0
      self._frames_dano_recente = 0
      self._frames_esperando_consecutivos = 0
      self._frames_ameaca_sem_decisao = 0
      self._frames_combo_intervalo = 0
      self._ts_frame_anterior = None
      self._contador_reward_rt = 0
      if self.exibir_debug:
        self._exibir_telas_debug(frame, info_vida=None, status_perceptron="estado=esperando")
      return {
        "terminal": None,
        "em_luta": False,
      }

    delta_tempo_seg = self._obter_delta_tempo_seg()
    info_vida = obter_info_vida(frame)
    self._registrar_log_especial(frame)
    sinais_estado = self._detectar_sinais_estado(info_vida)
    estado_atual = self.cerebro.obter_estado(
      frame=frame,
      info_vida=info_vida,
      sinais=sinais_estado,
    )

    if self._espera_sem_pause_logada:
      log("[LUTA] pausar-luta-button detectado -> retomando observacao e decisao da IA.")
      self._espera_sem_pause_logada = False

    if self.estado != "LUTANDO":
      log("Luta comecou")
      self.estado = "LUTANDO"

    info_recompensa = {}
    if self._estado_anterior is not None and self._acao_anterior is not None:
      destreza_consecutiva = (
        self._acao_anterior == ACAO_ESQUIVA
        and self._acao_recompensada_anterior == ACAO_ESQUIVA
      )
      bloqueio_consecutivo = (
        self._acao_anterior == ACAO_BLOQUEIO
        and self._acao_recompensada_anterior == ACAO_BLOQUEIO
      )
      inimigo_atacando_agora = bool(sinais_estado.get("inimigo_atacando"))
      aplicar_punicao_sem_decisao = self._deve_punir_sem_decisao(
        inimigo_atacando=inimigo_atacando_agora,
        acao_recompensada=self._acao_anterior,
      )
      aplicar_pontuacao_combo = self._deve_aplicar_pontuacao_combo()
      recompensa, _, info_recompensa = obter_recompensa(
        frame,
        acao_atual=self._acao_anterior,
        inimigo_atacando=inimigo_atacando_agora,
        aplicar_penalidade_nao_reagiu=aplicar_punicao_sem_decisao,
        aplicar_pontuacao_combo=aplicar_pontuacao_combo,
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
        delta_tempo_seg=delta_tempo_seg,
      )
      self._acao_recompensada_anterior = self._acao_anterior
      self._ultima_recompensa = recompensa
      self._recompensa_total_episodio += recompensa
      self._log_recompensa_tempo_real(
        recompensa=recompensa,
        info_recompensa=info_recompensa,
        acao_recompensada=self._acao_anterior,
      )

      if info_recompensa.get("esquiva_tentada"):
        self._esquiva_tentada_episodio += 1
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
      if self._acao_anterior == ACAO_ATAQUE_PESADO:
        self._ataque_pesado_tentado_episodio += 1
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
    # Evita enviesar treino/metricas repetindo a acao em execucao quando a fila esta cheia.
    # Quando nao executa no frame atual, tratamos como esperar.
    acao_registrada = acao if acao_executada else ACAO_ESPERAR
    if not acao_executada:
      self._acoes_fila_cheia_episodio += 1

    self._estado_anterior = estado_atual
    self._acao_anterior = acao_registrada
    if acao_registrada == ACAO_ESPERAR:
      self._frames_esperando_consecutivos += 1
    else:
      self._frames_esperando_consecutivos = 0

    if self.exibir_debug:
      self._exibir_telas_debug(frame, info_vida=info_vida, status_perceptron="estado=lutando")

    self._contador_frames += 1
    if (self._contador_frames % self._log_luta_cada) == 0:
      log("ACAO:", nome_acao(acao_registrada), f"({acao_registrada})", "RECOMPENSA:", self._ultima_recompensa)
      if self._acoes_fila_cheia_episodio > 0:
        log("FILA_ACOES_CHEIA:", self._acoes_fila_cheia_episodio)

    self.vida_jogador_anterior = info_vida["vida_jogador_pct"]
    self.vida_inimigo_anterior = info_vida["vida_inimigo_pct"]

    return {
      "terminal": None,
      "em_luta": True,
    }

  def _log_recompensa_tempo_real(
    self,
    recompensa,
    info_recompensa,
    acao_recompensada,
  ):
    if not self._log_reward_rt:
      return

    self._contador_reward_rt += 1
    if (self._contador_reward_rt % self._log_reward_rt_cada) != 0:
      return

    valor = float(recompensa)
    if abs(valor) < 1e-9:
      return

    info = info_recompensa or {}
    tipo = "GRATIFICACAO" if valor > 0.0 else "PUNICAO"
    motivo = self._motivo_recompensa_principal(info)

    log(
      "[REWARD_RT]",
      tipo,
      f"acao={nome_acao(acao_recompensada)}",
      f"motivo={motivo}",
      f"valor={valor:+.3f}",
    )

  @staticmethod
  def _motivo_recompensa_principal(info):
    if info.get("ataque_pesado_tomou_dano"):
      return "pesado_tomou_dano"
    if info.get("destreza_perfeita"):
      return "destreza_perfeita"
    if info.get("aparar_perfeito"):
      return "aparar_perfeito"
    if info.get("tomou_dano"):
      return "tomou_dano"
    if info.get("causou_dano"):
      return "causou_dano"
    if float(info.get("combo_bonus", 0.0)) > 0.0:
      return "combo_mantido"
    if float(info.get("combo_penalidade", 0.0)) < 0.0:
      return "combo_perdido"
    if info.get("ataque_em_perigo"):
      return "ataque_em_perigo"
    if info.get("nao_reagiu_ataque"):
      return "nao_reagiu_ataque"
    if info.get("tentou_reagir_ataque"):
      return "reagiu_ataque"
    if info.get("spam_destreza"):
      return "spam_destreza"
    if info.get("spam_bloqueio"):
      return "spam_bloqueio"
    if info.get("parado_muito_tempo"):
      return "parado_muito_tempo"
    return "ajuste_geral"

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

  def _deve_punir_sem_decisao(self, inimigo_atacando, acao_recompensada):
    if not inimigo_atacando:
      self._frames_ameaca_sem_decisao = 0
      return False

    if acao_recompensada in (ACAO_ESQUIVA, ACAO_BLOQUEIO):
      self._frames_ameaca_sem_decisao = 0
      return False

    self._frames_ameaca_sem_decisao += 1
    if self._frames_ameaca_sem_decisao < self._janela_punicao_sem_decisao_frames:
      return False

    self._frames_ameaca_sem_decisao = 0
    return True

  def _deve_aplicar_pontuacao_combo(self):
    self._frames_combo_intervalo += 1
    if self._frames_combo_intervalo < self._janela_pontuacao_combo_frames:
      return False

    self._frames_combo_intervalo = 0
    return True

  def _atualizar_evento_debug(self, info_recompensa):
    if info_recompensa.get("ataque_pesado_tomou_dano"):
      self._mensagem_evento_debug = "PESADO PUNIDO -10"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("destreza_perfeita"):
      self._mensagem_evento_debug = "DESTREZA (ESQUIVA PERFEITA) +6"
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
      self._mensagem_evento_debug = "TOMOU DANO"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("destreza_longe_ruim"):
      self._mensagem_evento_debug = "ESQUIVA LONGE"
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
      "esquiva_tentada": int(self._esquiva_tentada_episodio),
      "destreza_tentada": int(self._destreza_tentada_episodio),
      "destreza_perfeita": int(self._destreza_perfeita_episodio),
      "destreza_errada": int(self._destreza_errada_episodio),
      "bloqueio_tentado": int(self._bloqueio_tentado_episodio),
      "aparar_perfeito": int(self._aparar_perfeito_episodio),
      "bloqueio_errado": int(self._bloqueio_errado_episodio),
      "ataque_leve_tentado": int(self._ataque_leve_tentado_episodio),
      "ataque_medio_tentado": int(self._ataque_medio_tentado_episodio),
      "ataque_pesado_tentado": int(self._ataque_pesado_tentado_episodio),
      "acoes_fila_cheia": int(self._acoes_fila_cheia_episodio),
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
    self._esquiva_tentada_episodio = 0
    self._destreza_tentada_episodio = 0
    self._destreza_perfeita_episodio = 0
    self._destreza_errada_episodio = 0
    self._bloqueio_tentado_episodio = 0
    self._aparar_perfeito_episodio = 0
    self._bloqueio_errado_episodio = 0
    self._ataque_leve_tentado_episodio = 0
    self._ataque_medio_tentado_episodio = 0
    self._ataque_pesado_tentado_episodio = 0
    self._acoes_fila_cheia_episodio = 0
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
    self._frames_ameaca_sem_decisao = 0
    self._frames_combo_intervalo = 0
    self._ts_frame_anterior = None
    self._contador_reward_rt = 0
    self._espera_sem_pause_logada = False
    self._ultimo_status_especial = None
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

  def _exibir_telas_debug(self, frame, info_vida=None, status_perceptron=""):
    frame_original = frame.copy()
    if info_vida is not None:
      frame_original = desenhar_info_vida(frame_original, info_vida)
    frame_original = self._desenhar_evento_debug(frame_original)
    frame_original = self._preparar_frame_debug(frame_original)

    frame_pixel = gerar_tela_pixel(self.cerebro, titulo="PPO PIXEL INPUT")
    frame_perceptron = gerar_tela_perceptron(
      self.cerebro,
      titulo="PPO PIXEL PERCEPTRON",
      linha_status=status_perceptron,
      tecla_snapshot=self._tecla_snapshot_filtro,
    )

    tecla_debug = exibir_multiplas(
      {
        NOME_JANELA: frame_original,
        NOME_JANELA_PIXEL: frame_pixel,
        NOME_JANELA_PERCEPTRON: frame_perceptron,
      },
      overlay_metricas_em={NOME_JANELA},
    )
    self._processar_tecla_debug(tecla_debug)

  def _processar_tecla_debug(self, tecla_debug):
    if not tecla_debug:
      return
    if str(tecla_debug).lower() != self._tecla_snapshot_filtro:
      return
    log(
      f"[SNAPSHOT_FILTRO] tecla '{self._tecla_snapshot_filtro.upper()}' detectada. "
      "Imprimindo filtros aprendidos."
    )
    self.cerebro.imprimir_snapshot_filtros(
      origem="treino_normal",
      max_filtros=self._qtd_filtros_snapshot,
    )

  def _registrar_log_especial(self, frame):
    info_poder = obter_info_poder(frame, log_segmentos=False)
    tem_jogador = info_poder.get("tem_especial_jogador")
    tem_inimigo = info_poder.get("tem_especial_inimigo")

    if tem_jogador is None or tem_inimigo is None:
      return

    estado_atual = (bool(tem_jogador), bool(tem_inimigo))
    if estado_atual == self._ultimo_status_especial:
      return

    self._ultimo_status_especial = estado_atual
    log(
      "[ESPECIAL]",
      f"player={'SIM' if estado_atual[0] else 'NAO'}",
      f"inimigo={'SIM' if estado_atual[1] else 'NAO'}",
    )

  def _preparar_frame_debug(self, frame):
    if frame is None:
      return None
    if self._debug_downscale <= 1:
      return frame
    return frame[:: self._debug_downscale, :: self._debug_downscale].copy()

  def _obter_delta_tempo_seg(self):
    agora = time.perf_counter()
    if self._ts_frame_anterior is None:
      self._ts_frame_anterior = agora
      return 1.0 / 60.0

    delta = max(0.0, min(0.25, agora - self._ts_frame_anterior))
    self._ts_frame_anterior = agora
    return delta

  @staticmethod
  def _ler_int_env(nome, padrao):
    try:
      return max(1, int(os.getenv(nome, str(padrao))))
    except ValueError:
      return int(padrao)

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
