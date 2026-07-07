import ctypes
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_BLOQUEIO,
  ACAO_ESPERAR,
  ACAO_ESQUIVA,
  ACAO_ESPECIAL,
  nome_acao,
)
from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import obter_estado_luta
from src.ai.modo_treino import obter_acoes_permitidas, resolver_modo_treino, ModoTreino
from src.ai.recompensa import (
  PESO_KO,
  PESO_VITORIA,
  obter_recompensa,
)
from src.bot.acoes_luta import (
  executar_acao,
  obter_acao_ativa,
  TECLA_ESQUIVA,
  TECLA_BLOQUEIO,
  TECLA_ESPECIAL,
  TECLA_ATAQUE_LEVE,
  TECLA_ATAQUE_MEDIO,
)
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


def converter_para_vkey(tecla_str):
  t = str(tecla_str or "").strip().upper()
  if not t:
    return 0
  if t == "ENTER": return 0x0D
  if t == "SPACE": return 0x20
  if t == "UP": return 0x26
  if t == "DOWN": return 0x28
  if t == "LEFT": return 0x25
  if t == "RIGHT": return 0x27
  if t == "CTRL": return 0x11
  if t == "SHIFT": return 0x10
  if t == "ALT": return 0x12
  if len(t) == 1:
    return ord(t)
  return 0


def esta_pressionada(vkey):
  if os.name != "nt" or vkey <= 0:
    return False
  try:
    return (ctypes.windll.user32.GetAsyncKeyState(vkey) & 0x8000) != 0
  except Exception:
    return False


class Luta:

  def __init__(self, exibir_debug=True, modo_treino="treino"):
    self.exibir_debug = exibir_debug
    self.modo_treino = resolver_modo_treino(modo_treino)
    self.cerebro = CerebroIA(
      acoes_permitidas=obter_acoes_permitidas(self.modo_treino),
      modo_treino=self.modo_treino,
    )
    self._frames_ataque_leve_segurado = 0
    self._vkey_esquiva = converter_para_vkey(TECLA_ESQUIVA)
    self._vkey_bloqueio = converter_para_vkey(TECLA_BLOQUEIO)
    self._vkey_especial = converter_para_vkey(TECLA_ESPECIAL)
    self._vkey_ataque_leve = converter_para_vkey(TECLA_ATAQUE_LEVE)
    self._vkey_ataque_medio = converter_para_vkey(TECLA_ATAQUE_MEDIO)
    self.estado = "BUSCANDO_LUTA"
    self.episodio = 0
    self.vida_jogador_anterior = None
    self.vida_inimigo_anterior = None
    self._especial_jogador_anterior = False
    self._oponente_tem_especial_anterior = False
    self._tempo_especial_inimigo_recente = 0.0
    self._ataques_sem_dano_consecutivos = 0
    self._consecutive_medios = 0
    self._consecutive_leves = 0
    self._consecutive_pesados = 0
    self._combo_hits = 0
    self._exposto = False
    self._tempo_exposto = 0.0
    self._tempo_bloqueio_consecutivo = 0.0
    self._tempo_bloqueio_sem_dano = 0.0
    self._tempo_bloqueio_combo = 0.0
    self._bloqueios_no_combo = 0
    self._esquivas_no_combo = 0
    self._ultimo_bloqueio_ativo = False
    self._ultima_esquiva_ativa = False
    self._ultimo_ataque_ts = 0.0

    self._estado_anterior = None
    self._acao_anterior = None
    self._acao_recompensada_anterior = None
    self._ultima_recompensa = 0.0
    self._historico_acoes = deque(maxlen=10)

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
      self._janela_bloqueio_dano = max(1, int(os.getenv("BOT_BLOQUEIO_DANO_FRAMES", "2")))
    except ValueError:
      self._janela_bloqueio_dano = 2
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
    self._frames_bloqueio_dano = 0
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

  def _detectar_acao_humana(self):
    if self._vkey_especial and esta_pressionada(self._vkey_especial):
      return ACAO_ESPECIAL
    
    if self._vkey_bloqueio and esta_pressionada(self._vkey_bloqueio):
      return ACAO_BLOQUEIO
      
    if self._vkey_esquiva and esta_pressionada(self._vkey_esquiva):
      return ACAO_ESQUIVA

    if self._vkey_ataque_leve and esta_pressionada(self._vkey_ataque_leve):
      self._frames_ataque_leve_segurado += 1
      if self._frames_ataque_leve_segurado >= 8:
        return ACAO_ATAQUE_PESADO
      return ACAO_ATAQUE_LEVE
    else:
      self._frames_ataque_leve_segurado = 0

    if self._vkey_ataque_medio and esta_pressionada(self._vkey_ataque_medio):
      return ACAO_ATAQUE_MEDIO

    return ACAO_ESPERAR

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
          especial_disponivel=self._especial_jogador_anterior,
          oponente_especial_recente=(self._tempo_especial_inimigo_recente > 0.0),
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
    info_poder = obter_info_poder(frame, log_segmentos=False)
    self._registrar_log_especial(info_poder)
    sinais_estado = self._detectar_sinais_estado(info_vida)
    estado_atual = self.cerebro.obter_estado(
      frame=frame,
      info_vida=info_vida,
      sinais=sinais_estado,
    )

    tem_especial = bool(info_poder.get("tem_especial_jogador"))
    inimigo_tem_especial = bool(info_poder.get("tem_especial_inimigo"))

    # Atualiza o cronômetro do especial recente do inimigo
    if self._tempo_especial_inimigo_recente > 0.0:
      self._tempo_especial_inimigo_recente = max(0.0, self._tempo_especial_inimigo_recente - delta_tempo_seg)

    # Detecção de transição: inimigo descarregou o especial
    if self._oponente_tem_especial_anterior and not inimigo_tem_especial:
      self._tempo_especial_inimigo_recente = 2.5
      from src.bot.acoes_luta import limpar_fila_acoes
      limpar_fila_acoes()
      log("[ESPECIAL] Inimigo soltou especial! Fila de acoes limpa. Janela defensiva de 2.5s iniciada.")

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
        historico_acoes=list(self._historico_acoes),
        nivel_especial_inimigo=info_poder.get("nivel_especial_inimigo"),
        oponente_especial_recente=(self._tempo_especial_inimigo_recente > 0.0),
      )
      # --- Atualização do Estado do Combo e Exposição ---
      causou_dano = bool(info_recompensa.get("causou_dano"))
      tomou_dano = bool(info_recompensa.get("tomou_dano"))
      aparar_perfeito = bool(info_recompensa.get("aparar_perfeito"))
      destreza_perfeita = bool(info_recompensa.get("destreza_perfeita"))

      # --- Bônus Específicos para Modos de Treino Focado ---
      if self.modo_treino not in (ModoTreino.COMPLETO, ModoTreino.ASSISTIDO):
        recompensa_focada = 0.0
        info_focada = info_recompensa.copy()

        if self.modo_treino == ModoTreino.DEFENDER:
          # Apenas gratifica se estiver bloqueando E tomou dano (mitigou dano)
          if self._acao_anterior == ACAO_BLOQUEIO:
            self._tempo_bloqueio_consecutivo += delta_tempo_seg
            self._tempo_bloqueio_sem_dano += delta_tempo_seg

            if tomou_dano:
              recompensa_focada += 5.0
              info_focada["defesa_mitigou_dano"] = True
              self._tempo_bloqueio_sem_dano = 0.0

            if self._tempo_bloqueio_consecutivo > 10.0:
              recompensa_focada += -2.0
              info_focada["punicao_defesa_10s"] = True
            elif self._tempo_bloqueio_sem_dano > 5.0:
              recompensa_focada += -1.5
              info_focada["punicao_defesa_sem_dano_5s"] = True
          else:
            self._tempo_bloqueio_consecutivo = 0.0
            self._tempo_bloqueio_sem_dano = 0.0

        elif self.modo_treino == ModoTreino.COMBO:
          # Bloqueio prolongado penalidade em Combo Mode (> 2.5s)
          if self._acao_anterior == ACAO_BLOQUEIO:
            self._tempo_bloqueio_combo += delta_tempo_seg
            if self._tempo_bloqueio_combo > 2.5:
              recompensa_focada += -1.5
              info_focada["punicao_defesa_combo_2_5s"] = True
          else:
            self._tempo_bloqueio_combo = 0.0

          # Se tomou dano, o combo quebra e zera para 0
          if tomou_dano:
            self._combo_hits = 0

          # Se causou dano, reseta os contadores individuais de bloqueios e esquivas
          if causou_dano:
            self._bloqueios_no_combo = 0
            self._esquivas_no_combo = 0
            self._ultimo_bloqueio_ativo = False
            self._ultima_esquiva_ativa = False

          # Borda de subida Bloqueio
          if self._acao_anterior == ACAO_BLOQUEIO:
            if not self._ultimo_bloqueio_ativo:
              self._ultimo_bloqueio_ativo = True
              self._bloqueios_no_combo += 1
              log(f"[COMBO-DEFESA] Ativacao de Bloqueio {self._bloqueios_no_combo}/3")
              if self._bloqueios_no_combo >= 3:
                recompensa_focada += -3.0
                info_focada["punicao_limite_bloqueios_combo"] = True
          else:
            self._ultimo_bloqueio_ativo = False

          # Borda de subida Esquiva
          if self._acao_anterior == ACAO_ESQUIVA:
            if not self._ultima_esquiva_ativa:
              self._ultima_esquiva_ativa = True
              self._esquivas_no_combo += 1
              log(f"[COMBO-DEFESA] Ativacao de Esquiva {self._esquivas_no_combo}/3")
              if self._esquivas_no_combo >= 3:
                recompensa_focada += -3.0
                info_focada["punicao_limite_esquivas_combo"] = True
          else:
            self._ultima_esquiva_ativa = False

          # Ações de ataque:
          if self._acao_anterior in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO):
            if causou_dano:
              self._combo_hits += 1
              if self._combo_hits in (1, 2, 3):
                recompensa_focada += 1.0
                info_focada["combo_hit_intermediario"] = True
              elif self._combo_hits == 4:
                recompensa_focada += 2.0
                info_focada["combo_hit_finalizador"] = True
              elif self._combo_hits > 4:
                recompensa_focada += -3.0
                info_focada["combo_incorreto"] = True
            else:
              # Ataque não causou dano (errou ou bateu na defesa): punição!
              recompensa_focada += -1.5
              info_focada["punicao_ataque_sem_dano_treino"] = True

          # Ações de reset/defensivas (aplica bônus de reset correto se atingiu 4 hits, senão reseta combo para 0)
          elif self._acao_anterior in (ACAO_ESQUIVA, ACAO_BLOQUEIO):
            if self._combo_hits == 4:
              recompensa_focada += 10.0
              info_focada["combo_completado"] = True
              self._bloqueios_no_combo = 0
              self._esquivas_no_combo = 0
              self._ultimo_bloqueio_ativo = False
              self._ultima_esquiva_ativa = False
            self._combo_hits = 0

          # Esperar ou outras ações resetam o combo para 0
          elif self._acao_anterior == ACAO_ESPERAR:
            self._combo_hits = 0

        elif self.modo_treino == ModoTreino.APARAR:
          if aparar_perfeito:
            recompensa_focada += 5.0
            info_focada["aparar_perfeito_focado"] = True

        elif self.modo_treino == ModoTreino.DESTREZA:
          if destreza_perfeita:
            recompensa_focada += 6.0
            info_focada["destreza_perfeita_focado"] = True

        recompensa = recompensa_focada
        info_recompensa = info_focada

      # Se executou ação de reset, limpa contadores (apenas no modo completo ou assistido)
      if self.modo_treino in (ModoTreino.COMPLETO, ModoTreino.ASSISTIDO):
        if self._acao_anterior in (ACAO_ESQUIVA, ACAO_BLOQUEIO, ACAO_ESPERAR):
          self._combo_hits = 0
          self._consecutive_medios = 0
          self._consecutive_leves = 0
          self._consecutive_pesados = 0

        # Incrementa hits se causou dano
        if causou_dano:
          self._combo_hits += 1

      # Controle de spam de ataques sem dano
      is_ataque = self._acao_anterior in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO, ACAO_ESPECIAL)
      if is_ataque:
        if causou_dano:
          self._ataques_sem_dano_consecutivos = 0
        else:
          self._ataques_sem_dano_consecutivos += 1
      else:
        self._ataques_sem_dano_consecutivos = 0

      if self._ataques_sem_dano_consecutivos > 2:
        peso_spam_ataque = -2.0
        recompensa += peso_spam_ataque
        info_recompensa["punicao_ataque_sem_dano"] = True
        info_recompensa["punicao_ataque_sem_dano_valor"] = peso_spam_ataque

      # Contagem de ataques consecutivos
      if self._acao_anterior == ACAO_ATAQUE_MEDIO:
        self._consecutive_medios += 1
        self._consecutive_leves = 0
        self._consecutive_pesados = 0
      elif self._acao_anterior == ACAO_ATAQUE_LEVE:
        self._consecutive_leves += 1
        self._consecutive_medios = 0
        self._consecutive_pesados = 0
      elif self._acao_anterior == ACAO_ATAQUE_PESADO:
        self._consecutive_pesados += 1
        self._consecutive_leves = 0
        self._consecutive_medios = 0

      # Condições que ativam Exposição
      if self._consecutive_medios >= 2:
        if not self._exposto:
          log("[COMBO] Exposto por 2 ataques médios consecutivos!")
        self._exposto = True
        self._tempo_exposto = 0.0

      if self._consecutive_pesados >= 2:
        if not self._exposto:
          log("[COMBO] Exposto por 2 ataques pesados consecutivos!")
        self._exposto = True
        self._tempo_exposto = 0.0

      if self._consecutive_leves >= 4:
        if not self._exposto:
          log("[COMBO] Exposto por 4 ataques leves consecutivos!")
        self._exposto = True
        self._tempo_exposto = 0.0

      if self._combo_hits >= 5 and self._acao_anterior in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO):
        if not self._exposto:
          log(f"[COMBO] Exposto por finalizar o combo de {self._combo_hits} hits!")
        self._exposto = True
        self._tempo_exposto = 0.0

      # Condições que desativam Exposição
      if self._exposto:
        self._tempo_exposto += delta_tempo_seg
        if tomou_dano:
          self._exposto = False
          self._consecutive_medios = 0
          self._consecutive_leves = 0
          self._consecutive_pesados = 0
          self._combo_hits = 0
          log("[COMBO] Exposição resolvida por tomar dano.")
        elif aparar_perfeito:
          self._exposto = False
          self._consecutive_medios = 0
          self._consecutive_leves = 0
          self._consecutive_pesados = 0
          self._combo_hits = 0
          log("[COMBO] Exposição resolvida por Aparar Perfeito!")
        elif self._acao_anterior == ACAO_ESQUIVA:
          self._exposto = False
          self._consecutive_medios = 0
          self._consecutive_leves = 0
          self._consecutive_pesados = 0
          self._combo_hits = 0
          log("[COMBO] Exposição resolvida por realizar Esquiva.")
        elif self._tempo_exposto >= 1.5:
          self._exposto = False
          self._consecutive_medios = 0
          self._consecutive_leves = 0
          self._consecutive_pesados = 0
          self._combo_hits = 0
          log("[COMBO] Exposição expirada após esperar 1.5s.")

      # Aplica a punição se tomou ação inválida estando exposto (apenas no modo completo ou assistido)
      if self._exposto and self.modo_treino in (ModoTreino.COMPLETO, ModoTreino.ASSISTIDO):
        if self._acao_anterior in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO, ACAO_ESPECIAL):
          try:
            peso_exposicao = float(os.getenv("BOT_PESO_PUNICAO_EXPOSICAO", "-8.00"))
          except ValueError:
            peso_exposicao = -8.00
          
          recompensa += peso_exposicao
          info_recompensa["punicao_exposicao"] = True
          info_recompensa["punicao_exposicao_valor"] = peso_exposicao
          self._punicoes_exposicao_episodio += 1

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
        especial_disponivel=self._especial_jogador_anterior,
        oponente_especial_recente=(self._tempo_especial_inimigo_recente > 0.0),
      )
      self._atualizar_evento_debug(info_recompensa)

    # 1. Determina as acoes permitidas base do modo de treino
    permitidas = obter_acoes_permitidas(self.modo_treino)

    # 2. Se o bot esta tomando dano (janela de bloqueio ativa), restringe a apenas esperar
    if self._frames_bloqueio_dano > 0:
      permitidas = [ACAO_ESPERAR]
      self._frames_bloqueio_dano -= 1
    else:
      # Se saiu do bloqueio absoluto, mas ainda esta sob dano recente (hitstun residual/combo),
      # bloqueamos todas as acoes ofensivas para evitar contra-ataques inuteis e punicoes.
      if sinais_estado.get("tomou_dano_recente"):
        permitidas = [a for a in permitidas if a not in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO, ACAO_ESPECIAL)]

      # Caso contrario, aplica as restricoes especificas do modo de treino
      if self.modo_treino == ModoTreino.COMBO:
        # Cooldown de ataque
        if time.time() - self._ultimo_ataque_ts < 0.3:
          permitidas = [a for a in permitidas if a not in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO)]
          
        # Limite de bloqueios
        if self._bloqueios_no_combo >= 3:
          permitidas = [a for a in permitidas if a != ACAO_BLOQUEIO]
          
        # Limite de esquivas
        if self._esquivas_no_combo >= 3:
          permitidas = [a for a in permitidas if a != ACAO_ESQUIVA]

        # Se bloqueios e esquivas estao esgotados, forca o ataque mascarando esperar
        if self._bloqueios_no_combo >= 3 and self._esquivas_no_combo >= 3:
          permitidas = [a for a in permitidas if a != ACAO_ESPERAR]

    if not permitidas:
      permitidas = [ACAO_ESPERAR]
      
    self.cerebro.acoes_permitidas = permitidas

    if self.modo_treino == ModoTreino.ASSISTIDO:
      acao = self._detectar_acao_humana()
      acao_executada = True
      acao_registrada = acao
    else:
      acao_ativa = obter_acao_ativa()
      if acao_ativa != ACAO_ESPERAR:
        # O bot esta executando uma acao fisica. Nao chamamos o cerebro para escolher outra.
        # Apenas registramos que a acao atual continua sendo a acao_ativa.
        acao = acao_ativa
        acao_executada = False
      else:
        # O bot esta livre para decidir.
        acao = self.cerebro.escolher_acao(
          estado_atual,
          especial_disponivel=tem_especial,
          oponente_tem_especial=inimigo_tem_especial,
          oponente_especial_recente=(self._tempo_especial_inimigo_recente > 0.0),
        )
        acao_executada = executar_acao(acao)

      if acao_executada and acao in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO) and self.modo_treino == ModoTreino.COMBO:
        self._ultimo_ataque_ts = time.time()

      # Evita enviesar treino/metricas repetindo a acao em execucao quando a fila esta cheia.
      # Se a acao veio do worker de acao ativa, ela e a acao registrada de fato.
      if acao_ativa != ACAO_ESPERAR:
        acao_registrada = acao_ativa
      else:
        acao_registrada = acao if acao_executada else ACAO_ESPERAR
        if not acao_executada:
          self._acoes_fila_cheia_episodio += 1

    self._estado_anterior = estado_atual
    self._acao_anterior = acao_registrada
    self._historico_acoes.append(acao_registrada)
    
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
    self._especial_jogador_anterior = tem_especial
    self._oponente_tem_especial_anterior = inimigo_tem_especial

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
    if info.get("combo_reset_defensivo"):
      return "combo_reset_defensivo"
    if info.get("punicao_exposicao"):
      return "punicao_exposicao_combo"
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
    if info.get("punicao_agressividade_especial"):
      return "punicao_agressividade_especial"
    if info.get("punicao_defesa_sem_especial"):
      return "punicao_defesa_sem_especial"
    if info.get("punicao_oponente_e3"):
      return "punicao_oponente_e3"
    if info.get("punicao_ataque_sem_dano_treino"):
      return "ataque_sem_dano_treino"
    if info.get("punicao_limite_bloqueios_combo"):
      return "limite_bloqueios_combo"
    if info.get("punicao_limite_esquivas_combo"):
      return "limite_esquivas_combo"
    if info.get("defesa_mitigou_dano"):
      return "defesa_mitigou_dano"
    if info.get("punicao_defesa_10s"):
      return "punicao_defesa_10s"
    if info.get("punicao_defesa_sem_dano_5s"):
      return "punicao_defesa_sem_dano_5s"
    if info.get("punicao_defesa_combo_2_5s"):
      return "punicao_defesa_combo_2_5s"
    if info.get("combo_hit_intermediario"):
      return "combo_hit_intermediario"
    if info.get("combo_hit_finalizador"):
      return "combo_hit_finalizador"
    if info.get("combo_incorreto"):
      return "combo_incorreto"
    if info.get("punicao_ataque_sem_dano"):
      return "ataque_sem_dano"
    if info.get("punicao_esquiva_errada_treino"):
      return "esquiva_errada_treino"
    if info.get("punicao_bloqueio_errado_treino"):
      return "bloqueio_errado_treino"
    if info.get("decaimento_repeticao"):
      return "decaimento_repeticao"
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
      self._frames_bloqueio_dano = self._janela_bloqueio_dano
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
    if info_recompensa.get("punicao_limite_bloqueios_combo"):
      self._mensagem_evento_debug = "EXCESSO BLOQUEIO -3.0"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_limite_esquivas_combo"):
      self._mensagem_evento_debug = "EXCESSO ESQUIVA -3.0"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_ataque_sem_dano_treino"):
      self._mensagem_evento_debug = "ATK FALHOU -1.5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_defesa_combo_2_5s"):
      self._mensagem_evento_debug = "PUNIDO DEFESA COMBO -1.5"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("combo_hit_intermediario"):
      self._mensagem_evento_debug = "HIT COMBO +1.0"
      self._mensagem_evento_debug_restante = 16
      return
    if info_recompensa.get("combo_hit_finalizador"):
      self._mensagem_evento_debug = "HIT 4 +2.0"
      self._mensagem_evento_debug_restante = 16
      return
    if info_recompensa.get("defesa_mitigou_dano"):
      self._mensagem_evento_debug = "MITIGOU DANO +5.0"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_defesa_10s"):
      self._mensagem_evento_debug = "PUNIDO DEFESA 10S"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_defesa_sem_dano_5s"):
      self._mensagem_evento_debug = "PUNIDO DEFESA S/ DANO"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("combo_completado"):
      self._mensagem_evento_debug = "COMBO COMPLETADO +10.0"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("combo_incorreto"):
      self._mensagem_evento_debug = "COMBO INCORRETO -3.0"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_ataque_sem_dano"):
      self._mensagem_evento_debug = "PUNIDO SPAM ATK"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_ofensiva_especial_recente"):
      self._mensagem_evento_debug = "PUNIDO OFENSIVO ESP"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_espera_especial_recente"):
      self._mensagem_evento_debug = "PUNIDO ESPERA ESP"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("gratificacao_defesa_especial_recente"):
      self._mensagem_evento_debug = "DEFESA ESPECIAL +2"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("combo_reset_defensivo"):
      self._mensagem_evento_debug = "RESET DEFENSIVO +10"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_exposicao"):
      self._mensagem_evento_debug = "PUNIDO EXPOSICAO -8"
      self._mensagem_evento_debug_restante = 20
      return
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
    if info_recompensa.get("punicao_agressividade_especial"):
      self._mensagem_evento_debug = "PUNIDO AGRESSIVIDADE"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_defesa_sem_especial"):
      self._mensagem_evento_debug = "PUNIDO DEFESA SEM ESP"
      self._mensagem_evento_debug_restante = 20
      return
    if info_recompensa.get("punicao_oponente_e3"):
      self._mensagem_evento_debug = "PUNIDO OPONENTE E3!"
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
      "modo_treino": self.modo_treino.value,
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
      "punicoes_exposicao": int(self._punicoes_exposicao_episodio),
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
    self._tempo_especial_inimigo_recente = 0.0
    self._ataques_sem_dano_consecutivos = 0
    self._punicoes_exposicao_episodio = 0
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
    self._tempo_bloqueio_consecutivo = 0.0
    self._tempo_bloqueio_sem_dano = 0.0
    self._tempo_bloqueio_combo = 0.0
    self._bloqueios_no_combo = 0
    self._esquivas_no_combo = 0
    self._ultimo_bloqueio_ativo = False
    self._ultima_esquiva_ativa = False
    self._ultimo_ataque_ts = 0.0

  def _resetar_contexto_pos_episodio(self):
    self.estado = "BUSCANDO_LUTA"
    self.vida_jogador_anterior = None
    self.vida_inimigo_anterior = None
    self._especial_jogador_anterior = False
    self._oponente_tem_especial_anterior = False
    self._consecutive_medios = 0
    self._consecutive_leves = 0
    self._consecutive_pesados = 0
    self._combo_hits = 0
    self._exposto = False
    self._tempo_exposto = 0.0
    self._tempo_bloqueio_consecutivo = 0.0
    self._tempo_bloqueio_sem_dano = 0.0
    self._tempo_bloqueio_combo = 0.0
    self._bloqueios_no_combo = 0
    self._esquivas_no_combo = 0
    self._ultimo_bloqueio_ativo = False
    self._ultima_esquiva_ativa = False
    self._ultimo_ataque_ts = 0.0
    self._estado_anterior = None
    self._acao_anterior = None
    self._acao_recompensada_anterior = None
    self._ultima_recompensa = 0.0
    self._historico_acoes.clear()
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

  def _registrar_log_especial(self, info_poder):
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
