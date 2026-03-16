import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from pynput import keyboard

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_BLOQUEIO,
  ACAO_ESQUIVA,
)
from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import obter_estado_luta
from src.bot.controle_fps import ControleFPS, obter_fps_alvo
from src.bot.captura import CapturaAssincrona
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.rois import ROI_ACOES_PERFEITAS, obter_roi
from src.bot.vida import obter_info_vida
from src.bot.visao import encontrar_template
from src.utils.debug_rede_telas import gerar_tela_perceptron, gerar_tela_pixel
from src.utils.log import log
from src.utils.visao_debug import (
  NOME_JANELA,
  NOME_JANELA_PERCEPTRON,
  NOME_JANELA_PIXEL,
  atualizar_metricas,
  exibir_multiplas,
)

TITULO_JOGO = "Champions"
PASTA_DEMONSTRACOES = Path("data/demonstrations")
TEMPLATE_DESTREZA = "assets/destreza.png"
TEMPLATE_APARAR = "assets/aparar.png"


def obter_monitor_jogo():
  log("Procurando janela do jogo...")
  janela = None
  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(1)

  log("Janela encontrada!")
  auto_focus = os.getenv("BOT_OBS_AUTOFOCUS", "0").strip() == "1"
  if auto_focus:
    if focar_jogo(TITULO_JOGO):
      log("Janela do jogo focada automaticamente.")
      time.sleep(0.5)
    else:
      log("Nao foi possivel focar automaticamente a janela.")
  else:
    log("Treino Observacao com auto-focus desativado (sem interferir na gameplay).")
  return obter_bbox_janela(janela)


def _normalizar_tecla(tecla):
  if isinstance(tecla, keyboard.KeyCode):
    texto = (tecla.char or "").strip().lower()
    return texto if texto else None

  nome = str(tecla).strip().lower()
  if nome.startswith("key."):
    nome = nome[4:]
  return nome or None


def _env_float(nome, padrao):
  valor = os.getenv(nome, "").strip()
  if not valor:
    return float(padrao)
  try:
    return float(valor)
  except ValueError:
    return float(padrao)


def _env_tecla(nome_novo, nome_legado, padrao):
  valor = os.getenv(nome_novo, "").strip()
  if valor:
    return valor
  valor_legado = os.getenv(nome_legado, "").strip()
  if valor_legado:
    return valor_legado
  return padrao


def _env_int(nome, padrao):
  valor = os.getenv(nome, "").strip()
  if not valor:
    return int(padrao)
  try:
    return int(valor)
  except ValueError:
    return int(padrao)


def _preparar_frame_debug(frame):
  if frame is None:
    return None
  downscale = max(1, _env_int("BOT_DEBUG_DOWNSCALE", 2))
  if downscale <= 1:
    return frame
  return frame[::downscale, ::downscale].copy()


class _BufferFrames:
  FRAME_SIZE = 128
  STACK_SIZE = 4

  def __init__(self):
    self._stack = deque(maxlen=self.STACK_SIZE)

  def reset(self):
    self._stack.clear()

  def observar(self, frame):
    if frame is None:
      quadro = np.zeros((self.FRAME_SIZE, self.FRAME_SIZE), dtype=np.float32)
    else:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      red = cv2.resize(gray, (self.FRAME_SIZE, self.FRAME_SIZE), interpolation=cv2.INTER_AREA)
      quadro = red.astype(np.float32) / 255.0

    if not self._stack:
      for _i in range(self.STACK_SIZE):
        self._stack.append(quadro)
    else:
      self._stack.append(quadro)

    return np.stack(self._stack, axis=0).astype(np.float32)


class ColetorDemonstracao:

  def __init__(self):
    self._buffer = _BufferFrames()
    self._episodio_idx = self._obter_proximo_indice_episodio()
    self._transicoes = []
    self._eventos_acao = deque()
    self._pendente = None
    self._listener = None
    self._teclas_pressionadas = {}

    self._tecla_para_config = self._carregar_mapeamento_teclas()
    self._total_transicoes_salvas = 0
    try:
      self._delay_efeito_frames = max(1, int(os.getenv("BOT_OBS_EFFECT_DELAY_FRAMES", "3")))
    except ValueError:
      self._delay_efeito_frames = 3

    self._limiar_pesado_seg = max(
      0.05,
      _env_float(
        "BOT_OBS_HEAVY_HOLD_SEG",
        _env_float("BOT_TEMPO_ATAQUE_PESADO", 0.25),
      ),
    )
    self._limiar_destreza = _env_float("BOT_LIMIAR_DESTREZA", 0.8)
    self._limiar_aparar = _env_float("BOT_LIMIAR_APARAR", 0.8)
    self._bonus_destreza = _env_float("BOT_OBS_BONUS_DESTREZA", 1.0)
    self._bonus_aparar = _env_float("BOT_OBS_BONUS_APARAR", 0.7)
    self._log_acoes = os.getenv("BOT_OBS_LOG_ACOES", "1").strip() == "1"

  def _carregar_mapeamento_teclas(self):
    tecla_esquiva = _env_tecla("BOT_TECLA_ESQUIVA", "BOT_TECLA_DESTREZA", "f").lower()
    tecla_bloqueio = (os.getenv("BOT_TECLA_BLOQUEIO", "space").strip() or "space").lower()
    tecla_ataque_leve = (os.getenv("BOT_TECLA_ATAQUE_LEVE", "j").strip() or "j").lower()
    tecla_ataque_medio = (os.getenv("BOT_TECLA_ATAQUE_MEDIO", "k").strip() or "k").lower()

    return {
      tecla_esquiva: {
        "acao_human": "esquiva",
        "acao_id": ACAO_ESQUIVA,
        "modo": "press",
      },
      tecla_bloqueio: {
        "acao_human": "bloqueio",
        "acao_id": ACAO_BLOQUEIO,
        "modo": "press",
      },
      tecla_ataque_leve: {
        "acao_human": "ataque_leve",
        "acao_id": ACAO_ATAQUE_LEVE,
        "modo": "hold_light",
      },
      tecla_ataque_medio: {
        "acao_human": "ataque_medio",
        "acao_id": ACAO_ATAQUE_MEDIO,
        "modo": "press",
      },
    }

  def _obter_proximo_indice_episodio(self):
    if not PASTA_DEMONSTRACOES.exists():
      return 1

    maior = 0
    for arquivo in PASTA_DEMONSTRACOES.glob("episode_*.npz"):
      nome = arquivo.stem
      try:
        idx = int(nome.split("_")[-1])
      except ValueError:
        continue
      maior = max(maior, idx)
    return maior + 1

  def iniciar_listener(self):
    if self._listener is not None:
      return

    def _registrar_evento(acao_human, acao_id, tecla, ts, hold_ms=0.0):
      self._eventos_acao.append(
        {
          "acao_human": str(acao_human),
          "acao_id": int(acao_id),
          "tecla": str(tecla),
          "ts": float(ts),
          "hold_ms": float(hold_ms),
        }
      )

    def _on_press(tecla):
      nome = _normalizar_tecla(tecla)
      if not nome:
        return
      config = self._tecla_para_config.get(nome)
      if not config:
        return

      agora = time.time()
      if nome in self._teclas_pressionadas:
        return
      self._teclas_pressionadas[nome] = agora

      if config["modo"] == "press":
        _registrar_evento(
          acao_human=config["acao_human"],
          acao_id=config["acao_id"],
          tecla=nome,
          ts=agora,
          hold_ms=0.0,
        )

    def _on_release(tecla):
      nome = _normalizar_tecla(tecla)
      if not nome:
        return

      ts_press = self._teclas_pressionadas.pop(nome, None)
      config = self._tecla_para_config.get(nome)
      if ts_press is None or not config:
        return

      if config["modo"] != "hold_light":
        return

      hold_ms = max(0.0, (time.time() - ts_press) * 1000.0)
      if hold_ms >= (self._limiar_pesado_seg * 1000.0):
        acao_human = "ataque_pesado"
        acao_id = ACAO_ATAQUE_PESADO
      else:
        acao_human = "ataque_leve"
        acao_id = ACAO_ATAQUE_LEVE

      _registrar_evento(
        acao_human=acao_human,
        acao_id=acao_id,
        tecla=nome,
        ts=ts_press,
        hold_ms=hold_ms,
      )

    self._listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    self._listener.start()
    log(
      "Coleta humana ativa. Acoes salvas quando voce executar acao.",
      f"ataque_pesado>= {int(self._limiar_pesado_seg * 1000)}ms",
    )

  def parar_listener(self):
    if self._listener is None:
      return
    try:
      self._listener.stop()
    except Exception:
      pass
    self._listener = None

  def reset_episodio(self):
    self._buffer.reset()
    self._transicoes = []
    self._eventos_acao.clear()
    self._pendente = None
    self._teclas_pressionadas.clear()

  def _classificar_resultado_acao(self, frame, acao_human):
    acao_registrada = str(acao_human)
    destreza_perfeita = False
    aparar_perfeito = False

    if frame is None:
      return {
        "acao_registrada": acao_registrada,
        "destreza_perfeita": destreza_perfeita,
        "aparar_perfeito": aparar_perfeito,
      }

    roi_perfeitas = obter_roi(ROI_ACOES_PERFEITAS)

    if acao_human == "esquiva":
      destreza_perfeita = (
        encontrar_template(
          frame,
          TEMPLATE_DESTREZA,
          limiar=self._limiar_destreza,
          roi=roi_perfeitas,
        ) is not None
      )
      if destreza_perfeita:
        acao_registrada = "destreza"
    elif acao_human == "bloqueio":
      aparar_perfeito = (
        encontrar_template(
          frame,
          TEMPLATE_APARAR,
          limiar=self._limiar_aparar,
          roi=roi_perfeitas,
        ) is not None
      )
      if aparar_perfeito:
        acao_registrada = "aparar"

    return {
      "acao_registrada": acao_registrada,
      "destreza_perfeita": bool(destreza_perfeita),
      "aparar_perfeito": bool(aparar_perfeito),
    }

  def _bonus_resultado(self, resultado_acao):
    bonus = 0.0
    if resultado_acao.get("destreza_perfeita"):
      bonus += self._bonus_destreza
    if resultado_acao.get("aparar_perfeito"):
      bonus += self._bonus_aparar
    return float(bonus)

  def atualizar(self, frame, info_vida, done=False, terminal_label=None):
    estado = self._buffer.observar(frame)
    info_vida = info_vida or {}

    # Finaliza transicao pendente somente apos alguns frames para capturar o efeito da acao.
    if self._pendente is not None:
      if (not done) and self._pendente["frames_restantes"] > 0:
        self._pendente["frames_restantes"] -= 1
      else:
        resultado_acao = self._classificar_resultado_acao(frame, self._pendente["acao_human"])
        recompensa_base = self._calcular_recompensa_demo(
          self._pendente["vida_antes"],
          info_vida,
          done,
          terminal_label,
        )
        bonus_resultado = self._bonus_resultado(resultado_acao)
        recompensa = recompensa_base + bonus_resultado
        transicao = {
          "state": self._pendente["state"],
          "action": resultado_acao["acao_registrada"],
          "action_base": self._pendente["acao_human"],
          "action_id": int(self._pendente["acao_id"]),
          "reward": float(recompensa),
          "reward_base": float(recompensa_base),
          "reward_bonus": float(bonus_resultado),
          "next_state": estado.copy(),
          "done": bool(done),
          "ts_action": float(self._pendente["ts"]),
          "hold_ms": float(self._pendente.get("hold_ms", 0.0)),
          "destreza_perfeita": bool(resultado_acao["destreza_perfeita"]),
          "aparar_perfeito": bool(resultado_acao["aparar_perfeito"]),
        }
        self._transicoes.append(transicao)
        if self._log_acoes:
          log(
            "[OBS_ACAO]",
            f"base={transicao['action_base']}",
            f"registro={transicao['action']}",
            f"id={transicao['action_id']}",
            f"reward={transicao['reward']:.3f}",
            f"hold_ms={transicao['hold_ms']:.1f}",
          )
        self._pendente = None
        self._total_transicoes_salvas += 1

    if done:
      return

    # Registra acao humana apenas quando houver tecla acionada.
    while (self._pendente is None) and self._eventos_acao:
      evento = self._eventos_acao.popleft()
      self._pendente = {
        "state": estado.copy(),
        "acao_human": evento["acao_human"],
        "acao_id": int(evento["acao_id"]),
        "tecla": evento["tecla"],
        "ts": float(evento["ts"]),
        "hold_ms": float(evento.get("hold_ms", 0.0)),
        "vida_antes": dict(info_vida),
        "frames_restantes": int(self._delay_efeito_frames),
      }

  @staticmethod
  def _calcular_recompensa_demo(vida_antes, vida_depois, done, terminal_label):
    recompensa = 0.0

    vj0 = vida_antes.get("vida_jogador_pct")
    vj1 = vida_depois.get("vida_jogador_pct")
    vi0 = vida_antes.get("vida_inimigo_pct")
    vi1 = vida_depois.get("vida_inimigo_pct")

    if vj0 is not None and vj1 is not None and vj1 < vj0:
      recompensa -= float(vj0 - vj1) * 2.0
    if vi0 is not None and vi1 is not None and vi1 < vi0:
      recompensa += float(vi0 - vi1) * 1.5

    if done:
      if terminal_label == "vitoria":
        recompensa += 20.0
      elif terminal_label == "ko":
        recompensa -= 15.0

    return recompensa

  def salvar_episodio(self, resultado):
    if not self._transicoes:
      log("Episodio sem transicoes de acao humana. Nada para salvar.")
      self.reset_episodio()
      return None

    PASTA_DEMONSTRACOES.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_DEMONSTRACOES / f"episode_{self._episodio_idx:03d}.npz"

    states = np.asarray([t["state"] for t in self._transicoes], dtype=np.float16)
    next_states = np.asarray([t["next_state"] for t in self._transicoes], dtype=np.float16)
    actions = np.asarray([t["action_id"] for t in self._transicoes], dtype=np.int16)
    action_labels = np.asarray([t["action"] for t in self._transicoes], dtype="<U32")
    action_base_labels = np.asarray([t["action_base"] for t in self._transicoes], dtype="<U32")
    rewards = np.asarray([t["reward"] for t in self._transicoes], dtype=np.float32)
    rewards_base = np.asarray([t["reward_base"] for t in self._transicoes], dtype=np.float32)
    rewards_bonus = np.asarray([t["reward_bonus"] for t in self._transicoes], dtype=np.float32)
    dones = np.asarray([t["done"] for t in self._transicoes], dtype=np.bool_)
    timestamps = np.asarray([t["ts_action"] for t in self._transicoes], dtype=np.float64)
    holds_ms = np.asarray([t["hold_ms"] for t in self._transicoes], dtype=np.float32)
    destreza_perfeita = np.asarray([t["destreza_perfeita"] for t in self._transicoes], dtype=np.bool_)
    aparar_perfeito = np.asarray([t["aparar_perfeito"] for t in self._transicoes], dtype=np.bool_)

    np.savez_compressed(
      caminho,
      states=states,
      next_states=next_states,
      actions=actions,
      action_labels=action_labels,
      action_base_labels=action_base_labels,
      rewards=rewards,
      rewards_base=rewards_base,
      rewards_bonus=rewards_bonus,
      dones=dones,
      timestamps=timestamps,
      holds_ms=holds_ms,
      destreza_perfeita=destreza_perfeita,
      aparar_perfeito=aparar_perfeito,
      resultado=np.asarray([str(resultado)], dtype="<U16"),
    )

    total = len(self._transicoes)
    totais = {
      "destreza": int(np.count_nonzero(action_labels == "destreza")),
      "aparar": int(np.count_nonzero(action_labels == "aparar")),
      "ataque_pesado": int(np.count_nonzero(action_labels == "ataque_pesado")),
    }
    self._episodio_idx += 1
    self.reset_episodio()
    log(
      f"Episodio salvo: {caminho} | transicoes={total}",
      f"destreza={totais['destreza']}",
      f"aparar={totais['aparar']}",
      f"ataque_pesado={totais['ataque_pesado']}",
    )
    return caminho


def iniciar_treino_observacao(monitor=None):
  if monitor is None:
    monitor = obter_monitor_jogo()

  fps_alvo = obter_fps_alvo()
  controle_fps = ControleFPS(fps_alvo)
  log(f"Treino Observacao iniciado | FPS alvo: {fps_alvo}")
  try:
    intervalo_log_fps = max(0.5, float(os.getenv("BOT_LOG_FPS_INTERVALO", "2.0")))
  except ValueError:
    intervalo_log_fps = 2.0
  exibir_debug = os.getenv("BOT_DEBUG", "1").strip() == "1"
  try:
    qtd_filtros_snapshot = max(1, int(os.getenv("BOT_DEBUG_SNAPSHOT_FILTROS", "3")))
  except ValueError:
    qtd_filtros_snapshot = 3
  if exibir_debug:
    log("Tela debug da rede (perceptron) ativada no treino observacao.")
    observador_rede = CerebroIA(modo_treino="treino")
  else:
    log("Tela debug da rede desativada (BOT_DEBUG=0).")
    observador_rede = None
  tecla_snapshot = (os.getenv("BOT_DEBUG_SNAPSHOT_KEY", "p").strip() or "p")[0].lower()
  if exibir_debug:
    log(
      f"Debug filtro ativo. Pressione '{tecla_snapshot.upper()}' "
      "na janela de debug para imprimir snapshot no console."
    )

  capturador = CapturaAssincrona(monitor)
  coletor = ColetorDemonstracao()
  coletor.iniciar_listener()
  capturador.iniciar()

  em_luta_ativo = False
  observacao_pausada = False
  ultimo_info_vida = None
  ultimo_log_fps_ts = 0.0
  log("pausar-luta-button ausente -> modo ESPERAR | observacao pausada.")
  observacao_pausada = True

  try:
    while True:
      inicio_ciclo = controle_fps.iniciar_ciclo()
      estado_luta = {
        "nocaute": False,
        "vitoria": False,
        "em_luta": False,
      }
      try:
        frame = capturador.obter_frame(timeout=0.05)
        if frame is None:
          continue

        estado_luta = obter_estado_luta(frame)
        done = bool(estado_luta.get("nocaute") or estado_luta.get("vitoria"))
        info_vida = obter_info_vida(frame) if estado_luta.get("em_luta") else (ultimo_info_vida or {})

        if estado_luta.get("em_luta") and not em_luta_ativo:
          em_luta_ativo = True
          coletor.reset_episodio()
          log("Luta detectada pelo botao pausar-luta-button. Observacao da IA ativada.")
          observacao_pausada = False

        if em_luta_ativo:
          terminal_label = "vitoria" if estado_luta.get("vitoria") else ("ko" if estado_luta.get("nocaute") else None)
          coletor.atualizar(frame, info_vida, done=done, terminal_label=terminal_label)

          if done:
            log(f"Luta terminou ({terminal_label}). Observacao pausada ate detectar pausar-luta-button novamente.")
            coletor.salvar_episodio(resultado=terminal_label or "terminal")
            em_luta_ativo = False
            observacao_pausada = True
          elif not estado_luta.get("em_luta"):
            # Saiu da luta sem detectar KO/vitoria (ex.: retorno abrupto de tela).
            log("pausar-luta-button ausente. Observacao pausada (estado interrompido).")
            coletor.salvar_episodio(resultado="interrompido")
            em_luta_ativo = False
            observacao_pausada = True
        elif (not estado_luta.get("em_luta")) and (not observacao_pausada):
          log("pausar-luta-button ausente -> modo ESPERAR | observacao pausada.")
          observacao_pausada = True

        if observador_rede is not None:
          estado_rede = observador_rede.obter_estado(
            frame=frame,
            info_vida=info_vida,
            sinais=None,
          )
          observador_rede.inferir_acao_passiva(estado_rede)
          frame_perceptron = gerar_tela_perceptron(
            observador_rede,
            titulo="PPO PIXEL PERCEPTRON (OBS)",
            linha_status=(
              f"obs={'ATIVA' if em_luta_ativo else 'PAUSADA'} | "
              f"pause_btn={'1' if estado_luta.get('em_luta') else '0'}"
            ),
            tecla_snapshot=tecla_snapshot,
          )
          frame_pixel = gerar_tela_pixel(observador_rede, titulo="PPO PIXEL INPUT (OBS)")
          tecla_debug = exibir_multiplas(
            {
              NOME_JANELA: _preparar_frame_debug(frame.copy()),
              NOME_JANELA_PIXEL: frame_pixel,
              NOME_JANELA_PERCEPTRON: frame_perceptron,
            },
            overlay_metricas_em={NOME_JANELA},
          )
          if tecla_debug and str(tecla_debug).lower() == tecla_snapshot:
            log(
              f"[SNAPSHOT_FILTRO] tecla '{tecla_snapshot.upper()}' detectada. "
              "Imprimindo filtros aprendidos."
            )
            observador_rede.imprimir_snapshot_filtros(
              origem="treino_observacao",
              max_filtros=qtd_filtros_snapshot,
            )

        ultimo_info_vida = info_vida
      finally:
        controle_fps.finalizar_ciclo(inicio_ciclo)
        metricas = controle_fps.obter_metricas()
        atualizar_metricas(metricas)
        agora = time.perf_counter()
        if (
          metricas.get("fps_real", 0.0) > 0.0
          and (agora - ultimo_log_fps_ts) >= intervalo_log_fps
        ):
          ultimo_log_fps_ts = agora
          status_obs = "ATIVA" if em_luta_ativo else "PAUSADA"
          pause_btn = "1" if estado_luta.get("em_luta") else "0"
          log(
            "[OBS_FPS]",
            f"real={metricas['fps_real']:.1f}",
            f"alvo={metricas['fps_alvo']}",
            f"proc_ms={metricas['proc_ms']:.2f}",
            f"sleep_ms={metricas['sleep_ms']:.2f}",
            f"obs={status_obs}",
            f"pause_btn={pause_btn}",
          )
  finally:
    if em_luta_ativo:
      coletor.salvar_episodio(resultado="interrompido")
    coletor.parar_listener()
    capturador.parar()
