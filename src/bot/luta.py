import os
import time

import cv2

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENSIVO, ACAO_DESTREZA
from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import obter_estado_luta
from src.ai.recompensa import PESO_KO, PESO_VITORIA, obter_recompensa
from src.bot.acoes_luta import executar_acao, obter_acao_em_execucao
from src.bot.poder import desenhar_info_poder, obter_info_poder
from src.bot.vida import desenhar_info_vida, obter_info_vida
from src.bot.visao import encontrar_templates_em_paralelo
from src.utils.log import log
from src.utils.visao_debug import exibir
from src.visao.personagens import RastreadorPersonagens, desenhar_info_personagens


class Luta:

  def __init__(self, exibir_debug=True):
    self.exibir_debug = exibir_debug
    self.cerebro = CerebroIA()
    self.estado = "BUSCANDO_LUTA"
    self.episodio = 0
    self.vida_jogador_anterior = None
    self.vida_inimigo_anterior = None

    self._estado_anterior = None
    self._acao_anterior = None
    self._adversario_com_especial_anterior = None
    self._ultima_recompensa = 0.0

    try:
      self._log_luta_cada = max(1, int(os.getenv("BOT_LOG_LUTA_CADA", "30")))
    except ValueError:
      self._log_luta_cada = 30
    try:
      self._debug_downscale = max(1, int(os.getenv("BOT_DEBUG_DOWNSCALE", "2")))
    except ValueError:
      self._debug_downscale = 2

    self._contador_frames = 0
    self._mensagem_evento_debug = ""
    self._mensagem_evento_debug_restante = 0
    self._distancia_norm_anterior = None
    self._frames_ataque_jogador_restante = 0
    self._frames_sem_movimento = 0

    try:
      self._duracao_ataque_frames = max(1, int(os.getenv("BOT_ATAQUE_FRAMES", "4")))
    except ValueError:
      self._duracao_ataque_frames = 4
    try:
      self._janela_sem_movimento = max(1, int(os.getenv("BOT_IDLE_FRAMES", "8")))
    except ValueError:
      self._janela_sem_movimento = 8
    try:
      self._limiar_movimento_dist = max(0.001, float(os.getenv("BOT_IDLE_DIST_DELTA", "0.015")))
    except ValueError:
      self._limiar_movimento_dist = 0.015
    try:
      self._limiar_template_especial = float(os.getenv("BOT_LIMIAR_ESPECIAL", "0.8"))
    except ValueError:
      self._limiar_template_especial = 0.8

    self._template_especial_jogador = os.getenv(
      "BOT_TEMPLATE_ESPECIAL_JOGADOR",
      "assets/especial-jogador.png",
    ).strip()
    self._template_especial_inimigo = os.getenv(
      "BOT_TEMPLATE_ESPECIAL_INIMIGO",
      "assets/especial-inimigo.png",
    ).strip()
    self._tem_template_especial_jogador = os.path.exists(self._template_especial_jogador)
    self._tem_template_especial_inimigo = os.path.exists(self._template_especial_inimigo)

    self.rastreador_personagens = RastreadorPersonagens(
      usar_yolo=True,
      caminho_modelo="src/modelos/personagens.pt",
      somente_yolo=True,
    )

  def processar_frame(self, frame):
    estado_luta = obter_estado_luta(frame)
    em_luta = estado_luta["em_luta"]

    terminal_recompensa = None
    terminal_texto = None
    if estado_luta["nocaute"]:
      terminal_recompensa = PESO_KO
      terminal_texto = "K.O detectado"
    elif estado_luta.get("vitoria"):
      terminal_recompensa = PESO_VITORIA
      terminal_texto = "Vitoria detectada"

    if terminal_recompensa is not None:
      if self._estado_anterior is not None and self._acao_anterior is not None:
        self.cerebro.aprender(
          self._estado_anterior,
          self._acao_anterior,
          terminal_recompensa,
          proximo_estado=None,
          terminal=True,
        )

      log(terminal_texto)
      self.estado = "BUSCANDO_LUTA"
      self.episodio += 1
      self.vida_jogador_anterior = None
      self.vida_inimigo_anterior = None
      self._estado_anterior = None
      self._acao_anterior = None
      self._adversario_com_especial_anterior = None
      self._ultima_recompensa = 0.0
      self._contador_frames = 0
      self._mensagem_evento_debug = ""
      self._mensagem_evento_debug_restante = 0
      self._distancia_norm_anterior = None
      self._frames_ataque_jogador_restante = 0
      self._frames_sem_movimento = 0
      self.rastreador_personagens.reset()
      log("Episodio finalizado:", self.episodio)

      if self.exibir_debug:
        exibir(self._preparar_frame_debug(frame))

      time.sleep(3)
      return

    if not em_luta:
      self.rastreador_personagens.reset()
      self._estado_anterior = None
      self._acao_anterior = None
      self._adversario_com_especial_anterior = None
      self._mensagem_evento_debug = ""
      self._mensagem_evento_debug_restante = 0
      self._distancia_norm_anterior = None
      self._frames_ataque_jogador_restante = 0
      self._frames_sem_movimento = 0

      if self.exibir_debug:
        exibir(self._preparar_frame_debug(frame))
      return

    if self._frames_ataque_jogador_restante > 0:
      self._frames_ataque_jogador_restante -= 1

    info_vida = obter_info_vida(frame)
    info_poder = obter_info_poder(frame)
    info_personagens = self.rastreador_personagens.detectar(frame)
    sinais_estado = self._detectar_sinais_estado(frame, info_vida, info_poder, info_personagens)
    estado_atual = self.cerebro.obter_estado(
      info_vida=info_vida,
      info_personagens=info_personagens,
      sinais=sinais_estado,
    )

    if self.estado != "LUTANDO":
      log("Luta comecou")
      self.estado = "LUTANDO"

    info_recompensa = {
      "destreza_perfeita": False,
      "acao_destreza": False,
    }

    if self._estado_anterior is not None and self._acao_anterior is not None:
      recompensa, _, info_recompensa = obter_recompensa(
        frame,
        acao_atual=self._acao_anterior,
        adversario_com_especial=bool(self._adversario_com_especial_anterior),
        vida_jogador_atual=info_vida["vida_jogador_pct"],
        vida_jogador_anterior=self.vida_jogador_anterior,
        vida_inimigo_atual=info_vida["vida_inimigo_pct"],
        vida_inimigo_anterior=self.vida_inimigo_anterior,
        colunas_escuras_jogador_finais=info_vida["colunas_escuras_jogador_finais"],
        colunas_escuras_inimigo_finais=info_vida["colunas_escuras_inimigo_finais"],
        nocaute_detectado=estado_luta["nocaute"],
        vitoria_detectada=estado_luta.get("vitoria"),
      )
      self._ultima_recompensa = recompensa
      self.cerebro.aprender(
        self._estado_anterior,
        self._acao_anterior,
        recompensa,
        proximo_estado=estado_atual,
        terminal=False,
      )
      self._atualizar_evento_debug(info_recompensa)

    adversario_com_especial = bool(sinais_estado.get("adversario_com_especial"))
    eu_tenho_especial = bool(sinais_estado.get("eu_tenho_especial"))
    acao = self.cerebro.escolher_acao(estado_atual)
    if adversario_com_especial:
      acao = ACAO_DEFENSIVO

    acao_executada = executar_acao(acao, pode_soltar_especial=eu_tenho_especial)
    acao_registrada = acao if acao_executada else obter_acao_em_execucao()
    if acao_executada and acao_registrada == ACAO_DESTREZA:
      self._frames_ataque_jogador_restante = self._duracao_ataque_frames
    elif acao_executada and acao_registrada == ACAO_COMBO_SEGURO:
      self._frames_ataque_jogador_restante = max(self._duracao_ataque_frames, 12)
    elif acao_executada and acao_registrada == ACAO_DEFENSIVO:
      self._frames_ataque_jogador_restante = max(self._duracao_ataque_frames, 4)

    self._estado_anterior = estado_atual
    self._acao_anterior = acao_registrada
    self._adversario_com_especial_anterior = adversario_com_especial

    if self.exibir_debug:
      frame_debug = desenhar_info_vida(frame.copy(), info_vida)
      frame_debug = desenhar_info_poder(frame_debug, info_poder)
      frame_debug = desenhar_info_personagens(frame_debug, info_personagens)
      frame_debug = self._desenhar_evento_debug(frame_debug)
      exibir(self._preparar_frame_debug(frame_debug))

    self._contador_frames += 1
    if (self._contador_frames % self._log_luta_cada) == 0:
      log(
        "ACAO:",
        acao_registrada,
        "RECOMPENSA:",
        self._ultima_recompensa,
      )
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
      if info_personagens:
        log("DISTANCIA_HEROIS:", f"{info_personagens['distancia_px']}px")
      if info_recompensa.get("destreza_perfeita"):
        log("EVENTO: DESTREZA_PERFEITA (+2.5)")
      elif info_recompensa.get("destreza_sem_penalidade"):
        log("EVENTO: DESTREZA_SEM_PENALIDADE (combo/defensivo)")
      elif info_recompensa.get("acao_destreza"):
        log("EVENTO: DESTREZA_SEM_NECESSIDADE (-2.5)")
      if info_recompensa.get("causou_dano"):
        log("DANO_INIMIGO_DELTA:", f"{info_recompensa.get('delta_dano_inimigo', 0.0):.2f}%")

    self.vida_jogador_anterior = info_vida["vida_jogador_pct"]
    self.vida_inimigo_anterior = info_vida["vida_inimigo_pct"]

  def _atualizar_evento_debug(self, info_recompensa):
    if info_recompensa.get("destreza_perfeita"):
      self._mensagem_evento_debug = "DESTREZA PERFEITA +2.5"
      self._mensagem_evento_debug_restante = 20
      return

    if info_recompensa.get("acao_destreza"):
      if info_recompensa.get("destreza_sem_penalidade"):
        self._mensagem_evento_debug = "DESTREZA NO COMBO/DEFESA (SEM PENALIDADE)"
        self._mensagem_evento_debug_restante = 20
        return
      self._mensagem_evento_debug = "DESTREZA SEM NECESSIDADE -2.5"
      self._mensagem_evento_debug_restante = 20
      return

    if self._mensagem_evento_debug_restante > 0:
      self._mensagem_evento_debug_restante -= 1

  def _detectar_especiais(self, frame):
    consultas = []
    if self._tem_template_especial_jogador:
      consultas.append(("esp_eu", self._template_especial_jogador, self._limiar_template_especial))
    if self._tem_template_especial_inimigo:
      consultas.append(("esp_adv", self._template_especial_inimigo, self._limiar_template_especial))

    if not consultas:
      return None, None

    resultados = encontrar_templates_em_paralelo(frame, consultas)
    eu_tenho_especial = (
      resultados.get("esp_eu") is not None
      if self._tem_template_especial_jogador
      else None
    )
    adversario_com_especial = (
      resultados.get("esp_adv") is not None
      if self._tem_template_especial_inimigo
      else None
    )
    return eu_tenho_especial, adversario_com_especial

  def _detectar_sinais_estado(self, frame, info_vida, info_poder, info_personagens):
    vida_jogador = info_vida.get("vida_jogador_pct")
    vida_inimigo = info_vida.get("vida_inimigo_pct")
    distancia_norm = None
    if info_personagens:
      distancia_norm = info_personagens.get("distancia_norm")

    nivel_especial_jogador = info_poder.get("nivel_especial_jogador")
    nivel_especial_inimigo = info_poder.get("nivel_especial_inimigo")
    eu_tenho_especial = info_poder.get("tem_especial_jogador")
    adversario_com_especial = info_poder.get("tem_especial_inimigo")
    if (
      eu_tenho_especial is None
      or adversario_com_especial is None
      or nivel_especial_jogador is None
      or nivel_especial_inimigo is None
    ):
      tpl_eu, tpl_adv = self._detectar_especiais(frame)
      if eu_tenho_especial is None:
        eu_tenho_especial = tpl_eu
      if adversario_com_especial is None:
        adversario_com_especial = tpl_adv
      if nivel_especial_jogador is None and tpl_eu is not None:
        nivel_especial_jogador = 1 if tpl_eu else 0
      if nivel_especial_inimigo is None and tpl_adv is not None:
        nivel_especial_inimigo = 1 if tpl_adv else 0

    inimigo_atacando = False
    if (
      vida_jogador is not None
      and self.vida_jogador_anterior is not None
      and vida_jogador < self.vida_jogador_anterior
      and info_vida.get("colunas_escuras_jogador_finais", 0) >= 1
    ):
      inimigo_atacando = True

    eu_atacando = self._frames_ataque_jogador_restante > 0

    mudou_vida = False
    if (
      vida_jogador is not None
      and self.vida_jogador_anterior is not None
      and vida_jogador != self.vida_jogador_anterior
    ):
      mudou_vida = True
    if (
      vida_inimigo is not None
      and self.vida_inimigo_anterior is not None
      and vida_inimigo != self.vida_inimigo_anterior
    ):
      mudou_vida = True

    moveu = False
    if distancia_norm is not None and self._distancia_norm_anterior is not None:
      moveu = abs(distancia_norm - self._distancia_norm_anterior) > self._limiar_movimento_dist

    if distancia_norm is not None:
      self._distancia_norm_anterior = distancia_norm

    if (not moveu) and (not mudou_vida) and (not eu_atacando) and (not inimigo_atacando):
      self._frames_sem_movimento += 1
    else:
      self._frames_sem_movimento = 0

    sem_movimento = self._frames_sem_movimento >= self._janela_sem_movimento
    if adversario_com_especial is None:
      postura_tatica = None
    elif adversario_com_especial:
      postura_tatica = "defensivo"
    else:
      postura_tatica = "agressivo"

    return {
      "eu_tenho_especial": eu_tenho_especial,
      "adversario_com_especial": adversario_com_especial,
      "nivel_especial_jogador": nivel_especial_jogador,
      "nivel_especial_inimigo": nivel_especial_inimigo,
      "postura_tatica": postura_tatica,
      "eu_atacando": eu_atacando,
      "inimigo_atacando": inimigo_atacando,
      "sem_movimento": sem_movimento,
    }

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
    # Slicing reduz carga do debug sem custo de interpolacao.
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
