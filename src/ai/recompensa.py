import os

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_BLOQUEIO,
  ACAO_ESQUIVA,
)
from src.ai.deteccao_luta import TEMPLATE_DERROTA, TEMPLATE_VITORIA
from src.ai.modo_treino import resolver_modo_treino
from src.bot.rois import ROI_ACOES_PERFEITAS, ROI_SEQUENCIA_GOLPES, obter_roi
from src.bot.visao import encontrar_template

PESO_KO = -15.0
PESO_VITORIA = 20.0

PESO_DESTREZA_PERFEITA = 6.0
PESO_DESTREZA_QUASE = 2.0
PESO_DESTREZA_SEM_PERIGO = -1.0
PESO_DESTREZA_LONGE = -1.5
PESO_DESTREZA_LONGE_CONTEXTO = -0.3
PESO_SPAM_DESTREZA = -0.5

PESO_APARAR_PERFEITO = 3.0
PESO_BLOQUEIO_QUASE = 1.0
PESO_BLOQUEIO_SEM_PERIGO = -1.0
PESO_SPAM_BLOQUEIO = -0.5

PESO_TOMOU_DANO = 0.0  # legado
PESO_DANO_INIMIGO_POR_PCT = 1.5
PESO_DANO_TOMADO_POR_PCT = -2.0
PESO_TENTOU_REAGIR_ATAQUE = 0.5
PESO_NAO_REAGIU_ATAQUE = -0.2
PESO_SOBREVIVEU_JANELA_PERIGOSA = 0.5
PESO_PARADO_MUITO_TEMPO = -0.5
PESO_ATAQUE_EM_PERIGO = -0.2
PESO_ATAQUE_PESADO_TOMOU_DANO = -10.0

RECOMPENSA_COMBO_ATIVO = 5.0
PENALIDADE_SEM_COMBO = -1.0

PESO_BLOQUEIO_PERFEITO = PESO_APARAR_PERFEITO

try:
  LIMIAR_DISTANCIA_DESTREZA_RUIM = int(os.getenv("BOT_DISTANCIA_DESTREZA_RUIM_PX", "300"))
except ValueError:
  LIMIAR_DISTANCIA_DESTREZA_RUIM = 300

try:
  LIMIAR_PARADO_FRAMES = int(os.getenv("BOT_LIMIAR_PARADO_FRAMES", "45"))
except ValueError:
  LIMIAR_PARADO_FRAMES = 45

try:
  LIMIAR_DESTREZA = float(os.getenv("BOT_LIMIAR_DESTREZA", "0.8"))
except ValueError:
  LIMIAR_DESTREZA = 0.8

try:
  LIMIAR_APARAR = float(os.getenv("BOT_LIMIAR_APARAR", "0.8"))
except ValueError:
  LIMIAR_APARAR = 0.8

try:
  LIMIAR_COMBO = float(os.getenv("BOT_LIMIAR_COMBO", "0.8"))
except ValueError:
  LIMIAR_COMBO = 0.8

def _detectar_destreza_perfeita(frame):
  roi = obter_roi(ROI_ACOES_PERFEITAS)
  return encontrar_template(frame, "assets/destreza.png", limiar=LIMIAR_DESTREZA, roi=roi) is not None


def _detectar_aparar_perfeito(frame):
  roi = obter_roi(ROI_ACOES_PERFEITAS)
  return encontrar_template(frame, "assets/aparar.png", limiar=LIMIAR_APARAR, roi=roi) is not None


def _detectar_combo_ativo(frame):
  roi = obter_roi(ROI_SEQUENCIA_GOLPES)
  return encontrar_template(
    frame,
    "assets/sequencia-golpes.png",
    limiar=LIMIAR_COMBO,
    roi=roi,
  ) is not None


def _detectar_tomou_dano(vida_jogador_atual, vida_jogador_anterior, colunas_escuras_jogador_finais):
  if (
    vida_jogador_atual is None
    or vida_jogador_anterior is None
    or vida_jogador_atual >= vida_jogador_anterior
    or colunas_escuras_jogador_finais < 1
  ):
    return False, 0.0

  delta_dano = float(vida_jogador_anterior) - float(vida_jogador_atual)
  return True, max(0.0, delta_dano)


def _detectar_causou_dano(vida_inimigo_atual, vida_inimigo_anterior):
  if (
    vida_inimigo_atual is None
    or vida_inimigo_anterior is None
    or vida_inimigo_atual >= vida_inimigo_anterior
  ):
    return False, 0.0

  delta_dano = float(vida_inimigo_anterior) - float(vida_inimigo_atual)
  return True, max(0.0, delta_dano)


def _distancia_perto(distancia_px):
  if distancia_px is None:
    return None
  try:
    return float(distancia_px) <= float(LIMIAR_DISTANCIA_DESTREZA_RUIM)
  except (TypeError, ValueError):
    return None


def _base_info(acao_atual):
  return {
    "nocaute": False,
    "vitoria": False,
    "tomou_dano": False,
    "tomou_dano_delta": 0.0,
    "causou_dano": False,
    "dano_inimigo_delta": 0.0,
    "acao_esquiva": acao_atual == ACAO_ESQUIVA,
    "acao_destreza": acao_atual == ACAO_ESQUIVA,
    "acao_bloqueio": acao_atual == ACAO_BLOQUEIO,
    "acao_ataque": acao_atual in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO),
    "ataque_pesado_tentado": acao_atual == ACAO_ATAQUE_PESADO,
    "esquiva_tentada": False,
    "destreza_tentada": False,
    "destreza_perfeita": False,
    "destreza_quase": False,
    "destreza_errada": False,
    "aparar_perfeito": False,
    "bloqueio_tentado": False,
    "bloqueio_quase": False,
    "bloqueio_errado": False,
    "sobreviveu_perigo": False,
    "tentou_reagir_ataque": False,
    "nao_reagiu_ataque": False,
    "spam_destreza": False,
    "spam_bloqueio": False,
    "destreza_longe_ruim": False,
    "parado_muito_tempo": False,
    "combo_ativo": False,
    "combo_bonus": 0.0,
    "combo_penalidade": 0.0,
    "ataque_pesado_tomou_dano": False,
    "ataque_em_perigo": False,
  }


def _recompensa_acao(
  info,
  frame,
  acao_atual,
  inimigo_atacando,
  tomou_dano,
  distancia_px,
):
  recompensa = 0.0

  if acao_atual == ACAO_ESQUIVA:
    info["esquiva_tentada"] = True
    info["destreza_tentada"] = True
    dist_perto = _distancia_perto(distancia_px)
    destreza_longe_ruim = dist_perto is False
    if destreza_longe_ruim:
      recompensa += PESO_DESTREZA_LONGE_CONTEXTO if inimigo_atacando else PESO_DESTREZA_LONGE
      info["destreza_longe_ruim"] = True
      info["destreza_errada"] = True
      return recompensa

    destreza_perfeita = _detectar_destreza_perfeita(frame)
    info["destreza_perfeita"] = destreza_perfeita
    if destreza_perfeita:
      recompensa += PESO_DESTREZA_PERFEITA
    elif inimigo_atacando and (not tomou_dano):
      recompensa += PESO_DESTREZA_QUASE
      info["destreza_quase"] = True
    else:
      recompensa += PESO_DESTREZA_SEM_PERIGO
      info["destreza_errada"] = True
    return recompensa

  if acao_atual == ACAO_BLOQUEIO:
    info["bloqueio_tentado"] = True
    aparar_perfeito = _detectar_aparar_perfeito(frame)
    info["aparar_perfeito"] = aparar_perfeito
    if aparar_perfeito:
      recompensa += PESO_APARAR_PERFEITO
    elif inimigo_atacando and (not tomou_dano):
      recompensa += PESO_BLOQUEIO_QUASE
      info["bloqueio_quase"] = True
    else:
      recompensa += PESO_BLOQUEIO_SEM_PERIGO
      info["bloqueio_errado"] = True
    return recompensa

  if acao_atual == ACAO_ATAQUE_PESADO and tomou_dano:
    recompensa += PESO_ATAQUE_PESADO_TOMOU_DANO
    info["ataque_pesado_tomou_dano"] = True

  if acao_atual in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO) and inimigo_atacando:
    recompensa += PESO_ATAQUE_EM_PERIGO
    info["ataque_em_perigo"] = True

  return recompensa


def _aplicar_eventos_comuns(
  info,
  recompensa,
  acao_atual,
  inimigo_atacando,
  aplicar_penalidade_nao_reagiu,
  tomou_dano,
  destreza_consecutiva,
  bloqueio_consecutivo,
  tempo_parado_frames,
):
  acao_reacao = acao_atual in (ACAO_ESQUIVA, ACAO_BLOQUEIO)
  if acao_atual == ACAO_ESQUIVA and info.get("destreza_longe_ruim"):
    acao_reacao = False

  if inimigo_atacando:
    if acao_reacao:
      recompensa += PESO_TENTOU_REAGIR_ATAQUE
      info["tentou_reagir_ataque"] = True
    elif aplicar_penalidade_nao_reagiu:
      recompensa += PESO_NAO_REAGIU_ATAQUE
      info["nao_reagiu_ataque"] = True

  if inimigo_atacando and (not tomou_dano):
    recompensa += PESO_SOBREVIVEU_JANELA_PERIGOSA
    info["sobreviveu_perigo"] = True

  if destreza_consecutiva and acao_atual == ACAO_ESQUIVA:
    recompensa += PESO_SPAM_DESTREZA
    info["spam_destreza"] = True

  if bloqueio_consecutivo and acao_atual == ACAO_BLOQUEIO:
    recompensa += PESO_SPAM_BLOQUEIO
    info["spam_bloqueio"] = True

  if tempo_parado_frames >= LIMIAR_PARADO_FRAMES:
    recompensa += PESO_PARADO_MUITO_TEMPO
    info["parado_muito_tempo"] = True

  return recompensa


def obter_recompensa(
  frame,
  acao_atual=None,
  inimigo_atacando=False,
  aplicar_penalidade_nao_reagiu=True,
  aplicar_pontuacao_combo=True,
  vida_jogador_atual=None,
  vida_jogador_anterior=None,
  vida_inimigo_atual=None,
  vida_inimigo_anterior=None,
  colunas_escuras_jogador_finais=0,
  colunas_escuras_inimigo_finais=0,
  nocaute_detectado=None,
  vitoria_detectada=None,
  destreza_consecutiva=False,
  bloqueio_consecutivo=False,
  distancia_px=None,
  modo_treino="treino",
  tempo_parado_frames=0,
  delta_tempo_seg=None,
):
  _ = (colunas_escuras_inimigo_finais,)
  _ = resolver_modo_treino(modo_treino)

  if nocaute_detectado is None:
    nocaute = (frame is not None) and (encontrar_template(frame, TEMPLATE_DERROTA) is not None)
  else:
    nocaute = bool(nocaute_detectado)

  if vitoria_detectada is None:
    vitoria = (frame is not None) and (encontrar_template(frame, TEMPLATE_VITORIA) is not None)
  else:
    vitoria = bool(vitoria_detectada)

  info = _base_info(acao_atual)
  if nocaute:
    info["nocaute"] = True
    return PESO_KO, True, info

  if vitoria:
    info["vitoria"] = True
    return PESO_VITORIA, True, info

  tomou_dano, delta_dano = _detectar_tomou_dano(
    vida_jogador_atual,
    vida_jogador_anterior,
    colunas_escuras_jogador_finais,
  )
  causou_dano, delta_inimigo = _detectar_causou_dano(
    vida_inimigo_atual,
    vida_inimigo_anterior,
  )

  _ = (delta_tempo_seg,)

  recompensa = 0.0
  if frame is not None:
    combo_ativo = _detectar_combo_ativo(frame)
    info["combo_ativo"] = bool(combo_ativo)
    if aplicar_pontuacao_combo:
      if combo_ativo:
        recompensa += RECOMPENSA_COMBO_ATIVO
        info["combo_bonus"] = RECOMPENSA_COMBO_ATIVO
      else:
        recompensa += PENALIDADE_SEM_COMBO
        info["combo_penalidade"] = PENALIDADE_SEM_COMBO

  if tomou_dano:
    recompensa += delta_dano * PESO_DANO_TOMADO_POR_PCT
    info["tomou_dano"] = True
    info["tomou_dano_delta"] = delta_dano

  if causou_dano:
    recompensa += delta_inimigo * PESO_DANO_INIMIGO_POR_PCT
    info["causou_dano"] = True
    info["dano_inimigo_delta"] = delta_inimigo

  recompensa += _recompensa_acao(
    info,
    frame,
    acao_atual,
    bool(inimigo_atacando),
    tomou_dano,
    distancia_px,
  )

  recompensa = _aplicar_eventos_comuns(
    info,
    recompensa,
    acao_atual,
    bool(inimigo_atacando),
    bool(aplicar_penalidade_nao_reagiu),
    tomou_dano,
    bool(destreza_consecutiva),
    bool(bloqueio_consecutivo),
    int(tempo_parado_frames),
  )
  return recompensa, False, info
