import os

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENDER, ACAO_DESTREZA
from src.ai.modo_treino import ModoTreino, resolver_modo_treino
from src.ai.deteccao_luta import TEMPLATE_DERROTA, TEMPLATE_VITORIA
from src.bot.visao import encontrar_template

PESO_KO = -20.0
PESO_VITORIA = 0.0

# Fase DESTREZA (curriculum)
PESO_DESTREZA_PERFEITA = 6.0
PESO_DESTREZA_QUASE = 2.0
PESO_DESTREZA_SEM_PERIGO = -1.0
PESO_DESTREZA_LONGE = -1.5
PESO_DESTREZA_LONGE_CONTEXTO = -0.3
PESO_TOMOU_DANO_DESTREZA = -5.0
PESO_SOBREVIVEU_JANELA_PERIGOSA = 0.2
PESO_TENTOU_REAGIR_ATAQUE = 0.5
PESO_NAO_REAGIU_ATAQUE = -0.2
PESO_SPAM_DESTREZA = -0.5
PESO_PARADO_MUITO_TEMPO = -0.5

# Fase BLOQUEIO (curriculum)
PESO_APARAR_PERFEITO = 3.0
PESO_BLOQUEIO_QUASE_FASE2 = 1.0
PESO_BLOQUEIO_SEM_PERIGO = -1.0
PESO_TOMOU_DANO_BLOQUEIO = -5.0
PESO_SPAM_BLOQUEIO = -0.5

# Modos mais avançados
PESO_TOMOU_DANO_PADRAO = -4.0
PESO_DANO_INIMIGO_CONTRA_ATAQUE = 2.0
PESO_COMBO_APOS_DESTREZA = 4.0

# Alias de compatibilidade para testes/regras antigas.
PESO_BLOQUEIO_PERFEITO = 6.0
PESO_BLOQUEIO_QUASE = 2.0
PESO_TOMOU_DANO = PESO_TOMOU_DANO_PADRAO

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


def _detectar_destreza_perfeita(frame):
  return encontrar_template(frame, "assets/destreza.png", limiar=LIMIAR_DESTREZA) is not None


def _detectar_aparar_perfeito(frame):
  return encontrar_template(frame, "assets/aparar.png", limiar=LIMIAR_APARAR) is not None


def _detectar_tomou_dano(vida_jogador_atual, vida_jogador_anterior, colunas_escuras_jogador_finais):
  if (
    vida_jogador_atual is None
    or vida_jogador_anterior is None
    or vida_jogador_atual >= vida_jogador_anterior
    or colunas_escuras_jogador_finais < 1
  ):
    return False, 0.0

  delta_dano = float(vida_jogador_anterior) - float(vida_jogador_atual)
  return True, delta_dano


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
    "acao_destreza": acao_atual == ACAO_DESTREZA,
    "acao_bloqueio": acao_atual == ACAO_DEFENDER,
    "destreza_tentada": False,
    "destreza_perfeita": False,
    "destreza_quase": False,
    "destreza_errada": False,
    "aparar_perfeito": False,
    "bloqueio_tentado": False,
    "bloqueio_quase": False,
    "bloqueio_errado": False,
    "combo_tentado": acao_atual == ACAO_COMBO_SEGURO,
    "combo_apos_destreza": False,
    "sobreviveu_perigo": False,
    "tentou_reagir_ataque": False,
    "nao_reagiu_ataque": False,
    "spam_destreza": False,
    "spam_bloqueio": False,
    "destreza_longe_ruim": False,
    "parado_muito_tempo": False,
  }


def _aplicar_eventos_comuns(
  info,
  recompensa,
  acao_atual,
  modo_treino,
  inimigo_atacando,
  tomou_dano,
  destreza_consecutiva,
  bloqueio_consecutivo,
  tempo_parado_frames,
):
  if modo_treino == ModoTreino.DESTREZA:
    acao_reacao = acao_atual == ACAO_DESTREZA
  elif modo_treino == ModoTreino.BLOQUEIO:
    acao_reacao = acao_atual == ACAO_DEFENDER
  else:
    acao_reacao = acao_atual in (ACAO_DESTREZA, ACAO_DEFENDER)

  if acao_atual == ACAO_DESTREZA and info.get("destreza_longe_ruim"):
    acao_reacao = False
  if inimigo_atacando:
    if acao_reacao:
      recompensa += PESO_TENTOU_REAGIR_ATAQUE
      info["tentou_reagir_ataque"] = True
    else:
      recompensa += PESO_NAO_REAGIU_ATAQUE
      info["nao_reagiu_ataque"] = True

  if inimigo_atacando and (not tomou_dano):
    recompensa += PESO_SOBREVIVEU_JANELA_PERIGOSA
    info["sobreviveu_perigo"] = True

  if destreza_consecutiva and acao_atual == ACAO_DESTREZA:
    recompensa += PESO_SPAM_DESTREZA
    info["spam_destreza"] = True
  if bloqueio_consecutivo and acao_atual == ACAO_DEFENDER:
    recompensa += PESO_SPAM_BLOQUEIO
    info["spam_bloqueio"] = True

  if tempo_parado_frames >= LIMIAR_PARADO_FRAMES:
    recompensa += PESO_PARADO_MUITO_TEMPO
    info["parado_muito_tempo"] = True

  return recompensa


def _recompensa_modo_destreza(
  info,
  frame,
  acao_atual,
  inimigo_atacando,
  tomou_dano,
  distancia_px,
):
  recompensa = 0.0
  dist_perto = _distancia_perto(distancia_px)
  destreza_longe_ruim = (
    acao_atual == ACAO_DESTREZA
    and dist_perto is False
  )

  if acao_atual == ACAO_DESTREZA:
    info["destreza_tentada"] = True
    if destreza_longe_ruim:
      if inimigo_atacando:
        recompensa += PESO_DESTREZA_LONGE_CONTEXTO
      else:
        recompensa += PESO_DESTREZA_LONGE
      info["destreza_longe_ruim"] = True
      info["destreza_errada"] = True
    else:
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


def _recompensa_modo_bloqueio(info, frame, acao_atual, inimigo_atacando, tomou_dano):
  recompensa = 0.0

  if acao_atual == ACAO_DEFENDER:
    info["bloqueio_tentado"] = True
    aparar_perfeito = _detectar_aparar_perfeito(frame)
    info["aparar_perfeito"] = aparar_perfeito
    if aparar_perfeito:
      recompensa += PESO_APARAR_PERFEITO
    elif inimigo_atacando and (not tomou_dano):
      recompensa += PESO_BLOQUEIO_QUASE_FASE2
      info["bloqueio_quase"] = True
    else:
      recompensa += PESO_BLOQUEIO_SEM_PERIGO
      info["bloqueio_errado"] = True

  return recompensa


def _recompensa_modo_completo(
  info,
  frame,
  acao_atual,
  inimigo_atacando,
  tomou_dano,
  distancia_px,
):
  recompensa = 0.0
  dist_perto = _distancia_perto(distancia_px)
  destreza_longe_ruim = (
    acao_atual == ACAO_DESTREZA
    and dist_perto is False
  )

  if acao_atual == ACAO_DESTREZA:
    info["destreza_tentada"] = True
    if destreza_longe_ruim:
      recompensa += PESO_DESTREZA_LONGE
      info["destreza_longe_ruim"] = True
      info["destreza_errada"] = True
    else:
      destreza_perfeita = _detectar_destreza_perfeita(frame)
      info["destreza_perfeita"] = destreza_perfeita
      if destreza_perfeita:
        recompensa += PESO_DESTREZA_PERFEITA
      elif inimigo_atacando and (not tomou_dano):
        recompensa += PESO_DESTREZA_QUASE
        info["destreza_quase"] = True
      elif not inimigo_atacando:
        recompensa += PESO_DESTREZA_SEM_PERIGO
        info["destreza_errada"] = True

  if acao_atual == ACAO_DEFENDER:
    info["bloqueio_tentado"] = True
    aparar_perfeito = _detectar_aparar_perfeito(frame)
    info["aparar_perfeito"] = aparar_perfeito
    if aparar_perfeito:
      recompensa += PESO_BLOQUEIO_PERFEITO
    elif inimigo_atacando and (not tomou_dano):
      recompensa += PESO_BLOQUEIO_QUASE
      info["bloqueio_quase"] = True
    elif not inimigo_atacando:
      recompensa += PESO_BLOQUEIO_SEM_PERIGO
      info["bloqueio_errado"] = True

  return recompensa


def obter_recompensa(
  frame,
  acao_atual=None,
  inimigo_atacando=False,
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
  combo_apos_destreza=False,
  distancia_px=None,
  modo_treino=ModoTreino.COMPLETO,
  tempo_parado_frames=0,
):
  _ = (colunas_escuras_inimigo_finais,)

  modo = resolver_modo_treino(modo_treino)

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
  if tomou_dano:
    info["tomou_dano"] = True
    info["tomou_dano_delta"] = delta_dano

  recompensa = 0.0
  if modo == ModoTreino.DESTREZA:
    if tomou_dano:
      recompensa += PESO_TOMOU_DANO_DESTREZA
    recompensa += _recompensa_modo_destreza(
      info,
      frame,
      acao_atual,
      inimigo_atacando,
      tomou_dano,
      distancia_px,
    )
  elif modo == ModoTreino.BLOQUEIO:
    if tomou_dano:
      recompensa += PESO_TOMOU_DANO_BLOQUEIO
    recompensa += _recompensa_modo_bloqueio(
      info,
      frame,
      acao_atual,
      inimigo_atacando,
      tomou_dano,
    )
  else:
    if tomou_dano:
      recompensa += PESO_TOMOU_DANO_PADRAO
    recompensa += _recompensa_modo_completo(
      info,
      frame,
      acao_atual,
      inimigo_atacando,
      tomou_dano,
      distancia_px,
    )
    if (
      modo == ModoTreino.CONTRA_ATAQUE
      and vida_inimigo_atual is not None
      and vida_inimigo_anterior is not None
      and vida_inimigo_atual < vida_inimigo_anterior
    ):
      recompensa += PESO_DANO_INIMIGO_CONTRA_ATAQUE
    if modo == ModoTreino.CONTRA_ATAQUE and acao_atual == ACAO_COMBO_SEGURO and combo_apos_destreza:
      recompensa += PESO_COMBO_APOS_DESTREZA
      info["combo_apos_destreza"] = True

  recompensa = _aplicar_eventos_comuns(
    info,
    recompensa,
    acao_atual,
    modo,
    inimigo_atacando,
    tomou_dano,
    destreza_consecutiva,
    bloqueio_consecutivo,
    tempo_parado_frames,
  )

  return recompensa, False, info
