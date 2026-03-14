import os

from src.ai.acoes import ACAO_DESTREZA
from src.ai.deteccao_luta import TEMPLATE_DERROTA, TEMPLATE_VITORIA
from src.bot.visao import encontrar_template

# ETAPA 1 / FASE 1 (ATIVA): foco em sobrevivencia e tempo de resposta.
PESO_KO = -20.0
PESO_VITORIA = 0.0
PESO_TOMOU_DANO = -4.0
PESO_DESTREZA_PERFEITA = 3.0
PESO_SOBREVIVEU_JANELA_PERIGOSA = 0.2
PESO_DESTREZA_SEM_PERIGO = -1.0

# FASE 2 (PENDENTE): introduzir bonus/penalidade de punicao ofensiva.
# PESO_ACERTOU_PUNICAO = 1.0
# PESO_TOMOU_CONTRA_ATAQUE = -2.0

# FASE 3 (PENDENTE): considerar vitoria como bonus final da luta.
# PESO_VITORIA = 50.0

try:
  LIMIAR_DESTREZA = float(os.getenv("BOT_LIMIAR_DESTREZA", "0.8"))
except ValueError:
  LIMIAR_DESTREZA = 0.8


def _detectar_destreza_perfeita(frame):
  return encontrar_template(frame, "assets/destreza.png", limiar=LIMIAR_DESTREZA) is not None


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
):
  _ = (vida_inimigo_atual, vida_inimigo_anterior, colunas_escuras_inimigo_finais)

  if nocaute_detectado is None:
    nocaute = (frame is not None) and (encontrar_template(frame, TEMPLATE_DERROTA) is not None)
  else:
    nocaute = bool(nocaute_detectado)

  if vitoria_detectada is None:
    vitoria = (frame is not None) and (encontrar_template(frame, TEMPLATE_VITORIA) is not None)
  else:
    vitoria = bool(vitoria_detectada)

  info = {
    "nocaute": False,
    "vitoria": False,
    "tomou_dano": False,
    "tomou_dano_delta": 0.0,
    "acao_destreza": acao_atual == ACAO_DESTREZA,
    "destreza_perfeita": False,
    "sobreviveu_perigo": False,
  }

  if nocaute:
    info["nocaute"] = True
    return PESO_KO, True, info

  if vitoria:
    info["vitoria"] = True
    return PESO_VITORIA, True, info

  recompensa = 0.0
  tomou_dano = False
  delta_dano = 0.0

  if (
    vida_jogador_atual is not None
    and vida_jogador_anterior is not None
    and vida_jogador_atual < vida_jogador_anterior
    and colunas_escuras_jogador_finais >= 1
  ):
    tomou_dano = True
    delta_dano = float(vida_jogador_anterior) - float(vida_jogador_atual)
    recompensa += PESO_TOMOU_DANO
    info["tomou_dano"] = True
    info["tomou_dano_delta"] = delta_dano

  if acao_atual == ACAO_DESTREZA:
    destreza_perfeita = _detectar_destreza_perfeita(frame)
    info["destreza_perfeita"] = destreza_perfeita
    if destreza_perfeita:
      recompensa += PESO_DESTREZA_PERFEITA
    elif not inimigo_atacando:
      recompensa += PESO_DESTREZA_SEM_PERIGO

  if inimigo_atacando and (not tomou_dano):
    recompensa += PESO_SOBREVIVEU_JANELA_PERIGOSA
    info["sobreviveu_perigo"] = True

  return recompensa, False, info
