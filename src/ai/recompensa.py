import os

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENSIVO, ACAO_DESTREZA
from src.ai.deteccao_luta import TEMPLATE_DERROTA, TEMPLATE_VITORIA
from src.bot.visao import encontrar_template

PESO_KO = -50.0
PESO_VITORIA = 50.0
PESO_PERDA_VIDA = -5.0
PESO_DANO_INIMIGO_POR_PONTO = 1.2
PESO_DESTREZA_PERFEITA = 2.5
PESO_DESTREZA_SEM_PERFEICAO = -2.5
PESO_BASE_SEM_EVENTO = 0.0

try:
  LIMIAR_DESTREZA = float(os.getenv("BOT_LIMIAR_DESTREZA", "0.8"))
except ValueError:
  LIMIAR_DESTREZA = 0.8


def _detectar_destreza_perfeita(frame):
  return encontrar_template(frame, "assets/destreza.png", limiar=LIMIAR_DESTREZA) is not None


def obter_recompensa(
  frame,
  acao_atual=None,
  adversario_com_especial=False,
  vida_jogador_atual=None,
  vida_jogador_anterior=None,
  vida_inimigo_atual=None,
  vida_inimigo_anterior=None,
  colunas_escuras_jogador_finais=0,
  colunas_escuras_inimigo_finais=0,
  nocaute_detectado=None,
  vitoria_detectada=None,
):
  if nocaute_detectado is None:
    nocaute = (frame is not None) and (encontrar_template(frame, TEMPLATE_DERROTA) is not None)
  else:
    nocaute = bool(nocaute_detectado)

  if vitoria_detectada is None:
    vitoria = (frame is not None) and (encontrar_template(frame, TEMPLATE_VITORIA) is not None)
  else:
    vitoria = bool(vitoria_detectada)

  houve_destreza = acao_atual in {ACAO_DESTREZA, ACAO_COMBO_SEGURO, ACAO_DEFENSIVO}
  info = {
    "nocaute": False,
    "vitoria": False,
    "perdeu_vida": False,
    "causou_dano": False,
    "acao_destreza": houve_destreza,
    "destreza_perfeita": False,
    "destreza_sem_penalidade": False,
    "delta_dano_inimigo": 0.0,
  }

  if nocaute:
    info["nocaute"] = True
    return PESO_KO, True, info

  if vitoria:
    info["vitoria"] = True
    return PESO_VITORIA, True, info

  recompensa = 0.0

  if (
    vida_jogador_atual is not None
    and vida_jogador_anterior is not None
    and vida_jogador_atual < vida_jogador_anterior
    and colunas_escuras_jogador_finais >= 1
  ):
    recompensa += PESO_PERDA_VIDA
    info["perdeu_vida"] = True

  if (
    vida_inimigo_atual is not None
    and vida_inimigo_anterior is not None
    and vida_inimigo_atual < vida_inimigo_anterior
    and colunas_escuras_inimigo_finais >= 1
  ):
    delta_dano = float(vida_inimigo_anterior) - float(vida_inimigo_atual)
    recompensa += delta_dano * PESO_DANO_INIMIGO_POR_PONTO
    info["causou_dano"] = True
    info["delta_dano_inimigo"] = delta_dano

  if houve_destreza:
    destreza_perfeita = _detectar_destreza_perfeita(frame)
    info["destreza_perfeita"] = destreza_perfeita
    sem_penalidade = (acao_atual == ACAO_COMBO_SEGURO) or bool(adversario_com_especial)
    info["destreza_sem_penalidade"] = sem_penalidade

    if destreza_perfeita:
      recompensa += PESO_DESTREZA_PERFEITA
    elif not sem_penalidade:
      recompensa += PESO_DESTREZA_SEM_PERFEICAO

  if recompensa == 0.0:
    recompensa = PESO_BASE_SEM_EVENTO

  return recompensa, False, info
