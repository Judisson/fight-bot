from src.bot.visao import encontrar_template


def obter_recompensa(
  frame,
  vida_jogador_atual=None,
  vida_jogador_anterior=None,
  vida_inimigo_atual=None,
  vida_inimigo_anterior=None,
  colunas_escuras_jogador_finais=0,
  colunas_escuras_inimigo_finais=0,
):
  nocaute = encontrar_template(frame, "assets/derrota.png")

  if nocaute:
    return -100, True

  recompensa = 0.0

  if (
    vida_jogador_atual is not None
    and vida_jogador_anterior is not None
  ):
    if vida_jogador_atual >= vida_jogador_anterior:
      recompensa += 1.0
    elif colunas_escuras_jogador_finais >= 1:
      recompensa -= 1.0

  if (
    vida_inimigo_atual is not None
    and vida_inimigo_anterior is not None
    and vida_inimigo_atual < vida_inimigo_anterior
    and colunas_escuras_inimigo_finais >= 1
  ):
    recompensa += 0.5

  if recompensa == 0.0:
    recompensa = -0.1

  return recompensa, False
