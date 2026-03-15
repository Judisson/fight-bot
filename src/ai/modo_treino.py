from enum import Enum

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENDER, ACAO_DESTREZA, ACAO_ESPERAR


class ModoTreino(str, Enum):
  DESTREZA = "destreza"
  BLOQUEIO = "bloqueio"
  CONTRA_ATAQUE = "contra_ataque"
  COMPLETO = "completo"


ACOES_POR_MODO = {
  ModoTreino.DESTREZA: [ACAO_ESPERAR, ACAO_DESTREZA],
  ModoTreino.BLOQUEIO: [ACAO_ESPERAR, ACAO_DEFENDER],
  ModoTreino.CONTRA_ATAQUE: [ACAO_ESPERAR, ACAO_DESTREZA, ACAO_DEFENDER, ACAO_COMBO_SEGURO],
  ModoTreino.COMPLETO: [ACAO_ESPERAR, ACAO_DESTREZA, ACAO_DEFENDER, ACAO_COMBO_SEGURO],
}


def resolver_modo_treino(valor):
  if isinstance(valor, ModoTreino):
    return valor

  texto = str(valor or "").strip().lower()
  for modo in ModoTreino:
    if texto == modo.value:
      return modo

  return ModoTreino.DESTREZA


def obter_acoes_permitidas(modo_treino):
  modo = resolver_modo_treino(modo_treino)
  return list(ACOES_POR_MODO.get(modo, ACOES_POR_MODO[ModoTreino.DESTREZA]))
