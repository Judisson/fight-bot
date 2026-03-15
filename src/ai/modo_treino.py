from enum import Enum

from src.ai.acoes import ACOES_IA


class ModoTreino(str, Enum):
  TREINO = "treino"


_ALIASES_TREINO = {
  "",
  "treino",
}


def resolver_modo_treino(valor):
  if isinstance(valor, ModoTreino):
    return valor

  texto = str(valor or "").strip().lower()
  if texto in _ALIASES_TREINO:
    return ModoTreino.TREINO

  return ModoTreino.TREINO


def obter_acoes_permitidas(modo_treino):
  _ = resolver_modo_treino(modo_treino)
  return list(ACOES_IA)
