from enum import Enum

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_BLOQUEIO,
  ACAO_ESPERAR,
  ACAO_ESQUIVA,
  ACOES_IA,
)


class ModoTreino(str, Enum):
  COMPLETO = "completo"
  APARAR = "aparar"
  DEFENDER = "defender"
  COMBO = "combo"
  DESTREZA = "destreza"
  ASSISTIDO = "assistido"


_ALIASES_TREINO = {
  "": ModoTreino.COMPLETO,
  "completo": ModoTreino.COMPLETO,
  "aparar": ModoTreino.APARAR,
  "defender": ModoTreino.DEFENDER,
  "combo": ModoTreino.COMBO,
  "destreza": ModoTreino.DESTREZA,
  "assistido": ModoTreino.ASSISTIDO,
}


def resolver_modo_treino(valor):
  if isinstance(valor, ModoTreino):
    return valor

  texto = str(valor or "").strip().lower()
  return _ALIASES_TREINO.get(texto, ModoTreino.COMPLETO)


def obter_acoes_permitidas(modo_treino):
  modo = resolver_modo_treino(modo_treino)
  if modo == ModoTreino.APARAR:
    return [ACAO_ESPERAR, ACAO_BLOQUEIO]
  elif modo == ModoTreino.DEFENDER:
    return [ACAO_ESPERAR, ACAO_BLOQUEIO]
  elif modo == ModoTreino.DESTREZA:
    return [ACAO_ESPERAR, ACAO_ESQUIVA]
  elif modo == ModoTreino.COMBO:
    return [ACAO_ESPERAR, ACAO_ESQUIVA, ACAO_BLOQUEIO, ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO]
  else:
    return list(ACOES_IA)
