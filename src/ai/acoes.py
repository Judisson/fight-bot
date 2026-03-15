ACAO_ESPERAR = 0
ACAO_DESTREZA = 1
ACAO_DEFENDER = 2
ACAO_ATAQUE_LEVE = 3
ACAO_ATAQUE_MEDIO = 4

ACOES_IA = [
  ACAO_ESPERAR,
  ACAO_DESTREZA,
  ACAO_DEFENDER,
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
]

NOMES_ACAO = {
  ACAO_ESPERAR: "esperar",
  ACAO_DESTREZA: "destreza",
  ACAO_DEFENDER: "defender",
  ACAO_ATAQUE_LEVE: "ataque_leve",
  ACAO_ATAQUE_MEDIO: "ataque_medio",
}


def nome_acao(acao):
  return NOMES_ACAO.get(int(acao), f"acao_{acao}")
