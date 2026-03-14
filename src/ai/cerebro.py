import atexit
import json
import random
from pathlib import Path

from src.ai.acoes import ACOES_IA
from src.utils.log import log

CAMINHO_MEMORIA_IA = Path("data/memoria_ia.json")


class CerebroIA:

  def __init__(self):

    self.epsilon = 1.0
    self.decaimento_epsilon = 0.9995
    self.epsilon_minimo = 0.05

    self.alpha = 0.2
    self.gamma = 0.9

    self.acoes = list(ACOES_IA)
    self.tabela_q = {}

    self._carregar_memoria()
    atexit.register(self.salvar_memoria)

  def _carregar_memoria(self):
    if not CAMINHO_MEMORIA_IA.exists():
      return

    try:
      dados = json.loads(CAMINHO_MEMORIA_IA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      log("Memoria IA invalida. Iniciando memoria nova.")
      return

    tabela_q = dados.get("tabela_q", {})
    if isinstance(tabela_q, dict):
      self.tabela_q = {}
      for estado, valor in tabela_q.items():
        estado_key = str(estado)
        if isinstance(valor, dict):
          self.tabela_q[estado_key] = {
            str(acao): float(q_valor)
            for acao, q_valor in valor.items()
          }
        elif isinstance(valor, int):
          # Compatibilidade com memoria antiga (estado -> acao inteira).
          q_estado = self._inicializar_estado_q(estado_key)
          q_estado[str(valor)] = 1.0
        else:
          self.tabela_q[estado_key] = self._inicializar_estado_q(estado_key)

    epsilon = dados.get("epsilon")
    if isinstance(epsilon, (int, float)):
      self.epsilon = float(epsilon)

    alpha = dados.get("alpha")
    if isinstance(alpha, (int, float)):
      self.alpha = float(alpha)

    gamma = dados.get("gamma")
    if isinstance(gamma, (int, float)):
      self.gamma = float(gamma)

    log(
      f"Memoria IA carregada: {len(self.tabela_q)} estados | epsilon={self.epsilon:.4f}"
    )

  def _inicializar_estado_q(self, estado):
    q_estado = {str(acao): 0.0 for acao in self.acoes}
    self.tabela_q[estado] = q_estado
    return q_estado

  def _obter_q_estado(self, estado):
    estado = str(estado)
    if estado not in self.tabela_q:
      return self._inicializar_estado_q(estado)

    q_estado = self.tabela_q[estado]
    for acao in self.acoes:
      chave_acao = str(acao)
      if chave_acao not in q_estado:
        q_estado[chave_acao] = 0.0

    return q_estado

  def salvar_memoria(self):
    dados = {
      "tabela_q": self.tabela_q,
      "epsilon": self.epsilon,
      "decaimento_epsilon": self.decaimento_epsilon,
      "epsilon_minimo": self.epsilon_minimo,
      "alpha": self.alpha,
      "gamma": self.gamma,
    }

    try:
      CAMINHO_MEMORIA_IA.parent.mkdir(parents=True, exist_ok=True)
      CAMINHO_MEMORIA_IA.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
      )
      log(f"Memoria IA salva em: {CAMINHO_MEMORIA_IA}")
    except OSError:
      log("Falha ao salvar memoria da IA.")

  @staticmethod
  def _bucket_distancia(distancia_norm):
    if distancia_norm is None:
      return "x"
    valor = max(0.0, min(1.0, float(distancia_norm)))
    if valor < 0.18:
      return "curta"
    if valor < 0.35:
      return "media"
    return "longa"

  @staticmethod
  def _bucket_bool(valor):
    if valor is None:
      return "x"
    return "1" if bool(valor) else "0"

  def obter_estado(self, info_vida=None, info_personagens=None, sinais=None):
    info_personagens = info_personagens or {}
    sinais = sinais or {}

    distancia_norm = info_personagens.get("distancia_norm")
    inimigo_atacando = sinais.get("inimigo_atacando")
    tomou_dano_recente = sinais.get("tomou_dano_recente")

    # ETAPA 1 / FASE 1 (ATIVA): estado reduzido para melhorar revisitacao de
    # cenarios na Q-table e acelerar aprendizado.
    estado = (
      f"dist={self._bucket_distancia(distancia_norm)}",
      f"atk_adv={self._bucket_bool(inimigo_atacando)}",
      f"dano_rec={self._bucket_bool(tomou_dano_recente)}",
    )
    # FASE 2/3 (PENDENTE): reintroduzir sinais de oportunidade ofensiva
    # (combo/special) apos estabilizar sobrevivencia na Fase 1.
    return "|".join(estado)

  def escolher_acao(self, estado):

    if random.random() < self.epsilon:
      return random.choice(self.acoes)

    q_estado = self._obter_q_estado(estado)
    return max(self.acoes, key=lambda acao: q_estado[str(acao)])

  def aprender(self, estado, acao, recompensa, proximo_estado=None, terminal=False):

    q_estado = self._obter_q_estado(estado)
    chave_acao = str(acao)
    if chave_acao not in q_estado:
      q_estado[chave_acao] = 0.0

    q_atual = q_estado[chave_acao]
    if terminal or proximo_estado is None:
      melhor_q_futuro = 0.0
    else:
      q_proximo = self._obter_q_estado(proximo_estado)
      melhor_q_futuro = max(q_proximo.values())

    alvo = recompensa + (self.gamma * melhor_q_futuro)
    q_estado[chave_acao] = q_atual + self.alpha * (alvo - q_atual)

    if self.epsilon > self.epsilon_minimo:
      self.epsilon *= self.decaimento_epsilon
      self.epsilon = max(self.epsilon, self.epsilon_minimo)
