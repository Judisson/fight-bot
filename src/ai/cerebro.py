import atexit
import json
import random
from pathlib import Path

CAMINHO_MEMORIA_IA = Path("data/memoria_ia.json")


class CerebroIA:

  def __init__(self):

    self.epsilon = 1.0
    self.decaimento_epsilon = 0.995
    self.epsilon_minimo = 0.05

    self.tabela_q = {}

    self._carregar_memoria()
    atexit.register(self.salvar_memoria)

  def _carregar_memoria(self):
    if not CAMINHO_MEMORIA_IA.exists():
      return

    try:
      dados = json.loads(CAMINHO_MEMORIA_IA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      print("Memoria IA invalida. Iniciando memoria nova.")
      return

    tabela_q = dados.get("tabela_q", {})
    if isinstance(tabela_q, dict):
      self.tabela_q = {str(k): int(v) for k, v in tabela_q.items()}

    epsilon = dados.get("epsilon")
    if isinstance(epsilon, (int, float)):
      self.epsilon = float(epsilon)

    print(
      f"Memoria IA carregada: {len(self.tabela_q)} estados | epsilon={self.epsilon:.4f}"
    )

  def salvar_memoria(self):
    dados = {
      "tabela_q": self.tabela_q,
      "epsilon": self.epsilon,
      "decaimento_epsilon": self.decaimento_epsilon,
      "epsilon_minimo": self.epsilon_minimo,
    }

    try:
      CAMINHO_MEMORIA_IA.parent.mkdir(parents=True, exist_ok=True)
      CAMINHO_MEMORIA_IA.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
      )
      print(f"Memoria IA salva em: {CAMINHO_MEMORIA_IA}")
    except OSError:
      print("Falha ao salvar memoria da IA.")

  def obter_estado(self):

    # estado simples
    return "luta"

  def escolher_acao(self, estado):

    if random.random() < self.epsilon:
      return random.randint(0, 1)

    return self.tabela_q.get(estado, 0)

  def aprender(self, estado, acao, recompensa):

    self.tabela_q[estado] = acao

    if self.epsilon > self.epsilon_minimo:
      self.epsilon *= self.decaimento_epsilon
