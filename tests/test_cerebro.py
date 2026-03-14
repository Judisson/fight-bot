from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_DEFENDER
from src.ai.cerebro import CerebroIA


def _novo_cerebro():
  with patch("src.ai.cerebro.atexit.register", lambda *_args, **_kwargs: None):
    with patch("src.ai.cerebro.CAMINHO_MEMORIA_IA", Path("data/memoria_ia_test_tmp.json")):
      return CerebroIA()


class TestCerebroIA(TestCase):

  def test_q_update_usa_proximo_estado(self):
    cerebro = _novo_cerebro()
    estado = "s0"
    proximo_estado = "s1"
    acao = ACAO_DEFENDER

    q_estado = cerebro._obter_q_estado(estado)
    q_estado[str(acao)] = 0.0

    q_proximo = cerebro._obter_q_estado(proximo_estado)
    q_proximo["0"] = 2.0
    q_proximo["1"] = 1.0
    q_proximo["2"] = 0.5

    cerebro.aprender(estado, acao, recompensa=1.0, proximo_estado=proximo_estado, terminal=False)
    esperado = 0.0 + cerebro.alpha * ((1.0 + (cerebro.gamma * 2.0)) - 0.0)

    self.assertAlmostEqual(cerebro._obter_q_estado(estado)[str(acao)], esperado, places=6)

  def test_q_update_terminal_ignora_futuro(self):
    cerebro = _novo_cerebro()
    estado = "terminal"
    proximo_estado = "nao_importa"

    q_estado = cerebro._obter_q_estado(estado)
    q_estado["1"] = 1.0

    q_proximo = cerebro._obter_q_estado(proximo_estado)
    q_proximo["0"] = 999.0

    cerebro.aprender(estado, 1, recompensa=-5.0, proximo_estado=proximo_estado, terminal=True)
    esperado = 1.0 + cerebro.alpha * ((-5.0 + (cerebro.gamma * 0.0)) - 1.0)

    self.assertAlmostEqual(cerebro._obter_q_estado(estado)["1"], esperado, places=6)

  def test_estado_reduzido_da_fase_1(self):
    cerebro = _novo_cerebro()

    estado = cerebro.obter_estado(
      info_personagens={"distancia_norm": 0.2},
      sinais={
        "inimigo_atacando": True,
        "tomou_dano_recente": True,
      },
    )

    self.assertEqual(estado, "dist=media|atk_adv=1|dano_rec=1")

  def test_epsilon_respeita_minimo(self):
    cerebro = _novo_cerebro()
    cerebro.epsilon = 0.051
    cerebro.decaimento_epsilon = 0.5
    cerebro.epsilon_minimo = 0.05

    cerebro.aprender("s", 0, recompensa=0.0, proximo_estado="s2", terminal=False)
    self.assertGreaterEqual(cerebro.epsilon, cerebro.epsilon_minimo)
    self.assertAlmostEqual(cerebro.epsilon, 0.05, places=8)
