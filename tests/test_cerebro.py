from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENSIVO
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
    acao = ACAO_COMBO_SEGURO

    q_estado = cerebro._obter_q_estado(estado)
    q_estado[str(acao)] = 0.0

    q_proximo = cerebro._obter_q_estado(proximo_estado)
    q_proximo["0"] = 2.0
    q_proximo["1"] = 1.0
    q_proximo["2"] = 0.5
    q_proximo["3"] = 0.1

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

  def test_aprender_acao_forcada_fora_da_lista_nao_quebra(self):
    cerebro = _novo_cerebro()
    estado = "forcado"

    cerebro.aprender(estado, ACAO_DEFENSIVO, recompensa=1.0, proximo_estado=estado, terminal=False)
    q_estado = cerebro._obter_q_estado(estado)

    self.assertIn(str(ACAO_DEFENSIVO), q_estado)
    self.assertGreater(q_estado[str(ACAO_DEFENSIVO)], 0.0)

  def test_estado_inclui_postura_tatica(self):
    cerebro = _novo_cerebro()

    estado_def = cerebro.obter_estado(
      info_vida={"vida_jogador_pct": 80.0, "vida_inimigo_pct": 70.0},
      info_personagens={"distancia_norm": 0.2, "fonte": "yolo"},
      sinais={
        "eu_tenho_especial": False,
        "adversario_com_especial": True,
        "nivel_especial_jogador": 0,
        "nivel_especial_inimigo": 2,
        "postura_tatica": "defensivo",
        "eu_atacando": False,
        "inimigo_atacando": True,
        "sem_movimento": False,
      },
    )
    estado_agr = cerebro.obter_estado(
      info_vida={"vida_jogador_pct": 80.0, "vida_inimigo_pct": 70.0},
      info_personagens={"distancia_norm": 0.2, "fonte": "yolo"},
      sinais={
        "eu_tenho_especial": False,
        "adversario_com_especial": False,
        "nivel_especial_jogador": 0,
        "nivel_especial_inimigo": 0,
        "postura_tatica": "agressivo",
        "eu_atacando": True,
        "inimigo_atacando": False,
        "sem_movimento": False,
      },
    )

    self.assertIn("postura=defensivo", estado_def)
    self.assertIn("postura=agressivo", estado_agr)

  def test_epsilon_respeita_minimo(self):
    cerebro = _novo_cerebro()
    cerebro.epsilon = 0.051
    cerebro.decaimento_epsilon = 0.5
    cerebro.epsilon_minimo = 0.05

    cerebro.aprender("s", 0, recompensa=0.0, proximo_estado="s2", terminal=False)
    self.assertGreaterEqual(cerebro.epsilon, cerebro.epsilon_minimo)
    self.assertAlmostEqual(cerebro.epsilon, 0.05, places=8)
