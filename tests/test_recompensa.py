from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DESTREZA
from src.ai.recompensa import (
  PESO_KO,
  PESO_VITORIA,
  PESO_DANO_INIMIGO_POR_PONTO,
  PESO_DESTREZA_SEM_PERFEICAO,
  obter_recompensa,
)


class TestRecompensa(TestCase):

  def test_ko_aplica_penalidade_terminal(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      nocaute_detectado=True,
      vitoria_detectada=False,
    )

    self.assertTrue(terminal)
    self.assertEqual(recompensa, PESO_KO)
    self.assertTrue(info["nocaute"])

  def test_vitoria_aplica_bonus_terminal(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      nocaute_detectado=False,
      vitoria_detectada=True,
    )

    self.assertTrue(terminal)
    self.assertEqual(recompensa, PESO_VITORIA)
    self.assertTrue(info["vitoria"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_combo_nao_penaliza_destreza(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_COMBO_SEGURO,
      adversario_com_especial=False,
      nocaute_detectado=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, 0.0)
    self.assertTrue(info["acao_destreza"])
    self.assertTrue(info["destreza_sem_penalidade"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_destreza_sem_especial_mantem_penalidade(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      adversario_com_especial=False,
      nocaute_detectado=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_DESTREZA_SEM_PERFEICAO)
    self.assertTrue(info["acao_destreza"])
    self.assertFalse(info["destreza_sem_penalidade"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_destreza_com_especial_inimigo_nao_penaliza(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      adversario_com_especial=True,
      nocaute_detectado=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, 0.0)
    self.assertTrue(info["acao_destreza"])
    self.assertTrue(info["destreza_sem_penalidade"])

  def test_dano_inimigo_e_proporcional_ao_delta_de_vida(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      adversario_com_especial=False,
      vida_inimigo_atual=65.0,
      vida_inimigo_anterior=70.0,
      colunas_escuras_inimigo_finais=1,
      nocaute_detectado=False,
    )

    esperado = 5.0 * PESO_DANO_INIMIGO_POR_PONTO
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
    self.assertTrue(info["causou_dano"])
    self.assertAlmostEqual(info["delta_dano_inimigo"], 5.0, places=6)
