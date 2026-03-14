from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_DESTREZA
from src.ai.recompensa import (
  PESO_DESTREZA_PERFEITA,
  PESO_DESTREZA_SEM_PERIGO,
  PESO_KO,
  PESO_SOBREVIVEU_JANELA_PERIGOSA,
  PESO_TOMOU_DANO,
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

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_tomou_dano_aplica_penalidade(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      inimigo_atacando=True,
      vida_jogador_atual=70.0,
      vida_jogador_anterior=75.0,
      colunas_escuras_jogador_finais=1,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_TOMOU_DANO)
    self.assertTrue(info["tomou_dano"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=True)
  def test_destreza_perfeita_pontua_positivo(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      inimigo_atacando=True,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    esperado = PESO_DESTREZA_PERFEITA + PESO_SOBREVIVEU_JANELA_PERIGOSA
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
    self.assertTrue(info["destreza_perfeita"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_destreza_sem_perigo_penaliza(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      inimigo_atacando=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_DESTREZA_SEM_PERIGO)
    self.assertTrue(info["acao_destreza"])

  def test_sobreviveu_janela_perigosa_ganha_bonus(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      inimigo_atacando=True,
      vida_jogador_atual=80.0,
      vida_jogador_anterior=80.0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_SOBREVIVEU_JANELA_PERIGOSA)
    self.assertTrue(info["sobreviveu_perigo"])
