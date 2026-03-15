from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_ATAQUE_LEVE, ACAO_DEFENDER, ACAO_DESTREZA
from src.ai.recompensa import (
  PESO_APARAR_PERFEITO,
  PESO_DANO_INIMIGO_POR_PCT,
  PESO_DESTREZA_PERFEITA,
  PESO_KO,
  PESO_NAO_REAGIU_ATAQUE,
  PESO_SOBREVIVEU_JANELA_PERIGOSA,
  PESO_TENTOU_REAGIR_ATAQUE,
  PESO_TOMOU_DANO,
  PESO_ATAQUE_EM_PERIGO,
  PESO_VITORIA,
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
    esperado = PESO_TOMOU_DANO + PESO_NAO_REAGIU_ATAQUE
    self.assertEqual(recompensa, esperado)
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
    esperado = PESO_DESTREZA_PERFEITA + PESO_TENTOU_REAGIR_ATAQUE + PESO_SOBREVIVEU_JANELA_PERIGOSA
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
    self.assertTrue(info["destreza_perfeita"])

  @patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=True)
  def test_aparar_perfeito_pontua_positivo(self, _mock_aparar):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DEFENDER,
      inimigo_atacando=True,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    esperado = PESO_APARAR_PERFEITO + PESO_TENTOU_REAGIR_ATAQUE + PESO_SOBREVIVEU_JANELA_PERIGOSA
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
    self.assertTrue(info["aparar_perfeito"])

  def test_dano_inimigo_gera_recompensa(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      inimigo_atacando=False,
      vida_inimigo_atual=82.0,
      vida_inimigo_anterior=90.0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, 8.0 * PESO_DANO_INIMIGO_POR_PCT, places=6)
    self.assertTrue(info["causou_dano"])

  def test_ataque_em_perigo_penaliza(self):
    recompensa, terminal, _info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      inimigo_atacando=True,
      vida_jogador_atual=80.0,
      vida_jogador_anterior=80.0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    esperado = PESO_ATAQUE_EM_PERIGO + PESO_NAO_REAGIU_ATAQUE + PESO_SOBREVIVEU_JANELA_PERIGOSA
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
