from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_COMBO_SEGURO, ACAO_DEFENDER, ACAO_DESTREZA
from src.ai.modo_treino import ModoTreino
from src.ai.recompensa import (
  PESO_BLOQUEIO_PERFEITO,
  PESO_BLOQUEIO_QUASE,
  PESO_BLOQUEIO_SEM_PERIGO,
  PESO_DESTREZA_QUASE,
  PESO_DESTREZA_PERFEITA,
  PESO_DESTREZA_SEM_PERIGO,
  PESO_DESTREZA_LONGE,
  PESO_KO,
  PESO_NAO_REAGIU_ATAQUE,
  PESO_SPAM_BLOQUEIO,
  PESO_SPAM_DESTREZA,
  PESO_SOBREVIVEU_JANELA_PERIGOSA,
  PESO_TENTOU_REAGIR_ATAQUE,
  PESO_TOMOU_DANO,
  PESO_PARADO_MUITO_TEMPO,
  PESO_APARAR_PERFEITO,
  PESO_COMBO_APOS_DESTREZA,
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

    esperado = (
      PESO_DESTREZA_PERFEITA
      + PESO_TENTOU_REAGIR_ATAQUE
      + PESO_SOBREVIVEU_JANELA_PERIGOSA
    )
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

  @patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=True)
  def test_aparar_perfeito_pontua_positivo(self, _mock_aparar):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DEFENDER,
      inimigo_atacando=True,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    esperado = (
      PESO_BLOQUEIO_PERFEITO
      + PESO_TENTOU_REAGIR_ATAQUE
      + PESO_SOBREVIVEU_JANELA_PERIGOSA
    )
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
    self.assertTrue(info["aparar_perfeito"])

  @patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=False)
  def test_bloqueio_quase_pontua_positivo(self, _mock_aparar):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DEFENDER,
      inimigo_atacando=True,
      vida_jogador_atual=80.0,
      vida_jogador_anterior=80.0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    esperado = (
      PESO_BLOQUEIO_QUASE
      + PESO_TENTOU_REAGIR_ATAQUE
      + PESO_SOBREVIVEU_JANELA_PERIGOSA
    )
    self.assertFalse(terminal)
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["bloqueio_quase"])

  @patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=False)
  def test_bloqueio_sem_perigo_penaliza(self, _mock_aparar):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DEFENDER,
      inimigo_atacando=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_BLOQUEIO_SEM_PERIGO)
    self.assertTrue(info["acao_bloqueio"])

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
    esperado = PESO_SOBREVIVEU_JANELA_PERIGOSA + PESO_NAO_REAGIU_ATAQUE
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["sobreviveu_perigo"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_destreza_quase_pontua_positivo(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      inimigo_atacando=True,
      vida_jogador_atual=80.0,
      vida_jogador_anterior=80.0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )

    esperado = (
      PESO_DESTREZA_QUASE
      + PESO_TENTOU_REAGIR_ATAQUE
      + PESO_SOBREVIVEU_JANELA_PERIGOSA
    )
    self.assertFalse(terminal)
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["destreza_quase"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_spam_destreza_aplica_penalidade(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      inimigo_atacando=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
      destreza_consecutiva=True,
    )

    esperado = PESO_DESTREZA_SEM_PERIGO + PESO_SPAM_DESTREZA
    self.assertFalse(terminal)
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["spam_destreza"])

  @patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=False)
  def test_spam_bloqueio_aplica_penalidade(self, _mock_aparar):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DEFENDER,
      inimigo_atacando=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
      bloqueio_consecutivo=True,
    )

    esperado = PESO_BLOQUEIO_SEM_PERIGO + PESO_SPAM_BLOQUEIO
    self.assertFalse(terminal)
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["spam_bloqueio"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=True)
  def test_destreza_longe_300px_e_ruim(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DESTREZA,
      inimigo_atacando=True,
      vida_jogador_atual=80.0,
      vida_jogador_anterior=80.0,
      nocaute_detectado=False,
      vitoria_detectada=False,
      distancia_px=301,
    )

    esperado = PESO_DESTREZA_LONGE + PESO_NAO_REAGIU_ATAQUE + PESO_SOBREVIVEU_JANELA_PERIGOSA
    self.assertFalse(terminal)
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["destreza_longe_ruim"])
    self.assertFalse(info["destreza_perfeita"])

  @patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=True)
  def test_modo_bloqueio_aparar_perfeito_usa_bonus_fase2(self, _mock_aparar):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_DEFENDER,
      inimigo_atacando=True,
      nocaute_detectado=False,
      vitoria_detectada=False,
      modo_treino=ModoTreino.BLOQUEIO,
    )

    esperado = PESO_APARAR_PERFEITO + PESO_TENTOU_REAGIR_ATAQUE + PESO_SOBREVIVEU_JANELA_PERIGOSA
    self.assertFalse(terminal)
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["aparar_perfeito"])

  def test_modo_destreza_penaliza_parado_muito_tempo(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      inimigo_atacando=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
      modo_treino=ModoTreino.DESTREZA,
      tempo_parado_frames=999,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_PARADO_MUITO_TEMPO)
    self.assertTrue(info["parado_muito_tempo"])

  def test_modo_contra_ataque_bonus_combo_pos_destreza(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_COMBO_SEGURO,
      inimigo_atacando=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
      modo_treino=ModoTreino.CONTRA_ATAQUE,
      combo_apos_destreza=True,
    )

    self.assertFalse(terminal)
    self.assertEqual(recompensa, PESO_COMBO_APOS_DESTREZA)
    self.assertTrue(info["combo_apos_destreza"])
