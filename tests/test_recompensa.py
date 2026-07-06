from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_ATAQUE_LEVE, ACAO_ATAQUE_PESADO, ACAO_BLOQUEIO, ACAO_ESQUIVA
from src.ai.recompensa import (
  PESO_ATAQUE_PESADO_TOMOU_DANO,
  PESO_APARAR_PERFEITO,
  PESO_DANO_TOMADO_POR_PCT,
  PESO_DANO_INIMIGO_POR_PCT,
  PESO_DESTREZA_PERFEITA,
  PESO_KO,
  PESO_NAO_REAGIU_ATAQUE,
  PESO_SOBREVIVEU_JANELA_PERIGOSA,
  PESO_TENTOU_REAGIR_ATAQUE,
  PESO_ATAQUE_EM_PERIGO,
  PESO_VITORIA,
  PENALIDADE_SEM_COMBO,
  RECOMPENSA_COMBO_ATIVO,
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
    esperado = (5.0 * PESO_DANO_TOMADO_POR_PCT) + PESO_NAO_REAGIU_ATAQUE
    self.assertEqual(recompensa, esperado)
    self.assertTrue(info["tomou_dano"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False)
  def test_nao_reagiu_so_pune_quando_habilitado(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=0,
      inimigo_atacando=True,
      aplicar_penalidade_nao_reagiu=False,
      vida_jogador_atual=70.0,
      vida_jogador_anterior=75.0,
      colunas_escuras_jogador_finais=1,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertFalse(terminal)
    esperado = 5.0 * PESO_DANO_TOMADO_POR_PCT
    self.assertEqual(recompensa, esperado)
    self.assertFalse(info["nao_reagiu_ataque"])

  @patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=True)
  def test_destreza_perfeita_pontua_positivo(self, _mock_destreza):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ESQUIVA,
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
      acao_atual=ACAO_BLOQUEIO,
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

  @patch("src.ai.recompensa._detectar_combo_ativo", return_value=True)
  def test_combo_ativo_recompensa_por_tempo(self, _mock_combo):
    recompensa, terminal, info = obter_recompensa(
      frame=object(),
      acao_atual=0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, RECOMPENSA_COMBO_ATIVO, places=6)
    self.assertTrue(info["combo_ativo"])

  @patch("src.ai.recompensa._detectar_combo_ativo", return_value=False)
  def test_sem_combo_penaliza_por_tempo(self, _mock_combo):
    recompensa, terminal, info = obter_recompensa(
      frame=object(),
      acao_atual=0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, PENALIDADE_SEM_COMBO, places=6)
    self.assertFalse(info["combo_ativo"])

  @patch("src.ai.recompensa._detectar_combo_ativo", return_value=True)
  def test_combo_nao_pontua_quando_desabilitado_no_frame(self, _mock_combo):
    recompensa, terminal, info = obter_recompensa(
      frame=object(),
      acao_atual=0,
      aplicar_pontuacao_combo=False,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, 0.0, places=6)
    self.assertTrue(info["combo_ativo"])

  def test_pesado_tomou_dano_aplica_penalidade_extra(self):
    recompensa, terminal, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_PESADO,
      inimigo_atacando=True,
      vida_jogador_atual=70.0,
      vida_jogador_anterior=75.0,
      colunas_escuras_jogador_finais=1,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    esperado = (
      (5.0 * PESO_DANO_TOMADO_POR_PCT)
      + PESO_ATAQUE_PESADO_TOMOU_DANO
      + PESO_ATAQUE_EM_PERIGO
      + PESO_NAO_REAGIU_ATAQUE
    )
    self.assertFalse(terminal)
    self.assertAlmostEqual(recompensa, esperado, places=6)
    self.assertTrue(info["ataque_pesado_tomou_dano"])

  def test_punicao_agressividade_especial_1(self):
    recompensa_com_e1, _, info_com_e1 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      nivel_especial_inimigo=1,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    recompensa_sem_e1, _, info_sem_e1 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      nivel_especial_inimigo=None,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    from src.ai.recompensa import PESO_PUNICAO_AGRESSIVIDADE_E1
    self.assertAlmostEqual(recompensa_com_e1 - recompensa_sem_e1, PESO_PUNICAO_AGRESSIVIDADE_E1, places=6)
    self.assertTrue(info_com_e1["punicao_agressividade_especial"])
    self.assertFalse(info_sem_e1["punicao_agressividade_especial"])

  def test_punicao_agressividade_especial_2(self):
    recompensa_com_e2, _, info_com_e2 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      nivel_especial_inimigo=2,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    recompensa_sem_e2, _, info_sem_e2 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      nivel_especial_inimigo=None,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    from src.ai.recompensa import PESO_PUNICAO_AGRESSIVIDADE_E2
    self.assertAlmostEqual(recompensa_com_e2 - recompensa_sem_e2, PESO_PUNICAO_AGRESSIVIDADE_E2, places=6)
    self.assertTrue(info_com_e2["punicao_agressividade_especial"])
    self.assertFalse(info_sem_e2["punicao_agressividade_especial"])

  def test_punicao_defesa_sem_especial(self):
    recompensa_com_e0, _, info_com_e0 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ESQUIVA,
      nivel_especial_inimigo=0,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    recompensa_sem_e0, _, info_sem_e0 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ESQUIVA,
      nivel_especial_inimigo=None,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    from src.ai.recompensa import PESO_PUNICAO_DEFESA_SEM_ESPECIAL
    self.assertAlmostEqual(recompensa_com_e0 - recompensa_sem_e0, PESO_PUNICAO_DEFESA_SEM_ESPECIAL, places=6)
    self.assertTrue(info_com_e0["punicao_defesa_sem_especial"])
    self.assertFalse(info_sem_e0["punicao_defesa_sem_especial"])

  def test_punicao_oponente_e3(self):
    recompensa_com_e3, _, info_com_e3 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ESQUIVA,
      nivel_especial_inimigo=3,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    recompensa_sem_e3, _, info_sem_e3 = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ESQUIVA,
      nivel_especial_inimigo=None,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    from src.ai.recompensa import PESO_PUNICAO_OPONENTE_E3
    self.assertAlmostEqual(recompensa_com_e3 - recompensa_sem_e3, PESO_PUNICAO_OPONENTE_E3, places=6)
    self.assertTrue(info_com_e3["punicao_oponente_e3"])
    self.assertFalse(info_sem_e3["punicao_oponente_e3"])

  def test_recompensa_oponente_especial_recente(self):
    # Teste 1: ACAO_BLOQUEIO sem aparo perfeito -> Recompensa +2.0
    with patch("src.ai.recompensa._detectar_aparar_perfeito", return_value=False):
      recompensa, _, info = obter_recompensa(
        frame=None,
        acao_atual=ACAO_BLOQUEIO,
        oponente_especial_recente=True,
        nocaute_detectado=False,
        vitoria_detectada=False,
      )
      self.assertEqual(recompensa, 2.0)
      self.assertTrue(info["gratificacao_defesa_especial_recente"])

    # Teste 2: ACAO_ESQUIVA sem destreza perfeita -> Recompensa 0.0 (neutra)
    with patch("src.ai.recompensa._detectar_destreza_perfeita", return_value=False):
      recompensa, _, info = obter_recompensa(
        frame=None,
        acao_atual=ACAO_ESQUIVA,
        oponente_especial_recente=True,
        nocaute_detectado=False,
        vitoria_detectada=False,
      )
      self.assertEqual(recompensa, 0.0)
      self.assertTrue(info["destreza_neutra_especial_recente"])

    # Teste 3: Acao ofensiva (ataque) -> Recompensa -5.0
    recompensa, _, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ATAQUE_LEVE,
      oponente_especial_recente=True,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertEqual(recompensa, -5.0)
    self.assertTrue(info["punicao_ofensiva_especial_recente"])

    # Teste 4: Acao de esperar -> Recompensa -2.0
    from src.ai.acoes import ACAO_ESPERAR
    recompensa, _, info = obter_recompensa(
      frame=None,
      acao_atual=ACAO_ESPERAR,
      oponente_especial_recente=True,
      nocaute_detectado=False,
      vitoria_detectada=False,
    )
    self.assertEqual(recompensa, -2.0)
    self.assertTrue(info["punicao_espera_especial_recente"])
