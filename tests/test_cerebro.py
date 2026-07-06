from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.ai.cerebro import CerebroIA


def _novo_cerebro(modo_treino="treino"):
  with patch("src.ai.cerebro.atexit.register", lambda *_args, **_kwargs: None):
    with patch("src.ai.cerebro.CAMINHO_MODELO_IA", Path("data/modelo_pixel_ppo_test.pt")):
      from src.ai.modo_treino import obter_acoes_permitidas
      acoes = obter_acoes_permitidas(modo_treino)
      cerebro = CerebroIA(acoes_permitidas=acoes, modo_treino=modo_treino)
      cerebro.rollout_size = 2
      cerebro.update_epochs = 1
      cerebro.minibatch_size = 2
      return cerebro


class TestCerebroIA(TestCase):

  def setUp(self):
    self.frame = np.zeros((360, 640, 3), dtype=np.uint8)
    self.frame[100:200, 200:260] = [0, 0, 255]  # Vermelho BGR quente para ativar a mascara termica

  def test_obter_estado_pixel_stack(self):
    cerebro = _novo_cerebro()
    estado = cerebro.obter_estado(frame=self.frame)

    self.assertEqual(estado.shape, (4, 128, 128))
    self.assertGreaterEqual(float(estado.min()), 0.0)
    self.assertLessEqual(float(estado.max()), 1.0)

  def test_escolher_acao_pertence_espaco(self):
    cerebro = _novo_cerebro()
    estado = cerebro.obter_estado(frame=self.frame)
    acao = cerebro.escolher_acao(estado)

    self.assertIn(acao, cerebro.acoes)
    debug = cerebro.obter_debug_rede()
    self.assertEqual(debug["acao"], acao)
    self.assertAlmostEqual(sum(debug["probs"].values()), 1.0, places=4)

  def test_aprender_dispara_update_ppo(self):
    cerebro = _novo_cerebro()
    estado0 = cerebro.obter_estado(frame=self.frame)
    acao0 = cerebro.escolher_acao(estado0)

    frame2 = self.frame.copy()
    frame2[:, :] = 30
    estado1 = cerebro.obter_estado(frame=frame2)
    cerebro.aprender(estado0, acao0, recompensa=1.5, proximo_estado=estado1, terminal=False)

    acao1 = cerebro.escolher_acao(estado1)
    cerebro.aprender(estado1, acao1, recompensa=-0.5, proximo_estado=None, terminal=True)

    self.assertGreaterEqual(cerebro.obter_debug_rede()["updates"], 1)

  def test_reset_observacao_limpa_stack(self):
    cerebro = _novo_cerebro()
    estado_a = cerebro.obter_estado(frame=self.frame)
    cerebro.reset_observacao()

    frame_b = np.zeros((360, 640, 3), dtype=np.uint8)  # Diferente do anterior (completamente escuro, sem tons quentes)
    estado_b = cerebro.obter_estado(frame=frame_b)

    self.assertNotAlmostEqual(float(estado_a.mean()), float(estado_b.mean()), places=3)

  def test_escolher_acao_mascara_especial_se_indisponivel(self):
    cerebro = _novo_cerebro()
    estado = cerebro.obter_estado(frame=self.frame)

    acao = cerebro.escolher_acao(estado, especial_disponivel=False)
    debug = cerebro.obter_debug_rede()
    self.assertEqual(debug["probs"].get(6, 0.0), 0.0)
    self.assertNotEqual(acao, 6)

  def test_escolher_acao_nao_mascara_ataques_se_oponente_tem_especial(self):
    cerebro = _novo_cerebro()
    estado = cerebro.obter_estado(frame=self.frame)

    _acao = cerebro.escolher_acao(estado, oponente_tem_especial=True)
    debug = cerebro.obter_debug_rede()
    # Pelo menos uma das ações de ataque deve ter probabilidade maior que zero
    probs_ataque = [debug["probs"].get(a, 0.0) for a in [3, 4, 5, 6]]
    self.assertTrue(any(p > 0.0 for p in probs_ataque))

  @patch("src.ai.cerebro.os.getenv")
  def test_escolher_acao_epsilon_greedily_retains_collapsed_options(self, mock_getenv):
    # Simula BOT_EXPLORACAO_EPSILON = 0.12
    def side_effect(key, default=None):
      if key == "BOT_EXPLORACAO_EPSILON":
        return "0.12"
      return default
    mock_getenv.side_effect = side_effect

    cerebro = _novo_cerebro()
    estado = cerebro.obter_estado(frame=self.frame)

    # Executa escolha da ação
    _acao = cerebro.escolher_acao(estado, especial_disponivel=False, oponente_tem_especial=True)
    debug = cerebro.obter_debug_rede()

    # Como especial está indisponível, apenas 0, 1, 2, 3, 4, 5 são permitidos.
    # Com epsilon=0.12, cada uma das 6 permitidas deve ter probabilidade de pelo menos (0.12 / 6) = 0.02 (2%).
    # A ação 6 deve ser exatamente 0.0.
    for a in range(6):
      self.assertGreaterEqual(debug["probs"].get(a, 0.0), 0.019)
    self.assertEqual(debug["probs"].get(6, 0.0), 0.0)

  def test_escolher_acao_nao_mascara_ataques_se_treino_focado(self):
    cerebro = _novo_cerebro(modo_treino="combo")
    estado = cerebro.obter_estado(frame=self.frame)

    # Com oponente com especial, no modo de combo, ataques leves (3) e médios (4)
    # NÃO devem ser mascarados (probabilidade deve ser maior que 0).
    # Enquanto que 5 e 6 (pesado, especial) continuam bloqueados pelo modo combo.
    _acao = cerebro.escolher_acao(estado, oponente_tem_especial=True)
    debug = cerebro.obter_debug_rede()

    self.assertGreater(debug["probs"].get(3, 0.0), 0.0)
    self.assertGreater(debug["probs"].get(4, 0.0), 0.0)
    self.assertEqual(debug["probs"].get(5, 0.0), 0.0)
    self.assertEqual(debug["probs"].get(6, 0.0), 0.0)
