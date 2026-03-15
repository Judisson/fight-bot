from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.ai.cerebro import CerebroIA


def _novo_cerebro():
  with patch("src.ai.cerebro.atexit.register", lambda *_args, **_kwargs: None):
    with patch("src.ai.cerebro.CAMINHO_MODELO_IA", Path("data/modelo_pixel_ppo_test.pt")):
      cerebro = CerebroIA(modo_treino="treino")
      cerebro.rollout_size = 2
      cerebro.update_epochs = 1
      cerebro.minibatch_size = 2
      return cerebro


class TestCerebroIA(TestCase):

  def setUp(self):
    self.frame = np.zeros((360, 640, 3), dtype=np.uint8)
    self.frame[100:200, 200:260] = 255

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

    frame_b = np.full((360, 640, 3), 180, dtype=np.uint8)
    estado_b = cerebro.obter_estado(frame=frame_b)

    self.assertNotAlmostEqual(float(estado_a.mean()), float(estado_b.mean()), places=3)
