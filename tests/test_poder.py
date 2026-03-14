from unittest import TestCase

import numpy as np

from src.bot.poder import _detectar_nivel_especial


class TestPoder(TestCase):

  def test_segmentos_vazios_nao_devem_marcar_especial(self):
    roi = np.zeros((24, 180, 3), dtype=np.uint8)
    # Cor "vazia" pouco saturada, com um contorno colorido para simular borda.
    roi[:, :] = (90, 90, 90)
    roi[0:2, :] = (220, 90, 180)

    nivel, segmentos, _pct = _detectar_nivel_especial(roi, [0, 1, 2])
    self.assertEqual(nivel, 0)
    self.assertEqual(len(segmentos), 3)
    self.assertTrue(all((v is not None and v < 30.0) for v in segmentos))

  def test_primeiro_segmento_preenchido_marca_e1(self):
    roi = np.zeros((24, 180, 3), dtype=np.uint8)
    seg = roi.shape[1] // 3

    # Inimigo: E1 com preenchimento amarelo/laranja.
    roi[:, 0:seg] = (0, 180, 230)
    roi[:, seg:] = (90, 90, 90)

    nivel, segmentos, _pct = _detectar_nivel_especial(roi, [0, 1, 2])
    self.assertEqual(nivel, 1)
    self.assertGreater(segmentos[0], 55.0)
    self.assertLess(segmentos[1], 30.0)
