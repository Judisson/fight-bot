from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.bot import visao


class TestVisao(TestCase):

  def setUp(self):
    visao._CACHE_TEMPLATES.clear()

  @patch("src.bot.visao.cv2.imdecode")
  @patch("src.bot.visao.np.fromfile")
  @patch("src.bot.visao.cv2.imread", return_value=None)
  def test_carregar_template_fallback_para_unicode_path(
    self,
    _mock_imread,
    mock_fromfile,
    mock_imdecode,
  ):
    fake_template = np.ones((10, 10, 3), dtype=np.uint8)
    mock_fromfile.return_value = np.array([1, 2, 3, 4], dtype=np.uint8)
    mock_imdecode.return_value = fake_template

    resultado = visao.carregar_template("assets/vitória.png")
    self.assertIsNotNone(resultado)
    self.assertEqual(resultado.shape, (10, 10, 3))
