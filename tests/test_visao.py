from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from src.bot import visao


class TestVisao(TestCase):

  @patch("src.bot.visao.cv2.imdecode")
  @patch("src.bot.visao.np.fromfile")
  @patch("src.bot.visao.cv2.imread", return_value=None)
  def test_carregar_template_fallback_para_unicode_path(
    self,
    mock_imread,
    mock_fromfile,
    mock_imdecode,
  ):
    fake_template = np.ones((10, 10), dtype=np.uint8)
    mock_fromfile.return_value = np.array([1, 2, 3, 4], dtype=np.uint8)
    mock_imdecode.return_value = fake_template

    resultado = visao.carregar_template("assets/vitoria.png")
    self.assertIsNotNone(resultado)
    self.assertEqual(resultado.shape, (10, 10))
    mock_imread.assert_called_once_with("assets/vitoria.png", cv2.IMREAD_GRAYSCALE)
    mock_imdecode.assert_called_once()

  @patch("src.bot.visao.carregar_template")
  def test_encontrar_template_com_roi_ajusta_coordenadas_globais(self, mock_template):
    frame = np.zeros((120, 160), dtype=np.uint8)
    template = np.zeros((10, 10), dtype=np.uint8)
    template[:, :] = 20
    template[2:8, 3:7] = 220
    template[0:3, 0:3] = 255
    frame[40:50, 70:80] = template
    mock_template.return_value = template

    with patch.dict("os.environ", {"BOT_VISAO_MATCH_SCALE": "1.0"}, clear=False):
      resultado = visao.encontrar_template(
        frame,
        "assets/fake.png",
        limiar=0.8,
        roi=(60, 30, 120, 90),
      )

    self.assertIsNotNone(resultado)
    self.assertEqual(resultado["x"], 75)
    self.assertEqual(resultado["y"], 45)

  @patch("src.bot.visao.carregar_template")
  def test_encontrar_template_com_roi_fora_da_area_retorna_none(self, mock_template):
    frame = np.zeros((120, 160), dtype=np.uint8)
    template = np.zeros((10, 10), dtype=np.uint8)
    template[:, :] = 20
    template[2:8, 3:7] = 220
    template[0:3, 0:3] = 255
    frame[40:50, 70:80] = template
    mock_template.return_value = template

    with patch.dict("os.environ", {"BOT_VISAO_MATCH_SCALE": "1.0"}, clear=False):
      resultado = visao.encontrar_template(
        frame,
        "assets/fake.png",
        limiar=0.8,
        roi=(0, 0, 30, 30),
      )

    self.assertIsNone(resultado)

  @patch("src.bot.visao.carregar_template")
  def test_encontrar_template_com_escala_retorna_coord_no_frame_original(self, mock_template):
    frame = np.zeros((120, 160), dtype=np.uint8)
    template = np.zeros((10, 10), dtype=np.uint8)
    template[:, :] = 20
    template[2:8, 3:7] = 220
    template[0:3, 0:3] = 255
    frame[40:50, 70:80] = template
    mock_template.return_value = template

    with patch.dict("os.environ", {"BOT_VISAO_MATCH_SCALE": "0.5"}, clear=False):
      resultado = visao.encontrar_template(
        frame,
        "assets/fake.png",
        limiar=0.8,
      )

    self.assertIsNotNone(resultado)
    self.assertTrue(abs(resultado["x"] - 75) <= 2)
    self.assertTrue(abs(resultado["y"] - 45) <= 2)
