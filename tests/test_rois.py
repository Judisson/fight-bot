from unittest import TestCase
from unittest.mock import patch

from src.bot.rois import (
  ROI_COMBO_SEM_PERDA,
  ROI_JOGAR_NOVAMENTE,
  ROI_SEQUENCIA_GOLPES,
  obter_env_roi,
  obter_roi,
  obter_template_roi,
)


class TestRois(TestCase):

  def test_roi_padrao_jogar_novamente(self):
    with patch.dict("os.environ", {}, clear=True):
      roi = obter_roi(ROI_JOGAR_NOVAMENTE)
    self.assertEqual(roi, (960, 950, 1215, 1040))

  def test_roi_override_por_env(self):
    env = {obter_env_roi(ROI_JOGAR_NOVAMENTE): "100,200,300,400"}
    with patch.dict("os.environ", env, clear=True):
      roi = obter_roi(ROI_JOGAR_NOVAMENTE)
    self.assertEqual(roi, (100, 200, 300, 400))

  def test_roi_invalido_por_env_faz_fallback(self):
    env = {obter_env_roi(ROI_JOGAR_NOVAMENTE): "300,200,100,50"}
    with patch.dict("os.environ", env, clear=True):
      roi = obter_roi(ROI_JOGAR_NOVAMENTE)
    self.assertEqual(roi, (960, 950, 1215, 1040))

  def test_roi_padrao_combo_sem_perda(self):
    with patch.dict("os.environ", {}, clear=True):
      roi = obter_roi(ROI_COMBO_SEM_PERDA)
    self.assertEqual(roi, (260, 340, 510, 430))

  def test_alias_combo_aponta_para_sequencia(self):
    self.assertEqual(ROI_COMBO_SEM_PERDA, ROI_SEQUENCIA_GOLPES)

  def test_template_sequencia_golpes_configurado(self):
    self.assertEqual(obter_template_roi(ROI_SEQUENCIA_GOLPES), "assets/sequencia-golpes.png")
