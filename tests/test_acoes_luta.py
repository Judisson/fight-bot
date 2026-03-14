import time
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_BLOQUEIO, ACAO_COMBO_SEGURO, ACAO_NADA
from src.bot.acoes_luta import combo_seguro, executar_acao, obter_acao_em_execucao


class TestAcoesLuta(TestCase):

  @patch("src.bot.acoes_luta._executar_acao_sincrona")
  def test_estado_acao_em_execucao_reflete_worker(self, mock_executar):
    iniciou = Event()
    finalizou = Event()

    def _fake_execucao(acao, pode_soltar_especial=False):
      _ = (acao, pode_soltar_especial)
      iniciou.set()
      time.sleep(0.08)
      finalizou.set()

    mock_executar.side_effect = _fake_execucao

    self.assertTrue(executar_acao(ACAO_COMBO_SEGURO))
    self.assertTrue(iniciou.wait(0.5))
    self.assertEqual(obter_acao_em_execucao(), ACAO_COMBO_SEGURO)
    self.assertTrue(finalizou.wait(0.5))

    limite = time.time() + 0.5
    while time.time() < limite and obter_acao_em_execucao() != ACAO_NADA:
      time.sleep(0.01)
    self.assertEqual(obter_acao_em_execucao(), ACAO_NADA)

  @patch("src.bot.acoes_luta._executar_acao_sincrona")
  def test_fila_descarta_quando_lotada(self, mock_executar):
    iniciou = Event()

    def _fake_execucao(acao, pode_soltar_especial=False):
      _ = (acao, pode_soltar_especial)
      iniciou.set()
      time.sleep(0.12)

    mock_executar.side_effect = _fake_execucao

    self.assertTrue(executar_acao(ACAO_COMBO_SEGURO))
    self.assertTrue(iniciou.wait(0.5))
    self.assertTrue(executar_acao(ACAO_BLOQUEIO))
    self.assertFalse(executar_acao(ACAO_BLOQUEIO))

  @patch("src.bot.acoes_luta.time.sleep", return_value=None)
  @patch("src.bot.acoes_luta.especial")
  @patch("src.bot.acoes_luta.destreza")
  @patch("src.bot.acoes_luta.ataque_leve")
  @patch("src.bot.acoes_luta.ataque_medio")
  def test_combo_finaliza_com_especial_se_tiver_barra(
    self,
    mock_medio,
    mock_leve,
    mock_destreza,
    mock_especial,
    _mock_sleep,
  ):
    combo_seguro(pode_soltar_especial=True)
    self.assertEqual(mock_medio.call_count, 1)
    self.assertEqual(mock_leve.call_count, 3)
    self.assertEqual(mock_especial.call_count, 1)
    self.assertEqual(mock_destreza.call_count, 0)

  @patch("src.bot.acoes_luta.time.sleep", return_value=None)
  @patch("src.bot.acoes_luta.especial")
  @patch("src.bot.acoes_luta.destreza")
  @patch("src.bot.acoes_luta.ataque_leve")
  @patch("src.bot.acoes_luta.ataque_medio")
  def test_combo_finaliza_com_destreza_sem_barra(
    self,
    mock_medio,
    mock_leve,
    mock_destreza,
    mock_especial,
    _mock_sleep,
  ):
    combo_seguro(pode_soltar_especial=False)
    self.assertEqual(mock_medio.call_count, 1)
    self.assertEqual(mock_leve.call_count, 3)
    self.assertEqual(mock_especial.call_count, 0)
    self.assertEqual(mock_destreza.call_count, 1)
