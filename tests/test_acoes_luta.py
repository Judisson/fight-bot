import time
from threading import Event
from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import ACAO_BLOQUEIO, ACAO_ESQUIVA, ACAO_ESPERAR
from src.bot.acoes_luta import executar_acao, obter_acao_em_execucao


class TestAcoesLuta(TestCase):

  @patch("src.bot.acoes_luta._executar_acao_sincrona")
  def test_estado_acao_em_execucao_reflete_worker(self, mock_executar):
    iniciou = Event()
    finalizou = Event()

    def _fake_execucao(acao):
      _ = acao
      iniciou.set()
      time.sleep(0.08)
      finalizou.set()

    mock_executar.side_effect = _fake_execucao

    self.assertTrue(executar_acao(ACAO_ESQUIVA))
    self.assertTrue(iniciou.wait(0.5))
    self.assertEqual(obter_acao_em_execucao(), ACAO_ESQUIVA)
    self.assertTrue(finalizou.wait(0.5))

    limite = time.time() + 0.5
    while time.time() < limite and obter_acao_em_execucao() != ACAO_ESPERAR:
      time.sleep(0.01)
    self.assertEqual(obter_acao_em_execucao(), ACAO_ESPERAR)

  @patch("src.bot.acoes_luta._executar_acao_sincrona")
  def test_fila_descarta_quando_lotada(self, mock_executar):
    iniciou = Event()

    def _fake_execucao(acao):
      _ = acao
      iniciou.set()
      time.sleep(0.12)

    mock_executar.side_effect = _fake_execucao

    self.assertTrue(executar_acao(ACAO_ESQUIVA))
    self.assertTrue(iniciou.wait(0.5))
    self.assertTrue(executar_acao(ACAO_BLOQUEIO))
    self.assertFalse(executar_acao(ACAO_BLOQUEIO))

  def test_esperar_nao_dispara_worker(self):
    self.assertFalse(executar_acao(ACAO_ESPERAR))
