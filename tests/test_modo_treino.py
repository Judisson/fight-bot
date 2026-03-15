from unittest import TestCase

from src.ai.acoes import ACAO_DEFENDER, ACAO_DESTREZA, ACAO_ESPERAR
from src.ai.modo_treino import ModoTreino, obter_acoes_permitidas, resolver_modo_treino


class TestModoTreino(TestCase):

  def test_acoes_permitidas_destreza(self):
    acoes = obter_acoes_permitidas(ModoTreino.DESTREZA)
    self.assertEqual(acoes, [ACAO_ESPERAR, ACAO_DESTREZA])

  def test_acoes_permitidas_bloqueio(self):
    acoes = obter_acoes_permitidas(ModoTreino.BLOQUEIO)
    self.assertEqual(acoes, [ACAO_ESPERAR, ACAO_DEFENDER])

  def test_resolver_fallback_destreza(self):
    modo = resolver_modo_treino("invalido")
    self.assertEqual(modo, ModoTreino.DESTREZA)
