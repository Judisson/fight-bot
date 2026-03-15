from unittest import TestCase

from src.ai.acoes import ACOES_IA
from src.ai.modo_treino import ModoTreino, obter_acoes_permitidas, resolver_modo_treino


class TestModoTreino(TestCase):

  def test_resolver_treino(self):
    modo = resolver_modo_treino("treino")
    self.assertEqual(modo, ModoTreino.TREINO)

  def test_modo_invalido_faz_fallback_para_treino(self):
    self.assertEqual(resolver_modo_treino("destreza"), ModoTreino.TREINO)

  def test_acoes_permitidas_contem_todo_espaco(self):
    acoes = obter_acoes_permitidas("treino")
    self.assertEqual(acoes, list(ACOES_IA))
