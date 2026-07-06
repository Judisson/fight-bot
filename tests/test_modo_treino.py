from unittest import TestCase

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_BLOQUEIO,
  ACAO_ESPERAR,
  ACAO_ESQUIVA,
  ACOES_IA,
)
from src.ai.modo_treino import ModoTreino, obter_acoes_permitidas, resolver_modo_treino


class TestModoTreino(TestCase):

  def test_resolver_treino(self):
    self.assertEqual(resolver_modo_treino("completo"), ModoTreino.COMPLETO)
    self.assertEqual(resolver_modo_treino("aparar"), ModoTreino.APARAR)
    self.assertEqual(resolver_modo_treino("defender"), ModoTreino.DEFENDER)
    self.assertEqual(resolver_modo_treino("combo"), ModoTreino.COMBO)
    self.assertEqual(resolver_modo_treino("destreza"), ModoTreino.DESTREZA)

  def test_modo_invalido_faz_fallback_para_completo(self):
    self.assertEqual(resolver_modo_treino("invalido_qualquer"), ModoTreino.COMPLETO)

  def test_acoes_permitidas_completo(self):
    self.assertEqual(obter_acoes_permitidas("completo"), list(ACOES_IA))

  def test_acoes_permitidas_aparar(self):
    self.assertEqual(obter_acoes_permitidas("aparar"), [ACAO_ESPERAR, ACAO_BLOQUEIO])

  def test_acoes_permitidas_defender(self):
    self.assertEqual(obter_acoes_permitidas("defender"), [ACAO_ESPERAR, ACAO_ESQUIVA, ACAO_BLOQUEIO])

  def test_acoes_permitidas_destreza(self):
    self.assertEqual(obter_acoes_permitidas("destreza"), [ACAO_ESPERAR, ACAO_ESQUIVA])

  def test_acoes_permitidas_combo(self):
    self.assertEqual(
      obter_acoes_permitidas("combo"),
      [ACAO_ESPERAR, ACAO_ESQUIVA, ACAO_BLOQUEIO, ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO]
    )
