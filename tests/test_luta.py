import time
import numpy as np
from unittest import TestCase
from unittest.mock import patch, MagicMock

from src.bot.acoes_luta import (
  ACAO_ESPERAR,
  ACAO_ESQUIVA,
  ACAO_BLOQUEIO,
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ESPECIAL,
  ACAO_BLOQUEIO_3S,
)
from src.bot.luta import Luta


class TestLutaAutomatizada(TestCase):

  def setUp(self):
    self.patcher_vida = patch("src.bot.luta.obter_info_vida")
    self.mock_obter_vida = self.patcher_vida.start()
    self.mock_obter_vida.return_value = {"vida_jogador_pct": 100.0, "vida_inimigo_pct": 100.0}

  def tearDown(self):
    self.patcher_vida.stop()

  def test_luta_inicializacao(self):
    luta = Luta(exibir_debug=False)
    self.assertEqual(luta.episodio, 0)
    self.assertEqual(luta._acoes_combo, [])
    self.assertEqual(luta._bloqueio_especial_ate, 0.0)

  @patch("src.bot.luta.obter_estado_luta")
  def test_processar_frame_nao_em_luta(self, mock_estado):
    mock_estado.return_value = {"em_luta": False, "nocaute": False, "vitoria": False}
    
    luta = Luta(exibir_debug=False)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    res = luta.processar_frame(frame)
    self.assertFalse(res["em_luta"])
    self.assertIsNone(res["terminal"])

  @patch("src.bot.luta.obter_info_poder")
  @patch("src.bot.luta.obter_estado_luta")
  @patch("src.bot.luta.executar_acao")
  def test_combo_longa_distancia(self, mock_executar, mock_estado, mock_poder):
    mock_estado.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    mock_poder.return_value = {
      "tem_especial_jogador": False,
      "tem_especial_inimigo": False,
      "nivel_especial_inimigo": 0
    }
    
    luta = Luta(exibir_debug=False)
    
    # Simula que o rastreador de personagens encontrou distância > 500px
    luta.rastreador = MagicMock()
    luta.rastreador.detectar.return_value = {"distancia_px": 550}
    
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    res = luta.processar_frame(frame)
    
    self.assertTrue(res["em_luta"])
    # Deve iniciar o combo longa distância.
    # Restam 5 ações na fila de combo:
    self.assertEqual(len(luta._acoes_combo), 5)
    
    # Verifica se chamou a primeira ação (Médio)
    mock_executar.assert_any_call(ACAO_ATAQUE_MEDIO)

  @patch("src.bot.luta.obter_info_poder")
  @patch("src.bot.luta.obter_estado_luta")
  @patch("src.bot.luta.executar_acao")
  def test_combo_curta_distancia(self, mock_executar, mock_estado, mock_poder):
    mock_estado.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    mock_poder.return_value = {
      "tem_especial_jogador": False,
      "tem_especial_inimigo": False,
      "nivel_especial_inimigo": 0
    }
    
    luta = Luta(exibir_debug=False)
    
    # Simula que o rastreador de personagens encontrou distância <= 300px
    luta.rastreador = MagicMock()
    luta.rastreador.detectar.return_value = {"distancia_px": 250}
    
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    res = luta.processar_frame(frame)
    
    # Deve iniciar o combo curta distância.
    # A primeira ação (ACAO_ATAQUE_LEVE) é executada imediatamente
    self.assertEqual(len(luta._acoes_combo), 5)
    
    mock_executar.assert_any_call(ACAO_ATAQUE_LEVE)

  @patch("src.bot.luta.obter_info_poder")
  @patch("src.bot.luta.obter_estado_luta")
  @patch("src.bot.luta.executar_acao")
  def test_combo_especial(self, mock_executar, mock_estado, mock_poder):
    mock_estado.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    mock_poder.return_value = {
      "tem_especial_jogador": True, # Bot tem especial!
      "tem_especial_inimigo": False,
      "nivel_especial_inimigo": 0
    }
    
    luta = Luta(exibir_debug=False)
    luta.rastreador = MagicMock()
    luta.rastreador.detectar.return_value = {"distancia_px": 550}
    
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    res = luta.processar_frame(frame)
    
    # Deve iniciar o combo especial longo.
    # Primeira ação (Médio) disparada imediatamente
    self.assertEqual(len(luta._acoes_combo), 5)
    
    mock_executar.assert_any_call(ACAO_ATAQUE_MEDIO)

  @patch("src.bot.luta.obter_info_poder")
  @patch("src.bot.luta.obter_estado_luta")
  @patch("src.bot.luta.executar_acao")
  def test_combo_especial_curto(self, mock_executar, mock_estado, mock_poder):
    mock_estado.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    mock_poder.return_value = {
      "tem_especial_jogador": True,
      "tem_especial_inimigo": False,
      "nivel_especial_inimigo": 0
    }
    
    luta = Luta(exibir_debug=False)
    luta.rastreador = MagicMock()
    luta.rastreador.detectar.return_value = {"distancia_px": 250}
    
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    res = luta.processar_frame(frame)
    
    # Deve iniciar o combo especial curto.
    # Primeira ação (Leve) disparada imediatamente
    self.assertEqual(len(luta._acoes_combo), 5)
    
    mock_executar.assert_any_call(ACAO_ATAQUE_LEVE)

  @patch("src.bot.luta.obter_info_poder")
  @patch("src.bot.luta.obter_estado_luta")
  @patch("src.bot.luta.executar_acao")
  def test_defesa_especial_inimigo(self, mock_executar, mock_estado, mock_poder):
    luta = Luta(exibir_debug=False)
    luta.rastreador = MagicMock()
    luta.rastreador.detectar.return_value = {"distancia_px": 350}
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # 1. Simula 5 frames de especial ativo para confirmar
    mock_estado.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    mock_poder.return_value = {
      "tem_especial_jogador": False,
      "tem_especial_inimigo": True,
      "nivel_especial_inimigo": 1
    }
    
    for _ in range(5):
      luta.processar_frame(frame)
    
    self.assertTrue(luta._inimigo_tem_especial_confirmado)
    
    # 2. Simula 5 frames de especial inativo para detectar a liberação
    mock_poder.return_value = {
      "tem_especial_jogador": False,
      "tem_especial_inimigo": False,
      "nivel_especial_inimigo": 0
    }
    
    mock_executar.reset_mock()
    for _ in range(5):
      luta.processar_frame(frame)
    
    # Deve limpar o combo e executar ACAO_BLOQUEIO_3S
    self.assertEqual(luta._acoes_combo, [])
    mock_executar.assert_any_call(ACAO_BLOQUEIO_3S)
    self.assertTrue(luta._bloqueio_especial_ate > time.time())

  @patch("src.bot.luta.obter_info_poder")
  @patch("src.bot.luta.obter_estado_luta")
  @patch("src.bot.luta.executar_acao")
  def test_deteccao_tomou_dano(self, mock_executar, mock_estado, mock_poder):
    mock_estado.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    mock_poder.return_value = {
      "tem_especial_jogador": False,
      "tem_especial_inimigo": False,
      "nivel_especial_inimigo": 0
    }
    
    luta = Luta(exibir_debug=False)
    luta.rastreador = MagicMock()
    luta.rastreador.detectar.return_value = {"distancia_px": 550}
    
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # 1. Primeiro frame: vida está 100.0.
    self.mock_obter_vida.return_value = {"vida_jogador_pct": 100.0}
    res = luta.processar_frame(frame)
    
    # Iniciou combo longa distância
    self.assertEqual(len(luta._acoes_combo), 5)
    mock_executar.assert_any_call(ACAO_ATAQUE_MEDIO)
    mock_executar.reset_mock()
    
    # 2. Segundo frame: vida cai para 95.0 (tomou dano).
    # O combo anterior deve ser limpo e deve agendar Destreza, Destreza.
    self.mock_obter_vida.return_value = {"vida_jogador_pct": 95.0}
    res = luta.processar_frame(frame)
    
    # O combo agora deve ter a segunda esquiva na fila (1 ação restando)
    self.assertEqual(len(luta._acoes_combo), 1)
    self.assertEqual(luta._acoes_combo[0][0], ACAO_ESQUIVA)
    
    # Deve ter executado a primeira esquiva do reposicionamento imediatamente
    mock_executar.assert_called_once_with(ACAO_ESQUIVA)
