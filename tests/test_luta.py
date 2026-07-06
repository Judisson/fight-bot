import numpy as np
from unittest import TestCase
from unittest.mock import patch

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_ESQUIVA,
  ACAO_BLOQUEIO,
  ACAO_ESPERAR,
  ACAO_ESPECIAL,
)
from src.ai.modo_treino import resolver_modo_treino, obter_acoes_permitidas
from src.bot.luta import Luta


class TestLutaComboExposicao(TestCase):

  @patch("src.bot.luta.resolver_modo_treino")
  @patch("src.bot.luta.obter_acoes_permitidas")
  @patch("src.bot.luta.CerebroIA")
  @patch("src.bot.luta.obter_estado_luta")
  def test_exposicao_combo_regras(self, mock_obter_estado_luta, mock_cerebro, mock_obter_acoes, mock_resolver):
    mock_resolver.side_effect = resolver_modo_treino
    mock_obter_acoes.side_effect = obter_acoes_permitidas
    # Configura mocks para inicialização da Luta
    mock_obter_estado_luta.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}

    luta = Luta(exibir_debug=False)
    luta._log_reward_rt = False

    # Mock de obter_info_vida, obter_info_poder, obter_recompensa e executar_acao
    with patch("src.bot.luta.obter_info_vida") as mock_vida, \
         patch("src.bot.luta.obter_info_poder") as mock_poder, \
         patch("src.bot.luta.obter_recompensa") as mock_rec, \
         patch("src.bot.luta.executar_acao", return_value=True):

      mock_vida.return_value = {
        "vida_jogador_pct": 100.0,
        "vida_inimigo_pct": 100.0,
        "colunas_escuras_jogador_finais": 0,
        "colunas_escuras_inimigo_finais": 0,
      }
      mock_poder.return_value = {
        "nivel_especial_jogador": 0,
        "nivel_especial_inimigo": 0,
        "tem_especial_jogador": False,
        "tem_especial_inimigo": False,
      }

      # frame dummy
      frame = np.zeros((360, 640, 3), dtype=np.uint8)

      # ----------------------------------------------------
      # Teste 1: 2 Ataques Médios Consecutivos Ativam Exposição
      # ----------------------------------------------------
      mock_rec.return_value = (0.0, False, {"causou_dano": False})

      # Frame 1: Escolhe e executa o 1º ataque médio
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_MEDIO
      luta.processar_frame(frame)
      self.assertEqual(luta._consecutive_medios, 1)
      self.assertFalse(luta._exposto)

      # Frame 2: Executa o 2º ataque médio
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_MEDIO
      luta.processar_frame(frame)
      self.assertEqual(luta._consecutive_medios, 2)
      self.assertTrue(luta._exposto) # Ativou exposição!

      # Frame 3: Tenta atacar (ACAO_ATAQUE_LEVE) estando exposto -> Deve punir!
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_LEVE
      mock_rec.return_value = (1.5, False, {})
      luta.processar_frame(frame)
      # Recompensa esperada: 1.5 + (-8.0) + (-2.0 punição spam atk) = -8.5
      self.assertAlmostEqual(luta._ultima_recompensa, -8.5)

      # ----------------------------------------------------
      # Teste 2: Esquiva Reseta Exposição
      # ----------------------------------------------------
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ESQUIVA
      mock_rec.return_value = (0.0, False, {})
      luta.processar_frame(frame)
      self.assertFalse(luta._exposto)
      self.assertEqual(luta._consecutive_medios, 0)

      # ----------------------------------------------------
      # Teste 3: 4 Ataques Leves Consecutivos Ativam Exposição
      # ----------------------------------------------------
      mock_rec.return_value = (0.0, False, {"causou_dano": False})

      for i in range(1, 4):
        luta._estado_anterior = np.zeros((4, 128, 128))
        luta._acao_anterior = ACAO_ATAQUE_LEVE
        luta.processar_frame(frame)
        self.assertEqual(luta._consecutive_leves, i)
        self.assertFalse(luta._exposto)

      # 4º Ataque Leve
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_LEVE
      luta.processar_frame(frame)
      self.assertEqual(luta._consecutive_leves, 4)
      self.assertTrue(luta._exposto) # Ativou exposição!

      # ----------------------------------------------------
      # Teste 4: Aparar Perfeito Reseta Exposição
      # ----------------------------------------------------
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_BLOQUEIO
      mock_rec.return_value = (3.0, False, {"aparar_perfeito": True})
      luta.processar_frame(frame)
      self.assertFalse(luta._exposto)

      # ----------------------------------------------------
      # Teste 5: 5 hits de combo (M, L, L, L, M) ativam exposição
      # ----------------------------------------------------
      # Simula 5 hits landando (causou_dano = True)
      mock_rec.return_value = (1.0, False, {"causou_dano": True})

      sequencia = [
        ACAO_ATAQUE_MEDIO,
        ACAO_ATAQUE_LEVE,
        ACAO_ATAQUE_LEVE,
        ACAO_ATAQUE_LEVE,
        ACAO_ATAQUE_MEDIO,
      ]
      for acao_combo in sequencia:
        luta._estado_anterior = np.zeros((4, 128, 128))
        luta._acao_anterior = acao_combo
        luta.processar_frame(frame)

      self.assertEqual(luta._combo_hits, 5)
      self.assertTrue(luta._exposto) # Fica exposto imediatamente após finalizar o combo!

      # ----------------------------------------------------
      # Teste 6: Tomar dano (apanhar) Reseta Exposição
      # ----------------------------------------------------
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ESPERAR
      mock_rec.return_value = (-2.0, False, {"tomou_dano": True})
      luta.processar_frame(frame)
      self.assertFalse(luta._exposto)

      # ----------------------------------------------------
      # Teste 7: Tempo de 1.5s Reseta Exposição
      # ----------------------------------------------------
      # Entra em exposição via 2 M
      mock_rec.return_value = (0.0, False, {})
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_MEDIO
      luta.processar_frame(frame)
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_MEDIO
      luta.processar_frame(frame)
      self.assertTrue(luta._exposto)

      # Passam 1.6 segundos
      with patch("src.bot.luta.Luta._obter_delta_tempo_seg", return_value=1.6):
        luta._estado_anterior = np.zeros((4, 128, 128))
        luta._acao_anterior = ACAO_ESPERAR
        luta.processar_frame(frame)
        self.assertFalse(luta._exposto)

  @patch("src.bot.luta.resolver_modo_treino")
  @patch("src.bot.luta.obter_acoes_permitidas")
  @patch("src.bot.luta.CerebroIA")
  @patch("src.bot.luta.obter_estado_luta")
  def test_focused_training_rewards(self, mock_obter_estado_luta, mock_cerebro, mock_obter_acoes, mock_resolver):
    mock_resolver.side_effect = resolver_modo_treino
    mock_obter_acoes.side_effect = obter_acoes_permitidas
    mock_obter_estado_luta.return_value = {"em_luta": True, "nocaute": False, "vitoria": False}
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    with patch("src.bot.luta.obter_info_vida") as mock_vida, \
         patch("src.bot.luta.obter_info_poder") as mock_poder, \
         patch("src.bot.luta.obter_recompensa") as mock_rec, \
         patch("src.bot.luta.executar_acao", return_value=True):

      mock_vida.return_value = {
        "vida_jogador_pct": 100.0,
        "vida_inimigo_pct": 100.0,
        "colunas_escuras_jogador_finais": 0,
        "colunas_escuras_inimigo_finais": 0,
      }
      mock_poder.return_value = {
        "nivel_especial_jogador": 0,
        "nivel_especial_inimigo": 0,
        "tem_especial_jogador": False,
        "tem_especial_inimigo": False,
      }

      # ----------------------------------------------------
      # Teste A: APARAR mode doubles parry reward
      # ----------------------------------------------------
      from src.ai.modo_treino import ModoTreino
      from src.ai.recompensa import PESO_APARAR_PERFEITO

      luta = Luta(exibir_debug=False, modo_treino="aparar")
      luta._log_reward_rt = False

      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_BLOQUEIO
      # obter_recompensa retorna base de 3.0 para parry
      mock_rec.return_value = (PESO_APARAR_PERFEITO, False, {"aparar_perfeito": True})
      luta.processar_frame(frame)
      # Recompensa deve ser duplicada: 3.0 + 3.0 = 6.0
      self.assertAlmostEqual(luta._ultima_recompensa, 2 * PESO_APARAR_PERFEITO)

      # ----------------------------------------------------
      # Teste B: DESTREZA mode doubles dodge reward
      # ----------------------------------------------------
      from src.ai.recompensa import PESO_DESTREZA_PERFEITA

      luta2 = Luta(exibir_debug=False, modo_treino="destreza")
      luta2._log_reward_rt = False

      luta2._estado_anterior = np.zeros((4, 128, 128))
      luta2._acao_anterior = ACAO_ESQUIVA
      # obter_recompensa retorna base de 6.0 para destreza
      mock_rec.return_value = (PESO_DESTREZA_PERFEITA, False, {"destreza_perfeita": True})
      luta2.processar_frame(frame)
      # Recompensa deve ser duplicada: 6.0 + 6.0 = 12.0
      self.assertAlmostEqual(luta2._ultima_recompensa, 2 * PESO_DESTREZA_PERFEITA)

      # ----------------------------------------------------
      # Teste C: COMBO mode gives defensive reset bonus (+10.0)
      # ----------------------------------------------------
      luta3 = Luta(exibir_debug=False, modo_treino="combo")
      luta3._log_reward_rt = False
      luta3._combo_hits = 4 # já landed 4 hits

      luta3._estado_anterior = np.zeros((4, 128, 128))
      luta3._acao_anterior = ACAO_ESQUIVA
      mock_rec.return_value = (2.0, False, {})
      luta3.processar_frame(frame)
      # Recompensa deve incluir o bônus: 2.0 + 10.0 = 12.0
      self.assertAlmostEqual(luta3._ultima_recompensa, 12.0)
      self.assertEqual(luta3._combo_hits, 0) # deve ter resetado
