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
         patch("src.bot.luta.obter_acao_ativa", return_value=ACAO_ESPERAR) as mock_ativa, \
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

      # ----------------------------------------------------
      # Teste 8: 2 Ataques Pesados Consecutivos Ativam Exposição
      # ----------------------------------------------------
      luta._consecutive_pesados = 0
      luta._consecutive_medios = 0
      luta._consecutive_leves = 0
      luta._exposto = False
      
      mock_rec.return_value = (0.0, False, {})
      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_PESADO
      luta.processar_frame(frame)
      self.assertEqual(luta._consecutive_pesados, 1)
      self.assertFalse(luta._exposto)

      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_ATAQUE_PESADO
      luta.processar_frame(frame)
      self.assertEqual(luta._consecutive_pesados, 2)
      self.assertTrue(luta._exposto) # Ativou exposição!

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
         patch("src.bot.luta.obter_acao_ativa", return_value=ACAO_ESPERAR) as mock_ativa, \
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
      # Teste A: APARAR mode returns exactly 5.0 on parry
      # ----------------------------------------------------
      from src.ai.modo_treino import ModoTreino

      luta = Luta(exibir_debug=False, modo_treino="aparar")
      luta._log_reward_rt = False

      luta._estado_anterior = np.zeros((4, 128, 128))
      luta._acao_anterior = ACAO_BLOQUEIO
      mock_rec.return_value = (3.0, False, {"aparar_perfeito": True})
      luta.processar_frame(frame)
      self.assertAlmostEqual(luta._ultima_recompensa, 5.0)

      # ----------------------------------------------------
      # Teste B: DESTREZA mode returns exactly 6.0 on dodge
      # ----------------------------------------------------
      luta2 = Luta(exibir_debug=False, modo_treino="destreza")
      luta2._log_reward_rt = False

      luta2._estado_anterior = np.zeros((4, 128, 128))
      luta2._acao_anterior = ACAO_ESQUIVA
      mock_rec.return_value = (6.0, False, {"destreza_perfeita": True})
      luta2.processar_frame(frame)
      self.assertAlmostEqual(luta2._ultima_recompensa, 6.0)

      # ----------------------------------------------------
      # Teste C: COMBO mode tests (correct reset, intermediate hits, incorrect combo, and block duration)
      # ----------------------------------------------------
      luta3 = Luta(exibir_debug=False, modo_treino="combo")
      luta3._log_reward_rt = False
      luta3._combo_hits = 4

      # C.1 Reset Correto
      luta3._estado_anterior = np.zeros((4, 128, 128))
      luta3._acao_anterior = ACAO_ESQUIVA
      mock_rec.return_value = (2.0, False, {})
      luta3.processar_frame(frame)
      self.assertAlmostEqual(luta3._ultima_recompensa, 10.0)
      self.assertEqual(luta3._combo_hits, 0)

      # C.2 Hit Intermediário (Hit 3)
      luta3._combo_hits = 2
      luta3._acao_anterior = ACAO_ATAQUE_LEVE
      mock_rec.return_value = (0.0, False, {"causou_dano": True})
      luta3.processar_frame(frame)
      self.assertAlmostEqual(luta3._ultima_recompensa, 1.0)
      self.assertEqual(luta3._combo_hits, 3)

      # C.3 Hit Finalizador (Hit 4)
      luta3._combo_hits = 3
      luta3._acao_anterior = ACAO_ATAQUE_LEVE
      mock_rec.return_value = (0.0, False, {"causou_dano": True})
      luta3.processar_frame(frame)
      self.assertAlmostEqual(luta3._ultima_recompensa, 2.0)
      self.assertEqual(luta3._combo_hits, 4)

      # C.4 Combo Excedido com acerto (> 4 hits)
      luta3._combo_hits = 5
      luta3._acao_anterior = ACAO_ATAQUE_LEVE
      mock_rec.return_value = (0.0, False, {"causou_dano": True})
      luta3.processar_frame(frame)
      self.assertAlmostEqual(luta3._ultima_recompensa, -3.0)
      self.assertEqual(luta3._combo_hits, 6)

      # C.4b Ataque sem causar dano (falha)
      luta3._combo_hits = 1
      luta3._acao_anterior = ACAO_ATAQUE_LEVE
      mock_rec.return_value = (0.0, False, {})
      luta3.processar_frame(frame)
      self.assertAlmostEqual(luta3._ultima_recompensa, -1.5)
      self.assertEqual(luta3._combo_hits, 1)

      # C.4c Bloqueio com causou_dano (delay de input)
      luta3._combo_hits = 1
      luta3._acao_anterior = ACAO_BLOQUEIO
      mock_rec.return_value = (0.0, False, {"causou_dano": True})
      luta3.processar_frame(frame)
      self.assertAlmostEqual(luta3._ultima_recompensa, 0.0)
      self.assertEqual(luta3._combo_hits, 0)

      # C.4d Reset do combo por tomar dano
      luta3._combo_hits = 3
      luta3._acao_anterior = ACAO_ATAQUE_LEVE
      mock_rec.return_value = (0.0, False, {"tomou_dano": True})
      luta3.processar_frame(frame)
      self.assertEqual(luta3._combo_hits, 0)

      # C.5 Bloqueio Prolongado (> 2.5s)
      luta3b = Luta(exibir_debug=False, modo_treino="combo")
      luta3b.estado = "LUTANDO"
      luta3b._log_reward_rt = False
      luta3b._estado_anterior = np.zeros((4, 128, 128))
      luta3b._acao_anterior = ACAO_BLOQUEIO
      luta3b._tempo_bloqueio_combo = 2.6
      mock_rec.return_value = (0.0, False, {})
      luta3b.processar_frame(frame)
      self.assertAlmostEqual(luta3b._ultima_recompensa, -1.5)

      # C.6 Excesso de ativações de bloqueio/esquiva independentes (3 ativações sem causar dano)
      luta3c = Luta(exibir_debug=False, modo_treino="combo")
      luta3c.estado = "LUTANDO"
      luta3c._log_reward_rt = False
      luta3c._estado_anterior = np.zeros((4, 128, 128))
      mock_rec.return_value = (0.0, False, {})
      
      # Ativação 1 (Esquiva)
      luta3c._acao_anterior = ACAO_ESQUIVA
      luta3c.processar_frame(frame)
      self.assertEqual(luta3c._esquivas_no_combo, 1)
      self.assertEqual(luta3c._bloqueios_no_combo, 0)
      
      # Espera (limpa flag)
      luta3c._acao_anterior = ACAO_ESPERAR
      luta3c.processar_frame(frame)
      
      # Ativação 2 (Bloqueio)
      luta3c._acao_anterior = ACAO_BLOQUEIO
      luta3c.processar_frame(frame)
      self.assertEqual(luta3c._esquivas_no_combo, 1)
      self.assertEqual(luta3c._bloqueios_no_combo, 1)

      # Espera
      luta3c._acao_anterior = ACAO_ESPERAR
      luta3c.processar_frame(frame)

      # Bloqueio 2
      luta3c._acao_anterior = ACAO_BLOQUEIO
      luta3c.processar_frame(frame)
      # Espera
      luta3c._acao_anterior = ACAO_ESPERAR
      luta3c.processar_frame(frame)

      # Bloqueio 3 (atinge limite de 3) -> Deve punir!
      luta3c._acao_anterior = ACAO_BLOQUEIO
      luta3c.processar_frame(frame)
      self.assertEqual(luta3c._bloqueios_no_combo, 3)
      self.assertAlmostEqual(luta3c._ultima_recompensa, -3.0)

      # No próximo frame, ACAO_BLOQUEIO deve estar mascarada, mas ACAO_ESQUIVA não!
      # E a recompensa deve ser 0.0 (não repete a punição de -3.0)
      luta3c.processar_frame(frame)
      self.assertAlmostEqual(luta3c._ultima_recompensa, 0.0)
      self.assertIn(ACAO_ESQUIVA, luta3c.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_BLOQUEIO, luta3c.cerebro.acoes_permitidas)

      # Causar dano reseta ambos os contadores e reabilita tudo
      mock_rec.return_value = (0.0, False, {"causou_dano": True})
      luta3c._acao_anterior = ACAO_ATAQUE_LEVE
      luta3c.processar_frame(frame)
      self.assertEqual(luta3c._bloqueios_no_combo, 0)
      self.assertEqual(luta3c._esquivas_no_combo, 0)
      
      mock_rec.return_value = (0.0, False, {})
      luta3c.processar_frame(frame)
      self.assertIn(ACAO_BLOQUEIO, luta3c.cerebro.acoes_permitidas)
      self.assertIn(ACAO_ESQUIVA, luta3c.cerebro.acoes_permitidas)

      # C.7 Cooldown de ataques no modo COMBO
      luta3d = Luta(exibir_debug=False, modo_treino="combo")
      luta3d.estado = "LUTANDO"
      luta3d._log_reward_rt = False
      luta3d._estado_anterior = np.zeros((4, 128, 128))
      mock_rec.return_value = (0.0, False, {})
      
      # Força o tempo inicial do cooldown
      import time as pytime
      luta3d._ultimo_ataque_ts = pytime.time()
      
      # Como o cooldown está ativo, as ações de ataque devem ser mascaradas
      luta3d.processar_frame(frame)
      self.assertEqual(luta3d.cerebro.acoes_permitidas, [ACAO_ESPERAR, ACAO_ESQUIVA, ACAO_BLOQUEIO])
      
      # Força a expiração do cooldown
      luta3d._ultimo_ataque_ts = pytime.time() - 0.4
      luta3d.processar_frame(frame)
      self.assertEqual(luta3d.cerebro.acoes_permitidas, [ACAO_ESPERAR, ACAO_ESQUIVA, ACAO_BLOQUEIO, ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO])

      # ----------------------------------------------------
      # Teste D: DEFENDER mode returns +5.0 on mitigation
      # ----------------------------------------------------
      luta4 = Luta(exibir_debug=False, modo_treino="defender")
      luta4._log_reward_rt = False

      luta4._estado_anterior = np.zeros((4, 128, 128))
      luta4._acao_anterior = ACAO_BLOQUEIO
      mock_rec.return_value = (0.0, False, {"tomou_dano": True})
      luta4.processar_frame(frame)
      self.assertAlmostEqual(luta4._ultima_recompensa, 5.0)

      # ----------------------------------------------------
      # Teste E: Bloqueio de Acoes ao Tomar Dano (Hitstun & Dano Recente)
      # ----------------------------------------------------
      luta5 = Luta(exibir_debug=False, modo_treino="completo")
      luta5.estado = "LUTANDO"
      luta5._log_reward_rt = False
      luta5.vida_jogador_anterior = 100.0

      # Simula frame onde jogador toma dano (vida cai de 100 para 95 e colunas escuras >= 1)
      mock_vida.return_value = {
        "vida_jogador_pct": 95.0,
        "vida_inimigo_pct": 100.0,
        "colunas_escuras_jogador_finais": 1,
        "colunas_escuras_inimigo_finais": 0,
      }

      # Ao processar o frame, a vida diminui e o bloqueio de dano deve ser ativado (valor padrao de 2 frames)
      # Frame 1: Dano detectado. _frames_bloqueio_dano vira 2, decrementado para 1.
      luta5.processar_frame(frame)
      self.assertEqual(luta5._frames_bloqueio_dano, 1)
      self.assertEqual(luta5.cerebro.acoes_permitidas, [ACAO_ESPERAR]) # Apenas ACAO_ESPERAR permitida

      # Frame 2: Ainda sob bloqueio absoluto. _frames_bloqueio_dano vira 0.
      mock_vida.return_value = {
        "vida_jogador_pct": 95.0,
        "vida_inimigo_pct": 100.0,
        "colunas_escuras_jogador_finais": 1,
        "colunas_escuras_inimigo_finais": 0,
      }
      luta5.vida_jogador_anterior = 95.0 # Nao mudou a vida de novo
      luta5.processar_frame(frame)
      self.assertEqual(luta5._frames_bloqueio_dano, 0)
      self.assertEqual(luta5.cerebro.acoes_permitidas, [ACAO_ESPERAR])

      # Frame 3: Bloqueio absoluto expirou, mas tomou_dano_recente ainda e True.
      # Acoes de ataque devem ser removidas, permitindo apenas esquiva, bloqueio e esperar.
      luta5.processar_frame(frame)
      self.assertEqual(luta5._frames_bloqueio_dano, 0)
      self.assertTrue(luta5._frames_dano_recente > 0) # Ainda sob dano recente
      self.assertIn(ACAO_ESQUIVA, luta5.cerebro.acoes_permitidas)
      self.assertIn(ACAO_BLOQUEIO, luta5.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_ATAQUE_LEVE, luta5.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_ATAQUE_MEDIO, luta5.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_ATAQUE_PESADO, luta5.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_ESPECIAL, luta5.cerebro.acoes_permitidas)

      # Avancamos os frames ate a expiracao de _frames_dano_recente (originalmente 8 frames)
      # Ja se passaram 3 frames. Restam 5 frames para expirar.
      for _ in range(5):
        luta5.processar_frame(frame)

      # Frame 9: O dano recente expirou (_frames_dano_recente = 0).
      # Todas as acoes do modo completo devem ser liberadas de novo (incluindo ataques).
      luta5.processar_frame(frame)
      self.assertEqual(luta5._frames_dano_recente, 0)
      self.assertIn(ACAO_ATAQUE_PESADO, luta5.cerebro.acoes_permitidas)
      self.assertIn(ACAO_ATAQUE_LEVE, luta5.cerebro.acoes_permitidas)
      self.assertGreater(len(luta5.cerebro.acoes_permitidas), 3)

      # ----------------------------------------------------
      # Teste F: Bypass de Acao Ativa (Worker Ocupado)
      # ----------------------------------------------------
      luta6 = Luta(exibir_debug=False, modo_treino="completo")
      luta6.estado = "LUTANDO"
      luta6._log_reward_rt = False
      
      # Simula worker ocupado executando ACAO_BLOQUEIO
      mock_ativa.return_value = ACAO_BLOQUEIO
      luta6._estado_anterior = np.zeros((4, 128, 128))
      luta6._acao_anterior = ACAO_BLOQUEIO
      
      # Processa o frame. Como obter_acao_ativa retorna ACAO_BLOQUEIO,
      # o loop registra acao_registrada = ACAO_BLOQUEIO
      luta6.processar_frame(frame)
      self.assertEqual(luta6._acao_anterior, ACAO_BLOQUEIO)
      
      # E o cerebro nao deve ter sido chamado para escolher acao (nenhuma acao adicionada a _pendentes)
      self.assertEqual(len(luta6.cerebro._pendentes), 0)

      # ----------------------------------------------------
      # Teste G: Mascaramento de ACAO_ESPERAR no modo combo
      # ----------------------------------------------------
      # Reseta mock_ativa para ACAO_ESPERAR
      mock_ativa.return_value = ACAO_ESPERAR
      
      luta7 = Luta(exibir_debug=False, modo_treino="combo")
      luta7.estado = "LUTANDO"
      luta7._log_reward_rt = False
      
      # Forca o esgotamento de defesas
      luta7._bloqueios_no_combo = 3
      luta7._esquivas_no_combo = 3
      
      # Processa o frame. Como bloqueios e esquivas estao esgotados,
      # ACAO_ESPERAR deve ser mascarada, alem de esquiva e bloqueio, restando apenas ataques.
      luta7.processar_frame(frame)
      self.assertNotIn(ACAO_ESPERAR, luta7.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_BLOQUEIO, luta7.cerebro.acoes_permitidas)
      self.assertNotIn(ACAO_ESQUIVA, luta7.cerebro.acoes_permitidas)
      self.assertIn(ACAO_ATAQUE_LEVE, luta7.cerebro.acoes_permitidas)
      self.assertIn(ACAO_ATAQUE_MEDIO, luta7.cerebro.acoes_permitidas)

      # ----------------------------------------------------
      # Teste H: Modo Assistido (Imitacao do Humano)
      # ----------------------------------------------------
      luta8 = Luta(exibir_debug=False, modo_treino="assistido")
      luta8.estado = "LUTANDO"
      luta8._log_reward_rt = False
      
      with patch("src.bot.luta.esta_pressionada") as mock_press:
        # Simula que o humano pressionou a tecla de bloqueio
        mock_press.side_effect = lambda vkey: vkey == luta8._vkey_bloqueio
        
        luta8._estado_anterior = np.zeros((4, 128, 128))
        luta8._acao_anterior = ACAO_ESPERAR
        mock_rec.return_value = (4.5, False, {})
        
        luta8.processar_frame(frame)
        self.assertEqual(luta8._acao_anterior, ACAO_BLOQUEIO)
        self.assertAlmostEqual(luta8._ultima_recompensa, 4.5)
