import os

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_BLOQUEIO,
  ACAO_ESQUIVA,
  ACAO_ESPERAR,
  ACAO_ESPECIAL,
)
from src.ai.deteccao_luta import TEMPLATE_DERROTA, TEMPLATE_VITORIA
from src.ai.modo_treino import resolver_modo_treino
from src.bot.rois import ROI_ACOES_PERFEITAS, ROI_SEQUENCIA_GOLPES, obter_roi
from src.bot.visao import encontrar_template

PESO_KO = -15.0
PESO_VITORIA = 20.0

PESO_DESTREZA_PERFEITA = 6.0
PESO_DESTREZA_QUASE = 2.0
PESO_DESTREZA_SEM_PERIGO = -1.0
PESO_DESTREZA_LONGE = -1.5
PESO_DESTREZA_LONGE_CONTEXTO = -0.3
PESO_SPAM_DESTREZA = -0.5

PESO_APARAR_PERFEITO = 3.0
PESO_BLOQUEIO_QUASE = 1.0
PESO_BLOQUEIO_SEM_PERIGO = -1.0
PESO_SPAM_BLOQUEIO = -0.5

PESO_TOMOU_DANO = 0.0  # legado
PESO_DANO_INIMIGO_POR_PCT = 1.5
PESO_DANO_TOMADO_POR_PCT = -2.0
PESO_TENTOU_REAGIR_ATAQUE = 0.5
PESO_NAO_REAGIU_ATAQUE = -0.2
PESO_SOBREVIVEU_JANELA_PERIGOSA = 0.5
PESO_PARADO_MUITO_TEMPO = -0.5
PESO_ATAQUE_EM_PERIGO = -0.2
PESO_ATAQUE_PESADO_TOMOU_DANO = -10.0

try:
  PESO_PUNICAO_AGRESSIVIDADE_E1 = float(os.getenv("BOT_PESO_PUNICAO_AGRESSIVIDADE_E1", "-0.25"))
except ValueError:
  PESO_PUNICAO_AGRESSIVIDADE_E1 = -0.25

try:
  PESO_PUNICAO_AGRESSIVIDADE_E2 = float(os.getenv("BOT_PESO_PUNICAO_AGRESSIVIDADE_E2", "-1.00"))
except ValueError:
  PESO_PUNICAO_AGRESSIVIDADE_E2 = -1.00

try:
  PESO_PUNICAO_DEFESA_SEM_ESPECIAL = float(os.getenv("BOT_PESO_PUNICAO_DEFESA_SEM_ESPECIAL", "-0.50"))
except ValueError:
  PESO_PUNICAO_DEFESA_SEM_ESPECIAL = -0.50

try:
  PESO_PUNICAO_OPONENTE_E3 = float(os.getenv("BOT_PESO_PUNICAO_OPONENTE_E3", "-5.00"))
except ValueError:
  PESO_PUNICAO_OPONENTE_E3 = -5.00

RECOMPENSA_COMBO_ATIVO = 5.0
PENALIDADE_SEM_COMBO = -1.0

PESO_BLOQUEIO_PERFEITO = PESO_APARAR_PERFEITO

try:
  LIMIAR_DISTANCIA_DESTREZA_RUIM = int(os.getenv("BOT_DISTANCIA_DESTREZA_RUIM_PX", "300"))
except ValueError:
  LIMIAR_DISTANCIA_DESTREZA_RUIM = 300

try:
  LIMIAR_PARADO_FRAMES = int(os.getenv("BOT_LIMIAR_PARADO_FRAMES", "45"))
except ValueError:
  LIMIAR_PARADO_FRAMES = 45

try:
  LIMIAR_DESTREZA = float(os.getenv("BOT_LIMIAR_DESTREZA", "0.8"))
except ValueError:
  LIMIAR_DESTREZA = 0.8

try:
  LIMIAR_APARAR = float(os.getenv("BOT_LIMIAR_APARAR", "0.8"))
except ValueError:
  LIMIAR_APARAR = 0.8

try:
  LIMIAR_COMBO = float(os.getenv("BOT_LIMIAR_COMBO", "0.8"))
except ValueError:
  LIMIAR_COMBO = 0.8

def _detectar_destreza_perfeita(frame):
  roi = obter_roi(ROI_ACOES_PERFEITAS)
  return encontrar_template(frame, "assets/destreza.png", limiar=LIMIAR_DESTREZA, roi=roi) is not None


def _detectar_aparar_perfeito(frame):
  roi = obter_roi(ROI_ACOES_PERFEITAS)
  return encontrar_template(frame, "assets/aparar.png", limiar=LIMIAR_APARAR, roi=roi) is not None


def _detectar_combo_ativo(frame):
  roi = obter_roi(ROI_SEQUENCIA_GOLPES)
  return encontrar_template(
    frame,
    "assets/sequencia-golpes.png",
    limiar=LIMIAR_COMBO,
    roi=roi,
  ) is not None


def _detectar_tomou_dano(vida_jogador_atual, vida_jogador_anterior, colunas_escuras_jogador_finais):
  if (
    vida_jogador_atual is None
    or vida_jogador_anterior is None
    or vida_jogador_atual >= vida_jogador_anterior
    or colunas_escuras_jogador_finais < 1
  ):
    return False, 0.0

  delta_dano = float(vida_jogador_anterior) - float(vida_jogador_atual)
  return True, max(0.0, delta_dano)


def _detectar_causou_dano(vida_inimigo_atual, vida_inimigo_anterior):
  if (
    vida_inimigo_atual is None
    or vida_inimigo_anterior is None
    or vida_inimigo_atual >= vida_inimigo_anterior
  ):
    return False, 0.0

  delta_dano = float(vida_inimigo_anterior) - float(vida_inimigo_atual)
  return True, max(0.0, delta_dano)


def _distancia_perto(distancia_px):
  if distancia_px is None:
    return None
  try:
    return float(distancia_px) <= float(LIMIAR_DISTANCIA_DESTREZA_RUIM)
  except (TypeError, ValueError):
    return None


def _base_info(acao_atual):
  return {
    "nocaute": False,
    "vitoria": False,
    "tomou_dano": False,
    "tomou_dano_delta": 0.0,
    "causou_dano": False,
    "dano_inimigo_delta": 0.0,
    "acao_esquiva": acao_atual == ACAO_ESQUIVA,
    "acao_destreza": acao_atual == ACAO_ESQUIVA,
    "acao_bloqueio": acao_atual == ACAO_BLOQUEIO,
    "acao_ataque": acao_atual in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO),
    "ataque_pesado_tentado": acao_atual == ACAO_ATAQUE_PESADO,
    "esquiva_tentada": False,
    "destreza_tentada": False,
    "destreza_perfeita": False,
    "destreza_quase": False,
    "destreza_errada": False,
    "aparar_perfeito": False,
    "bloqueio_tentado": False,
    "bloqueio_quase": False,
    "bloqueio_errado": False,
    "sobreviveu_perigo": False,
    "tentou_reagir_ataque": False,
    "nao_reagiu_ataque": False,
    "spam_destreza": False,
    "spam_bloqueio": False,
    "destreza_longe_ruim": False,
    "parado_muito_tempo": False,
    "combo_ativo": False,
    "combo_bonus": 0.0,
    "combo_penalidade": 0.0,
    "ataque_pesado_tomou_dano": False,
    "ataque_em_perigo": False,
    "punicao_agressividade_especial": False,
    "punicao_defesa_sem_especial": False,
    "punicao_oponente_e3": False,
    "gratificacao_defesa_especial_recente": False,
    "destreza_neutra_especial_recente": False,
    "punicao_ofensiva_especial_recente": False,
    "punicao_espera_especial_recente": False,
  }


def _recompensa_acao(
  info,
  frame,
  acao_atual,
  inimigo_atacando,
  tomou_dano,
  distancia_px,
):
  recompensa = 0.0

  if acao_atual == ACAO_ESQUIVA:
    info["esquiva_tentada"] = True
    info["destreza_tentada"] = True
    dist_perto = _distancia_perto(distancia_px)
    destreza_longe_ruim = dist_perto is False
    if destreza_longe_ruim:
      recompensa += PESO_DESTREZA_LONGE_CONTEXTO if inimigo_atacando else PESO_DESTREZA_LONGE
      info["destreza_longe_ruim"] = True
      info["destreza_errada"] = True
      return recompensa

    destreza_perfeita = _detectar_destreza_perfeita(frame)
    info["destreza_perfeita"] = destreza_perfeita
    if destreza_perfeita:
      recompensa += PESO_DESTREZA_PERFEITA
    elif inimigo_atacando and (not tomou_dano):
      recompensa += PESO_DESTREZA_QUASE
      info["destreza_quase"] = True
    else:
      recompensa += PESO_DESTREZA_SEM_PERIGO
      info["destreza_errada"] = True
    return recompensa

  if acao_atual == ACAO_BLOQUEIO:
    info["bloqueio_tentado"] = True
    aparar_perfeito = _detectar_aparar_perfeito(frame)
    info["aparar_perfeito"] = aparar_perfeito
    if aparar_perfeito:
      recompensa += PESO_APARAR_PERFEITO
    elif inimigo_atacando and (not tomou_dano):
      recompensa += PESO_BLOQUEIO_QUASE
      info["bloqueio_quase"] = True
    else:
      recompensa += PESO_BLOQUEIO_SEM_PERIGO
      info["bloqueio_errado"] = True
    return recompensa

  if acao_atual == ACAO_ATAQUE_PESADO and tomou_dano:
    recompensa += PESO_ATAQUE_PESADO_TOMOU_DANO
    info["ataque_pesado_tomou_dano"] = True

  if acao_atual in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO) and inimigo_atacando:
    recompensa += PESO_ATAQUE_EM_PERIGO
    info["ataque_em_perigo"] = True

  return recompensa


def _aplicar_eventos_comuns(
  info,
  recompensa,
  acao_atual,
  inimigo_atacando,
  aplicar_penalidade_nao_reagiu,
  tomou_dano,
  destreza_consecutiva,
  bloqueio_consecutivo,
  tempo_parado_frames,
):
  acao_reacao = acao_atual in (ACAO_ESQUIVA, ACAO_BLOQUEIO)
  if acao_atual == ACAO_ESQUIVA and info.get("destreza_longe_ruim"):
    acao_reacao = False

  if inimigo_atacando:
    if acao_reacao:
      recompensa += PESO_TENTOU_REAGIR_ATAQUE
      info["tentou_reagir_ataque"] = True
    elif aplicar_penalidade_nao_reagiu:
      recompensa += PESO_NAO_REAGIU_ATAQUE
      info["nao_reagiu_ataque"] = True

  if inimigo_atacando and (not tomou_dano):
    recompensa += PESO_SOBREVIVEU_JANELA_PERIGOSA
    info["sobreviveu_perigo"] = True

  if destreza_consecutiva and acao_atual == ACAO_ESQUIVA:
    recompensa += PESO_SPAM_DESTREZA
    info["spam_destreza"] = True

  if bloqueio_consecutivo and acao_atual == ACAO_BLOQUEIO:
    recompensa += PESO_SPAM_BLOQUEIO
    info["spam_bloqueio"] = True

  if tempo_parado_frames >= LIMIAR_PARADO_FRAMES:
    recompensa += PESO_PARADO_MUITO_TEMPO
    info["parado_muito_tempo"] = True

  return recompensa


def obter_recompensa(
  frame,
  acao_atual=None,
  inimigo_atacando=False,
  aplicar_penalidade_nao_reagiu=True,
  aplicar_pontuacao_combo=True,
  vida_jogador_atual=None,
  vida_jogador_anterior=None,
  vida_inimigo_atual=None,
  vida_inimigo_anterior=None,
  colunas_escuras_jogador_finais=0,
  colunas_escuras_inimigo_finais=0,
  nocaute_detectado=None,
  vitoria_detectada=None,
  destreza_consecutiva=False,
  bloqueio_consecutivo=False,
  distancia_px=None,
  modo_treino="treino",
  tempo_parado_frames=0,
  delta_tempo_seg=None,
  historico_acoes=None,
  nivel_especial_inimigo=None,
  oponente_especial_recente=False,
):
  _ = (colunas_escuras_inimigo_finais,)
  _ = resolver_modo_treino(modo_treino)

  if nocaute_detectado is None:
    nocaute = (frame is not None) and (encontrar_template(frame, TEMPLATE_DERROTA) is not None)
  else:
    nocaute = bool(nocaute_detectado)

  if vitoria_detectada is None:
    vitoria = (frame is not None) and (encontrar_template(frame, TEMPLATE_VITORIA) is not None)
  else:
    vitoria = bool(vitoria_detectada)

  info = _base_info(acao_atual)
  if nocaute:
    info["nocaute"] = True
    # Penalty Decay: A punicao e mitigada pelo dano infligido ao oponente
    if vida_inimigo_atual is not None:
      multiplicador = max(0.0, float(vida_inimigo_atual)) / 100.0
      penalidade_final = PESO_KO * multiplicador
    else:
      penalidade_final = PESO_KO
    return penalidade_final, True, info

  if vitoria:
    info["vitoria"] = True
    return PESO_VITORIA, True, info

  tomou_dano, delta_dano = _detectar_tomou_dano(
    vida_jogador_atual,
    vida_jogador_anterior,
    colunas_escuras_jogador_finais,
  )
  causou_dano, delta_inimigo = _detectar_causou_dano(
    vida_inimigo_atual,
    vida_inimigo_anterior,
  )

  _ = (delta_tempo_seg,)

  recompensa = 0.0
  if frame is not None:
    combo_ativo = _detectar_combo_ativo(frame)
    info["combo_ativo"] = bool(combo_ativo)
    if aplicar_pontuacao_combo:
      if combo_ativo:
        recompensa += RECOMPENSA_COMBO_ATIVO
        info["combo_bonus"] = RECOMPENSA_COMBO_ATIVO
      else:
        recompensa += PENALIDADE_SEM_COMBO
        info["combo_penalidade"] = PENALIDADE_SEM_COMBO

  # Modelação de Recompensa Baseada em Potencial (PBRS)
  # F(s, a, s') = gamma * Phi(s') - Phi(s)
  gamma = 0.99
  fator_escala = 50.0  # Para manter o peso compativel com a escala heuristica anterior
  
  if vida_jogador_atual is not None and vida_inimigo_atual is not None and vida_jogador_anterior is not None and vida_inimigo_anterior is not None:
    # O diferencial de saude e normalizado (-1 a 1)
    phi_s = (float(vida_jogador_anterior) - float(vida_inimigo_anterior)) / 100.0
    phi_s_linha = (float(vida_jogador_atual) - float(vida_inimigo_atual)) / 100.0
    
    f_s_a_s_linha = (gamma * phi_s_linha) - phi_s
    recompensa += f_s_a_s_linha * fator_escala
  else:
    # Caso as informações de vida estejam incompletas para PBRS, usamos o dano clássico como fallback (ex: testes ou falha de visão)
    if tomou_dano:
      recompensa += delta_dano * PESO_DANO_TOMADO_POR_PCT
    if causou_dano:
      recompensa += delta_inimigo * PESO_DANO_INIMIGO_POR_PCT
    
  # Atualiza a info para depuracao e logs
  if tomou_dano:
    info["tomou_dano"] = True
    info["tomou_dano_delta"] = delta_dano

  if causou_dano:
    info["causou_dano"] = True
    info["dano_inimigo_delta"] = delta_inimigo

  if oponente_especial_recente:
    # Ignora recompensa padrão da ação e aplica regras específicas da janela defensiva
    if acao_atual == ACAO_BLOQUEIO:
      info["bloqueio_tentado"] = True
      aparar_perfeito = _detectar_aparar_perfeito(frame)
      info["aparar_perfeito"] = aparar_perfeito
      if aparar_perfeito:
        recompensa += PESO_APARAR_PERFEITO
      else:
        # Gratificação por segurar bloqueio durante especial
        recompensa += 2.0
        info["gratificacao_defesa_especial_recente"] = True
    elif acao_atual == ACAO_ESQUIVA:
      info["esquiva_tentada"] = True
      info["destreza_tentada"] = True
      destreza_perfeita = _detectar_destreza_perfeita(frame)
      info["destreza_perfeita"] = destreza_perfeita
      if destreza_perfeita:
        recompensa += PESO_DESTREZA_PERFEITA
      else:
        # Destreza não é punida durante o especial recente (recompensa neutra: 0.0)
        recompensa += 0.0
        info["destreza_neutra_especial_recente"] = True
    elif acao_atual in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO, ACAO_ESPECIAL):
      recompensa += -5.0
      info["punicao_ofensiva_especial_recente"] = True
    elif acao_atual == ACAO_ESPERAR:
      recompensa += -2.0
      info["punicao_espera_especial_recente"] = True
  else:
    recompensa += _recompensa_acao(
      info,
      frame,
      acao_atual,
      bool(inimigo_atacando),
      tomou_dano,
      distancia_px,
    )

  recompensa = _aplicar_eventos_comuns(
    info,
    recompensa,
    acao_atual,
    bool(inimigo_atacando),
    bool(aplicar_penalidade_nao_reagiu),
    tomou_dano,
    bool(destreza_consecutiva),
    bool(bloqueio_consecutivo),
    int(tempo_parado_frames),
  )

  # Stale Moves Penalty
  if historico_acoes and recompensa > 0.0 and acao_atual not in (ACAO_ESQUIVA, ACAO_BLOQUEIO, ACAO_ESPERAR):
    k = sum(1 for a in historico_acoes if a == acao_atual)
    if k > 0:
      lambda_degrade = 0.5
      recompensa = recompensa * (lambda_degrade ** k)
      if k >= 4:
        # Se passar mais de 4 frames na memória executando o mesmo golpe, aplica punição pesada
        recompensa -= 2.0 * (k - 3)

  if nivel_especial_inimigo is not None:
    # 1. Pune agressividade sob especial 1 e 2
    esta_atacando = acao_atual in (ACAO_ATAQUE_LEVE, ACAO_ATAQUE_MEDIO, ACAO_ATAQUE_PESADO)
    if esta_atacando:
      if nivel_especial_inimigo == 1:
        recompensa += PESO_PUNICAO_AGRESSIVIDADE_E1
        info["punicao_agressividade_especial"] = True
      elif nivel_especial_inimigo == 2:
        recompensa += PESO_PUNICAO_AGRESSIVIDADE_E2
        info["punicao_agressividade_especial"] = True

    # 2. Pune postura defensiva se o oponente não tem especial carregado (E0)
    elif acao_atual in (ACAO_ESQUIVA, ACAO_BLOQUEIO):
      if nivel_especial_inimigo == 0:
        recompensa += PESO_PUNICAO_DEFESA_SEM_ESPECIAL
        info["punicao_defesa_sem_especial"] = True

    # 3. Pune se o oponente atingir o especial de nível 3 (E3)
    if nivel_especial_inimigo == 3:
      recompensa += PESO_PUNICAO_OPONENTE_E3
      info["punicao_oponente_e3"] = True

  return recompensa, False, info
