import atexit
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.ai.acoes import ACOES_IA, nome_acao
from src.ai.curiosidade_intrinseca import CodificadorVisualICM, ModeloDiretoDinamicas, ModeloInversoDinamicas
from src.ai.modo_treino import ModoTreino, resolver_modo_treino
from src.utils.log import log
from src.visao.processamento_termico import ProcessadorVisaoTermica

CAMINHO_MODELO_IA = Path("data/modelo_pixel_ppo.pt")


def _env_float(nome, padrao):
  valor = os.getenv(nome, "").strip()
  if not valor:
    return float(padrao)
  try:
    return float(valor)
  except ValueError:
    return float(padrao)


def _env_int(nome, padrao):
  valor = os.getenv(nome, "").strip()
  if not valor:
    return int(padrao)
  try:
    return int(valor)
  except ValueError:
    return int(padrao)


class _RedePixelPPO(nn.Module):

  def __init__(self, canais_entrada, total_acoes, tamanho_frame):
    super().__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(canais_entrada, 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1),
      nn.ReLU(),
    )

    with torch.no_grad():
      dummy = torch.zeros(1, canais_entrada, tamanho_frame, tamanho_frame, dtype=torch.float32)
      saida = self.conv(dummy)
      total_flat = int(np.prod(saida.shape[1:]))

    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(total_flat, 512),
      nn.ReLU(),
    )
    # Rede Mestra: 0 = Movimento, 1 = Ofensiva
    self.rede_mestra = nn.Linear(512, 2)
    # Rede de Movimento: 3 acoes (0: ESPERAR, 1: ESQUIVA, 2: BLOQUEIO)
    self.rede_mov = nn.Linear(512, 3)
    # Rede de Ofensiva: 4 acoes (3: LEVE, 4: MEDIO, 5: PESADO, 6: ESPECIAL)
    self.rede_atk = nn.Linear(512, 4)
    self.cabeca_valor = nn.Linear(512, 1)

  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    
    # Obter logits de cada sub-rede
    logits_mestre = self.rede_mestra(x)
    logits_mov = self.rede_mov(x)
    logits_atk = self.rede_atk(x)
    
    # Converter para probabilidades
    probs_mestre = F.softmax(logits_mestre, dim=-1)
    probs_mov = F.softmax(logits_mov, dim=-1)
    probs_atk = F.softmax(logits_atk, dim=-1)
    
    # Probabilidades finais conjuntas
    prob_mov_scaled = probs_mestre[:, 0:1] * probs_mov
    prob_atk_scaled = probs_mestre[:, 1:2] * probs_atk
    
    # Concatena numa única distribuição de 7 ações
    probs_finais = torch.cat([prob_mov_scaled, prob_atk_scaled], dim=-1)
    
    # Retornamos log(probs) que funciona perfeitamente como logits no Categorical
    logits_finais = torch.log(probs_finais + 1e-8)
    
    valor = self.cabeca_valor(x)
    return logits_finais, valor


class CerebroIA:
  FRAME_SIZE = 128
  STACK_SIZE = 4

  def __init__(self, acoes_permitidas=None, modo_treino="treino"):
    self.modo_treino = resolver_modo_treino(modo_treino)
    self.acoes = list(ACOES_IA)
    self.acoes_permitidas = [int(acao) for acao in acoes_permitidas] if acoes_permitidas else list(ACOES_IA)

    pref_device = os.getenv("BOT_PPO_DEVICE", "auto").strip().lower()
    if pref_device in {"cuda", "gpu"} and torch.cuda.is_available():
      self._device = torch.device("cuda")
    elif pref_device == "cpu":
      self._device = torch.device("cpu")
    else:
      self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.gamma = _env_float("BOT_PPO_GAMMA", 0.99)
    self.gae_lambda = _env_float("BOT_PPO_GAE_LAMBDA", 0.95)
    self.clip_eps = _env_float("BOT_PPO_CLIP_EPS", 0.20)
    self.valor_coef = _env_float("BOT_PPO_VALUE_COEF", 0.50)
    self.entropia_coef = _env_float("BOT_PPO_ENTROPY_COEF", 0.01)
    self.lr = _env_float("BOT_PPO_LR", 0.00025)
    self.rollout_size = max(16, _env_int("BOT_PPO_ROLLOUT_SIZE", 256))
    self.update_epochs = max(1, _env_int("BOT_PPO_EPOCHS", 4))
    self.minibatch_size = max(8, _env_int("BOT_PPO_MINIBATCH", 64))
    self.max_grad_norm = _env_float("BOT_PPO_MAX_GRAD_NORM", 0.5)
    self.salvar_a_cada_updates = max(1, _env_int("BOT_PPO_SALVAR_CADA", 5))

    self._modelo = _RedePixelPPO(
      canais_entrada=self.STACK_SIZE,
      total_acoes=len(self.acoes),
      tamanho_frame=self.FRAME_SIZE,
    ).to(self._device)
    self._otimizador = torch.optim.Adam(self._modelo.parameters(), lr=self.lr)

    # Inicialização do Módulo de Curiosidade Intrínseca (ICM)
    self._icm_encoder = CodificadorVisualICM(
        canais_entrada=self.STACK_SIZE,
        tamanho_frame=self.FRAME_SIZE
    ).to(self._device)
    self._icm_inverso = ModeloInversoDinamicas(
        tamanho_latente=256,
        total_acoes=len(self.acoes)
    ).to(self._device)
    self._icm_direto = ModeloDiretoDinamicas(
        tamanho_latente=256,
        total_acoes=len(self.acoes)
    ).to(self._device)
    
    params_icm = list(self._icm_encoder.parameters()) + \
                 list(self._icm_inverso.parameters()) + \
                 list(self._icm_direto.parameters())
    self._otimizador_icm = torch.optim.Adam(params_icm, lr=1e-3)
    self.icm_eta = _env_float("BOT_ICM_ETA", 0.1)

    self._stack_frames = deque(maxlen=self.STACK_SIZE)
    self._ultima_observacao = np.zeros(
      (self.STACK_SIZE, self.FRAME_SIZE, self.FRAME_SIZE),
      dtype=np.float32,
    )
    self._processador_visao = ProcessadorVisaoTermica(frame_size=(self.FRAME_SIZE, self.FRAME_SIZE))
    self.frame_skip = max(1, _env_int("BOT_FRAME_SKIP", 4))
    self._frame_count = 0

    self._pendentes = deque()
    self._rollout_obs = []
    self._rollout_acoes = []
    self._rollout_recompensas = []
    self._rollout_dones = []
    self._rollout_logprobs = []
    self._rollout_valores = []
    self._rollout_next_obs = []
    self._rollout_especial_disponivel = []
    self._rollout_oponente_tem_especial = []

    self._passos_totais = 0
    self._atualizacoes = 0
    self._ultima_acao = self.acoes[0]
    self._ultimo_valor = 0.0
    self._ultima_entropia = 0.0
    self._ultimas_probs = {int(acao): 0.0 for acao in self.acoes}
    self._ultima_policy_loss = 0.0
    self._ultima_value_loss = 0.0
    self._ultima_perda_total = 0.0
    self._ultimo_movimento_score = 0.0
    self._ultimo_frame_diff = np.zeros((self.FRAME_SIZE, self.FRAME_SIZE), dtype=np.uint8)

    self._carregar_memoria()
    atexit.register(self.salvar_memoria)

    log(
      "Cerebro Pixel PPO ativo | "
      f"device={self._device} | frame={self.FRAME_SIZE}x{self.FRAME_SIZE} | "
      f"stack={self.STACK_SIZE} | acoes={len(self.acoes)}"
    )

  @staticmethod
  def _formatar_matriz_int(matriz):
    linhas = []
    for linha in matriz:
      linhas.append("[" + " ".join(f"{int(v):>3d}" for v in linha) + "]")
    return "\n".join(linhas)

  @staticmethod
  def _quantizar_intervalo(valor, max_abs, escala=10):
    if max_abs <= 1e-8:
      return 0
    return int(np.rint((float(valor) / float(max_abs)) * float(escala)))

  def _carregar_memoria(self):
    if not CAMINHO_MODELO_IA.exists():
      return

    try:
      dados = torch.load(CAMINHO_MODELO_IA, map_location=self._device)
      estado_modelo = dados.get("model_state_dict")
      estado_otimizador = dados.get("optimizer_state_dict")
      if estado_modelo:
        self._modelo.load_state_dict(estado_modelo)
      if estado_otimizador:
        self._otimizador.load_state_dict(estado_otimizador)
        
      icm_enc_state = dados.get("icm_enc_state")
      icm_inv_state = dados.get("icm_inv_state")
      icm_dir_state = dados.get("icm_dir_state")
      icm_opt_state = dados.get("icm_opt_state")
      
      if icm_enc_state:
        self._icm_encoder.load_state_dict(icm_enc_state)
      if icm_inv_state:
        self._icm_inverso.load_state_dict(icm_inv_state)
      if icm_dir_state:
        self._icm_direto.load_state_dict(icm_dir_state)
      if icm_opt_state:
        self._otimizador_icm.load_state_dict(icm_opt_state)

      self._atualizacoes = int(dados.get("atualizacoes", 0))
      self._passos_totais = int(dados.get("passos_totais", 0))
      log(
        f"Modelo Pixel PPO carregado: {CAMINHO_MODELO_IA} | "
        f"updates={self._atualizacoes} | passos={self._passos_totais}"
      )
    except Exception:
      log("Falha ao carregar modelo Pixel PPO. Iniciando do zero.")

  def salvar_memoria(self):
    try:
      CAMINHO_MODELO_IA.parent.mkdir(parents=True, exist_ok=True)
      torch.save(
        {
          "model_state_dict": self._modelo.state_dict(),
          "optimizer_state_dict": self._otimizador.state_dict(),
          "icm_enc_state": self._icm_encoder.state_dict(),
          "icm_inv_state": self._icm_inverso.state_dict(),
          "icm_dir_state": self._icm_direto.state_dict(),
          "icm_opt_state": self._otimizador_icm.state_dict(),
          "atualizacoes": int(self._atualizacoes),
          "passos_totais": int(self._passos_totais),
          "acoes": list(self.acoes),
          "frame_size": int(self.FRAME_SIZE),
          "stack_size": int(self.STACK_SIZE),
        },
        CAMINHO_MODELO_IA,
      )
      log(f"Modelo Pixel PPO salvo em: {CAMINHO_MODELO_IA}")
    except OSError:
      log("Falha ao salvar modelo Pixel PPO.")

  def reset_observacao(self):
    self._stack_frames.clear()
    self._processador_visao.reset()
    self._frame_count = 0

  def _preprocessar_frame(self, frame):
    return self._processador_visao.processar_frame(frame)

  def obter_estado(self, frame=None, info_vida=None, info_personagens=None, sinais=None):
    _ = (info_vida, info_personagens, sinais)

    self._frame_count += 1
    if self._frame_count == 1 or (self._frame_count % self.frame_skip == 0):
      quadro_proc = self._preprocessar_frame(frame)
      quadro_anterior = self._stack_frames[-1] if self._stack_frames else None
      if not self._stack_frames:
        for _i in range(self.STACK_SIZE):
          self._stack_frames.append(quadro_proc)
      else:
        self._stack_frames.append(quadro_proc)

      estado = np.stack(self._stack_frames, axis=0).astype(np.float32)
      self._ultima_observacao = estado

      if quadro_anterior is None:
        self._ultimo_movimento_score = 0.0
        self._ultimo_frame_diff = np.zeros((self.FRAME_SIZE, self.FRAME_SIZE), dtype=np.uint8)
      else:
        diff = np.abs(quadro_proc - quadro_anterior)
        self._ultimo_movimento_score = float(np.mean(diff))
        self._ultimo_frame_diff = np.clip(diff * 255.0, 0, 255).astype(np.uint8)

    return self._ultima_observacao

  def _tensor_obs(self, obs):
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(self._device)

  def _indice_acao(self, acao):
    try:
      return self.acoes.index(int(acao))
    except ValueError:
      return 0

  def escolher_acao(self, estado, especial_disponivel=True, oponente_tem_especial=False, oponente_especial_recente=False):
    obs_t = self._tensor_obs(estado)
    with torch.no_grad():
      logits, valor = self._modelo(obs_t)

      # Mascaramento da Ação de Especial (6) se indisponível
      if not especial_disponivel:
        idx_especial = self._indice_acao(6)
        logits = logits.clone()
        logits[0, idx_especial] = -1e9

      # Se o oponente soltou especial recentemente, força defesa/esquiva (bloqueia esperar e ataques)
      if oponente_especial_recente:
        logits = logits.clone()
        for acao_bloqueada in [0, 3, 4, 5, 6]:
          idx_blq = self._indice_acao(acao_bloqueada)
          logits[0, idx_blq] = -1e9

      # Mascaramento das ações não permitidas do modo focado
      logits = logits.clone()
      for acao_ia in self.acoes:
        if acao_ia not in self.acoes_permitidas:
          idx_ia = self._indice_acao(acao_ia)
          logits[0, idx_ia] = -1e9

      probs_torch = F.softmax(logits, dim=-1)
      probs = probs_torch.squeeze(0).detach().cpu().numpy().astype(np.float32)

      # Epsilon-Exploration para quebrar colapso de política durante o treino
      epsilon = _env_float("BOT_EXPLORACAO_EPSILON", 0.12)
      if epsilon > 0.0:
        # Corrige o bug olhando logits em vez de softmax
        permitidas = (logits.squeeze(0) > -1e8).float().cpu().numpy()
        total_permitidas = float(np.sum(permitidas))
        if total_permitidas > 0:
          uniforme = permitidas / total_permitidas
          probs = (1.0 - epsilon) * probs + epsilon * uniforme
          # Garante que somem 1.0 exatamente
          probs = probs / np.sum(probs)

      probs_t = torch.from_numpy(probs).unsqueeze(0).to(self._device)
      dist = Categorical(probs=probs_t)
      indice = int(dist.sample().item())
      logprob = float(dist.log_prob(torch.tensor(indice, device=self._device)).item())
      entropia = float(dist.entropy().item())
      valor_estado = float(valor.squeeze(0).item())

    acao = int(self.acoes[indice])
    self._passos_totais += 1
    self._ultima_acao = acao
    self._ultimo_valor = valor_estado
    self._ultima_entropia = entropia
    self._ultimas_probs = {int(self.acoes[i]): float(probs[i]) for i in range(len(self.acoes))}

    self._pendentes.append(
      {
        "obs": np.asarray(estado, dtype=np.float32).copy(),
        "acao": acao,
        "logprob": logprob,
        "valor": valor_estado,
      }
    )
    return acao

  def inferir_acao_passiva(self, estado, usar_amostragem=True, especial_disponivel=True, oponente_tem_especial=False, oponente_especial_recente=False):
    obs_t = self._tensor_obs(estado)
    with torch.no_grad():
      logits, valor = self._modelo(obs_t)

      # Mascaramento da Ação de Especial (6) se indisponível
      if not especial_disponivel:
        idx_especial = self._indice_acao(6)
        logits = logits.clone()
        logits[0, idx_especial] = -1e9

      # Se o oponente soltou especial recentemente, força defesa/esquiva (bloqueia esperar e ataques)
      if oponente_especial_recente:
        logits = logits.clone()
        for acao_bloqueada in [0, 3, 4, 5, 6]:
          idx_blq = self._indice_acao(acao_bloqueada)
          logits[0, idx_blq] = -1e9

      # Mascaramento das ações não permitidas do modo focado
      logits = logits.clone()
      for acao_ia in self.acoes:
        if acao_ia not in self.acoes_permitidas:
          idx_ia = self._indice_acao(acao_ia)
          logits[0, idx_ia] = -1e9

      dist = Categorical(logits=logits)
      if usar_amostragem:
        indice = int(dist.sample().item())
      else:
        indice = int(torch.argmax(dist.probs, dim=1).item())
      probs = dist.probs.squeeze(0).detach().cpu().numpy().astype(np.float32)
      entropia = float(dist.entropy().item())
      valor_estado = float(valor.squeeze(0).item())

    acao = int(self.acoes[indice])
    self._ultima_acao = acao
    self._ultimo_valor = valor_estado
    self._ultima_entropia = entropia
    self._ultimas_probs = {int(self.acoes[i]): float(probs[i]) for i in range(len(self.acoes))}
    return acao

  def _recompor_pendente(self, estado, acao, especial_disponivel=True, oponente_especial_recente=False):
    obs_t = self._tensor_obs(estado)
    indice = self._indice_acao(acao)
    with torch.no_grad():
      logits, valor = self._modelo(obs_t)

      # Mascaramento da Ação de Especial (6) se indisponível
      if not especial_disponivel:
        idx_especial = self._indice_acao(6)
        logits = logits.clone()
        logits[0, idx_especial] = -1e9

      # Se o oponente soltou especial recentemente, força defesa/esquiva (bloqueia esperar e ataques)
      if oponente_especial_recente:
        logits = logits.clone()
        for acao_bloqueada in [0, 3, 4, 5, 6]:
          idx_blq = self._indice_acao(acao_bloqueada)
          logits[0, idx_blq] = -1e9

      # Mascaramento das ações não permitidas do modo focado
      logits = logits.clone()
      for acao_ia in self.acoes:
        if acao_ia not in self.acoes_permitidas:
          idx_ia = self._indice_acao(acao_ia)
          logits[0, idx_ia] = -1e9

      probs_torch = F.softmax(logits, dim=-1)
      probs = probs_torch.squeeze(0).detach().cpu().numpy().astype(np.float32)

      # Replica a exploração estocástica para manter a consistência matemática dos gradientes PPO
      epsilon = _env_float("BOT_EXPLORACAO_EPSILON", 0.12)
      if epsilon > 0.0:
        # Corrige o bug olhando logits em vez de softmax
        permitidas = (logits.squeeze(0) > -1e8).float().cpu().numpy()
        total_permitidas = float(np.sum(permitidas))
        if total_permitidas > 0:
          uniforme = permitidas / total_permitidas
          probs = (1.0 - epsilon) * probs + epsilon * uniforme
          probs = probs / np.sum(probs)

      probs_t = torch.from_numpy(probs).unsqueeze(0).to(self._device)
      dist = Categorical(probs=probs_t)
      logprob = float(dist.log_prob(torch.tensor(indice, device=self._device)).item())
      valor_estado = float(valor.squeeze(0).item())

    return {
      "obs": np.asarray(estado, dtype=np.float32).copy(),
      "acao": int(acao),
      "logprob": logprob,
      "valor": valor_estado,
    }

  def aprender(
    self,
    estado,
    acao,
    recompensa,
    proximo_estado=None,
    terminal=False,
    especial_disponivel=True,
    oponente_especial_recente=False,
  ):
    if estado is None:
      return

    if self._pendentes:
      registro = self._pendentes.popleft()
      if int(registro["acao"]) != int(acao):
        registro = self._recompor_pendente(
          estado,
          acao,
          especial_disponivel=especial_disponivel,
          oponente_especial_recente=oponente_especial_recente,
        )
    else:
      registro = self._recompor_pendente(
        estado,
        acao,
        especial_disponivel=especial_disponivel,
        oponente_especial_recente=oponente_especial_recente,
      )

    r_int = 0.0
    obs_next_arr = np.zeros_like(estado)
    if proximo_estado is not None:
      obs_next_arr = np.asarray(proximo_estado, dtype=np.float32).copy()
      with torch.no_grad():
        t_st = self._tensor_obs(estado)
        t_st_next = self._tensor_obs(proximo_estado)
        idx_acao = self._indice_acao(acao)
        t_acao = torch.tensor([idx_acao], dtype=torch.long, device=self._device)

        phi_st = self._icm_encoder(t_st)
        phi_st_next = self._icm_encoder(t_st_next)
        hat_phi_st_next = self._icm_direto(phi_st, t_acao)

        erro = F.mse_loss(hat_phi_st_next, phi_st_next).item()
        r_int = (self.icm_eta / 2.0) * erro
        r_int = np.clip(r_int, 0.0, 5.0)
        recompensa += r_int

    self._rollout_obs.append(registro["obs"])
    self._rollout_next_obs.append(obs_next_arr)
    self._rollout_acoes.append(int(acao))
    self._rollout_recompensas.append(float(recompensa))
    self._rollout_dones.append(1.0 if terminal else 0.0)
    self._rollout_logprobs.append(float(registro["logprob"]))
    self._rollout_valores.append(float(registro["valor"]))
    self._rollout_especial_disponivel.append(1.0 if especial_disponivel else 0.0)
    self._rollout_oponente_tem_especial.append(1.0 if oponente_especial_recente else 0.0)

    if len(self._rollout_recompensas) >= self.rollout_size or terminal:
      self._atualizar_modelo(proximo_estado=proximo_estado, terminal=terminal)

  def _atualizar_modelo(self, proximo_estado=None, terminal=False):
    total_passos = len(self._rollout_recompensas)
    if total_passos == 0:
      return

    if terminal or proximo_estado is None:
      valor_bootstrap = 0.0
    else:
      with torch.no_grad():
        _, prox_valor = self._modelo(self._tensor_obs(proximo_estado))
        valor_bootstrap = float(prox_valor.squeeze(0).item())

    recompensas = np.asarray(self._rollout_recompensas, dtype=np.float32)
    dones = np.asarray(self._rollout_dones, dtype=np.float32)
    valores = np.asarray(self._rollout_valores, dtype=np.float32)

    vantagens = np.zeros_like(recompensas, dtype=np.float32)
    retornos = np.zeros_like(recompensas, dtype=np.float32)

    gae = 0.0
    prox_val = valor_bootstrap
    for idx in reversed(range(total_passos)):
      mask = 1.0 - dones[idx]
      delta = recompensas[idx] + (self.gamma * prox_val * mask) - valores[idx]
      gae = delta + (self.gamma * self.gae_lambda * mask * gae)
      vantagens[idx] = gae
      retornos[idx] = vantagens[idx] + valores[idx]
      prox_val = valores[idx]

    vantagens_t = torch.from_numpy(vantagens).to(self._device)
    vantagens_t = (vantagens_t - vantagens_t.mean()) / (vantagens_t.std() + 1e-8)

    obs_t = torch.from_numpy(np.asarray(self._rollout_obs, dtype=np.float32)).to(self._device)
    obs_next_t = torch.from_numpy(np.asarray(self._rollout_next_obs, dtype=np.float32)).to(self._device)
    acoes_idx = torch.tensor(
      [self._indice_acao(acao) for acao in self._rollout_acoes],
      dtype=torch.long,
      device=self._device,
    )
    logprobs_antigos_t = torch.tensor(
      self._rollout_logprobs,
      dtype=torch.float32,
      device=self._device,
    )
    retornos_t = torch.from_numpy(retornos).to(self._device)
    esp_disp_t = torch.tensor(
      self._rollout_especial_disponivel,
      dtype=torch.float32,
      device=self._device,
    )
    oponente_esp_t = torch.tensor(
      self._rollout_oponente_tem_especial,
      dtype=torch.float32,
      device=self._device,
    )

    perdas_policy = []
    perdas_valor = []
    perdas_total = []

    for _epoca in range(self.update_epochs):
      indices = torch.randperm(total_passos, device=self._device)
      for inicio in range(0, total_passos, self.minibatch_size):
        fim = min(inicio + self.minibatch_size, total_passos)
        lote = indices[inicio:fim]

        logits, valores_pred = self._modelo(obs_t[lote])
        valores_pred = valores_pred.squeeze(-1)

        # Aplicar mascaramento no lote de treino
        lote_esp = esp_disp_t[lote]
        lote_oponente_esp = oponente_esp_t[lote]
        logits = logits.clone()

        # Mascara 1: Especial jogador indisponível (6)
        idx_especial = self._indice_acao(6)
        mask_especial = (1.0 - lote_esp).unsqueeze(-1)
        logits[:, idx_especial] = logits[:, idx_especial] * (1.0 - mask_especial.squeeze(-1)) + (mask_especial.squeeze(-1) * -1e9)

        # Mascara 2: Oponente soltou especial recentemente (bloqueia esperar e ataques) - apenas no modo completo
        if self.modo_treino == ModoTreino.COMPLETO:
          mask_oponente = oponente_esp_t[lote].unsqueeze(-1)
          for acao_bloqueada in [0, 3, 4, 5, 6]:
            idx_blq = self._indice_acao(acao_bloqueada)
            logits[:, idx_blq] = logits[:, idx_blq] * (1.0 - mask_oponente.squeeze(-1)) + (mask_oponente.squeeze(-1) * -1e9)

        # Mascara 3: Ações não permitidas do modo focado
        for acao_ia in self.acoes:
          if acao_ia not in self.acoes_permitidas:
            idx_ia = self._indice_acao(acao_ia)
            logits[:, idx_ia] = -1e9

        probs_torch = F.softmax(logits, dim=-1)

        # Aplicar exploração estocástica no lote de treino
        epsilon = _env_float("BOT_EXPLORACAO_EPSILON", 0.12)
        if epsilon > 0.0:
          # Corrige o bug usando os logits para identificar ações permitidas
          permitidas = (logits > -1e8).float()
          total_permitidas = permitidas.sum(dim=-1, keepdim=True)
          uniforme = permitidas / (total_permitidas + 1e-8)
          probs_torch = (1.0 - epsilon) * probs_torch + epsilon * uniforme
          probs_torch = probs_torch / probs_torch.sum(dim=-1, keepdim=True)

        dist = Categorical(probs=probs_torch)

        novos_logprobs = dist.log_prob(acoes_idx[lote])
        razao = torch.exp(novos_logprobs - logprobs_antigos_t[lote])
        surr1 = razao * vantagens_t[lote]
        surr2 = torch.clamp(razao, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * vantagens_t[lote]
        perda_policy = -torch.min(surr1, surr2).mean()
        perda_valor = 0.5 * F.mse_loss(valores_pred, retornos_t[lote])
        entropia = dist.entropy().mean()
        perda_total = (
          perda_policy
          + (self.valor_coef * perda_valor)
          - (self.entropia_coef * entropia)
        )

        self._otimizador.zero_grad()
        perda_total.backward()
        nn.utils.clip_grad_norm_(self._modelo.parameters(), self.max_grad_norm)
        self._otimizador.step()

        # --- Treino do ICM ---
        obs_batch = obs_t[lote]
        obs_next_batch = obs_next_t[lote]
        acoes_batch = acoes_idx[lote]

        phi_st = self._icm_encoder(obs_batch)
        phi_st_next = self._icm_encoder(obs_next_batch)

        # Inverso: Prever acao_t
        pred_acoes = self._icm_inverso(phi_st, phi_st_next)
        perda_inverso = F.cross_entropy(pred_acoes, acoes_batch)

        # Direto: Prever phi(s_{t+1})
        hat_phi_st_next = self._icm_direto(phi_st, acoes_batch)
        perda_direto = F.mse_loss(hat_phi_st_next, phi_st_next.detach())

        # ICM Loss
        beta = 0.2
        perda_icm = (1.0 - beta) * perda_inverso + beta * perda_direto

        self._otimizador_icm.zero_grad()
        perda_icm.backward()
        self._otimizador_icm.step()

        perdas_policy.append(float(perda_policy.item()))
        perdas_valor.append(float(perda_valor.item()))
        perdas_total.append(float(perda_total.item()))

    self._ultima_policy_loss = float(np.mean(perdas_policy)) if perdas_policy else 0.0
    self._ultima_value_loss = float(np.mean(perdas_valor)) if perdas_valor else 0.0
    self._ultima_perda_total = float(np.mean(perdas_total)) if perdas_total else 0.0
    self._atualizacoes += 1

    self._rollout_obs.clear()
    self._rollout_next_obs.clear()
    self._rollout_acoes.clear()
    self._rollout_recompensas.clear()
    self._rollout_dones.clear()
    self._rollout_logprobs.clear()
    self._rollout_valores.clear()
    self._rollout_especial_disponivel.clear()
    self._rollout_oponente_tem_especial.clear()

    if (self._atualizacoes % self.salvar_a_cada_updates) == 0:
      self.salvar_memoria()

  def imprimir_snapshot_filtros(self, origem="debug", max_filtros=3):
    try:
      max_filtros = max(1, int(max_filtros))
    except (TypeError, ValueError):
      max_filtros = 3

    camada_conv1 = self._modelo.conv[0]
    pesos = camada_conv1.weight.detach().cpu().numpy().astype(np.float32)
    total_filtros = int(pesos.shape[0])
    qtd = min(total_filtros, max_filtros)

    log("")
    log("=" * 78)
    log(
      f"[SNAPSHOT_FILTRO] origem={origem} | updates={self._atualizacoes} | "
      f"passos={self._passos_totais} | conv1_shape={tuple(pesos.shape)}"
    )

    # Kernel medio por canal de entrada, quantizado para facilitar leitura no console.
    for idx in range(qtd):
      kernel = np.mean(pesos[idx], axis=0)  # [8, 8]
      max_abs = float(np.max(np.abs(kernel)))
      kernel_q = np.vectorize(self._quantizar_intervalo)(kernel, max_abs, 10).astype(np.int32)
      log(
        f"[SNAPSHOT_FILTRO] conv1_kernel[{idx}] media-canais "
        f"(escala -10..10 | max_abs={max_abs:.6f})"
      )
      log(self._formatar_matriz_int(kernel_q))

    # Mapa de ativacao: mostra onde a rede esta enxergando estrutura no frame atual.
    obs = np.asarray(self._ultima_observacao, dtype=np.float32)
    if obs.size == 0:
      log("[SNAPSHOT_FILTRO] sem observacao para gerar mapa de ativacao.")
      log("=" * 78)
      return

    with torch.no_grad():
      obs_t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
      ativacao_conv1 = camada_conv1(obs_t).detach().cpu().numpy()[0]  # [32, H, W]

    energia = np.mean(np.abs(ativacao_conv1), axis=(1, 2))
    idx_mais_ativo = int(np.argmax(energia))
    mapa = np.abs(ativacao_conv1[idx_mais_ativo]).astype(np.float32)
    mapa16 = cv2.resize(mapa, (16, 16), interpolation=cv2.INTER_AREA)
    max_abs_mapa = float(np.max(np.abs(mapa16)))
    mapa_q = np.vectorize(self._quantizar_intervalo)(mapa16, max_abs_mapa, 10).astype(np.int32)

    log(
      f"[SNAPSHOT_FILTRO] mapa_ativacao filtro={idx_mais_ativo} "
      f"(0..10 | fundo tende a 0, regiao relevante tende a valores altos)"
    )
    log(self._formatar_matriz_int(mapa_q))
    log("=" * 78)
    log("")

  def obter_debug_rede(self):
    return {
      "acao": int(self._ultima_acao),
      "acao_nome": nome_acao(self._ultima_acao),
      "valor": float(self._ultimo_valor),
      "entropia": float(self._ultima_entropia),
      "probs": dict(self._ultimas_probs),
      "policy_loss": float(self._ultima_policy_loss),
      "value_loss": float(self._ultima_value_loss),
      "loss_total": float(self._ultima_perda_total),
      "updates": int(self._atualizacoes),
      "passos": int(self._passos_totais),
      "buffer": int(len(self._rollout_recompensas)),
      "rollout_size": int(self.rollout_size),
      "movimento_score": float(self._ultimo_movimento_score),
      "frame_diff": self._ultimo_frame_diff.copy(),
    }

  def obter_debug_pixel(self):
    return {
      "stack": self._ultima_observacao.copy(),
      "frame_diff": self._ultimo_frame_diff.copy(),
    }
