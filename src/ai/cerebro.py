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
from src.ai.modo_treino import resolver_modo_treino
from src.utils.log import log

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
    self.cabeca_politica = nn.Linear(512, total_acoes)
    self.cabeca_valor = nn.Linear(512, 1)

  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    logits = self.cabeca_politica(x)
    valor = self.cabeca_valor(x)
    return logits, valor


class CerebroIA:
  FRAME_SIZE = 128
  STACK_SIZE = 4

  def __init__(self, acoes_permitidas=None, modo_treino="treino"):
    self.modo_treino = resolver_modo_treino(modo_treino)
    if acoes_permitidas:
      self.acoes = [int(acao) for acao in acoes_permitidas]
    else:
      self.acoes = list(ACOES_IA)

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

    self._stack_frames = deque(maxlen=self.STACK_SIZE)
    self._ultima_observacao = np.zeros(
      (self.STACK_SIZE, self.FRAME_SIZE, self.FRAME_SIZE),
      dtype=np.float32,
    )

    self._pendentes = deque()
    self._rollout_obs = []
    self._rollout_acoes = []
    self._rollout_recompensas = []
    self._rollout_dones = []
    self._rollout_logprobs = []
    self._rollout_valores = []

    self._passos_totais = 0
    self._atualizacoes = 0
    self._ultima_acao = self.acoes[0]
    self._ultimo_valor = 0.0
    self._ultima_entropia = 0.0
    self._ultimas_probs = {int(acao): 0.0 for acao in self.acoes}
    self._ultima_policy_loss = 0.0
    self._ultima_value_loss = 0.0
    self._ultima_perda_total = 0.0

    self._carregar_memoria()
    atexit.register(self.salvar_memoria)

    log(
      "Cerebro Pixel PPO ativo | "
      f"device={self._device} | frame={self.FRAME_SIZE}x{self.FRAME_SIZE} | "
      f"stack={self.STACK_SIZE} | acoes={len(self.acoes)}"
    )

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

  def _preprocessar_frame(self, frame):
    if frame is None:
      return np.zeros((self.FRAME_SIZE, self.FRAME_SIZE), dtype=np.float32)

    if len(frame.shape) == 3:
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
      frame_gray = frame

    frame_red = cv2.resize(
      frame_gray,
      (self.FRAME_SIZE, self.FRAME_SIZE),
      interpolation=cv2.INTER_AREA,
    )
    return frame_red.astype(np.float32) / 255.0

  def obter_estado(self, frame=None, info_vida=None, info_personagens=None, sinais=None):
    _ = (info_vida, info_personagens, sinais)

    quadro_proc = self._preprocessar_frame(frame)
    if not self._stack_frames:
      for _i in range(self.STACK_SIZE):
        self._stack_frames.append(quadro_proc)
    else:
      self._stack_frames.append(quadro_proc)

    estado = np.stack(self._stack_frames, axis=0).astype(np.float32)
    self._ultima_observacao = estado
    return estado

  def _tensor_obs(self, obs):
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(self._device)

  def _indice_acao(self, acao):
    try:
      return self.acoes.index(int(acao))
    except ValueError:
      return 0

  def escolher_acao(self, estado):
    obs_t = self._tensor_obs(estado)
    with torch.no_grad():
      logits, valor = self._modelo(obs_t)
      dist = Categorical(logits=logits)
      indice = int(dist.sample().item())
      logprob = float(dist.log_prob(torch.tensor(indice, device=self._device)).item())
      probs = dist.probs.squeeze(0).detach().cpu().numpy().astype(np.float32)
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

  def _recompor_pendente(self, estado, acao):
    obs_t = self._tensor_obs(estado)
    indice = self._indice_acao(acao)
    with torch.no_grad():
      logits, valor = self._modelo(obs_t)
      dist = Categorical(logits=logits)
      logprob = float(dist.log_prob(torch.tensor(indice, device=self._device)).item())
      valor_estado = float(valor.squeeze(0).item())

    return {
      "obs": np.asarray(estado, dtype=np.float32).copy(),
      "acao": int(acao),
      "logprob": logprob,
      "valor": valor_estado,
    }

  def aprender(self, estado, acao, recompensa, proximo_estado=None, terminal=False):
    if estado is None:
      return

    if self._pendentes:
      registro = self._pendentes.popleft()
      if int(registro["acao"]) != int(acao):
        registro = self._recompor_pendente(estado, acao)
    else:
      registro = self._recompor_pendente(estado, acao)

    self._rollout_obs.append(registro["obs"])
    self._rollout_acoes.append(int(acao))
    self._rollout_recompensas.append(float(recompensa))
    self._rollout_dones.append(1.0 if terminal else 0.0)
    self._rollout_logprobs.append(float(registro["logprob"]))
    self._rollout_valores.append(float(registro["valor"]))

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
        dist = Categorical(logits=logits)

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

        perdas_policy.append(float(perda_policy.item()))
        perdas_valor.append(float(perda_valor.item()))
        perdas_total.append(float(perda_total.item()))

    self._ultima_policy_loss = float(np.mean(perdas_policy)) if perdas_policy else 0.0
    self._ultima_value_loss = float(np.mean(perdas_valor)) if perdas_valor else 0.0
    self._ultima_perda_total = float(np.mean(perdas_total)) if perdas_total else 0.0
    self._atualizacoes += 1

    self._rollout_obs.clear()
    self._rollout_acoes.clear()
    self._rollout_recompensas.clear()
    self._rollout_dones.clear()
    self._rollout_logprobs.clear()
    self._rollout_valores.clear()

    if (self._atualizacoes % self.salvar_a_cada_updates) == 0:
      self.salvar_memoria()

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
    }
