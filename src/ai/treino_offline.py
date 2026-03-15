import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ai.cerebro import CerebroIA
from src.utils.log import log

PASTA_DEMONSTRACOES = Path("data/demonstrations")


def _listar_episodios():
  if not PASTA_DEMONSTRACOES.exists():
    return []
  return sorted(PASTA_DEMONSTRACOES.glob("episode_*.npz"))


def _carregar_dataset(cerebro, max_amostras=0):
  arquivos = _listar_episodios()
  if not arquivos:
    return None, None, 0

  estados = []
  acoes_idx = []
  total_episodios = 0
  total_bruto = 0

  for arquivo in arquivos:
    try:
      dados = np.load(arquivo)
    except Exception:
      log(f"[OFFLINE] Falha ao abrir: {arquivo}")
      continue

    states = dados.get("states")
    actions = dados.get("actions")
    if states is None or actions is None:
      continue

    total_episodios += 1
    total_bruto += int(actions.shape[0])
    for i in range(int(actions.shape[0])):
      acao = int(actions[i])
      if acao not in cerebro.acoes:
        continue
      estados.append(states[i].astype(np.float32))
      acoes_idx.append(cerebro.acoes.index(acao))

      if max_amostras > 0 and len(estados) >= max_amostras:
        break
    if max_amostras > 0 and len(estados) >= max_amostras:
      break

  if not estados:
    return None, None, total_episodios

  x = np.asarray(estados, dtype=np.float32)
  y = np.asarray(acoes_idx, dtype=np.int64)
  log(
    f"[OFFLINE] Episodios lidos={total_episodios} | bruto={total_bruto} | "
    f"validos={x.shape[0]}"
  )
  return x, y, total_episodios


def iniciar_treino_offline():
  cerebro = CerebroIA(modo_treino="treino")
  try:
    epochs = max(1, int(os.getenv("BOT_OFFLINE_EPOCHS", "5")))
  except ValueError:
    epochs = 5
  try:
    batch_size = max(16, int(os.getenv("BOT_OFFLINE_BATCH", "128")))
  except ValueError:
    batch_size = 128
  try:
    max_amostras = max(0, int(os.getenv("BOT_OFFLINE_MAX_SAMPLES", "0")))
  except ValueError:
    max_amostras = 0
  try:
    lr = float(os.getenv("BOT_OFFLINE_LR", "0.0001"))
  except ValueError:
    lr = 0.0001

  x_np, y_np, total_episodios = _carregar_dataset(cerebro, max_amostras=max_amostras)
  if x_np is None or y_np is None:
    if total_episodios == 0:
      log("[OFFLINE] Nenhum episodio encontrado em data/demonstrations.")
    else:
      log("[OFFLINE] Episodios encontrados, mas sem acoes validas para treino.")
    return

  x = torch.from_numpy(x_np).to(cerebro._device)
  y = torch.from_numpy(y_np).to(cerebro._device)

  dataset = TensorDataset(x, y)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
  criterio = nn.CrossEntropyLoss()
  otimizador = torch.optim.Adam(cerebro._modelo.parameters(), lr=lr)

  cerebro._modelo.train()
  log(
    f"[OFFLINE] Iniciando imitacao supervisionada | amostras={len(dataset)} | "
    f"epochs={epochs} | batch={batch_size} | lr={lr}"
  )

  for epoca in range(1, epochs + 1):
    perda_total = 0.0
    acertos = 0
    total = 0
    for xb, yb in loader:
      logits, _valor = cerebro._modelo(xb)
      loss = criterio(logits, yb)

      otimizador.zero_grad()
      loss.backward()
      otimizador.step()

      perda_total += float(loss.item()) * int(yb.shape[0])
      preds = torch.argmax(logits, dim=1)
      acertos += int((preds == yb).sum().item())
      total += int(yb.shape[0])

    perda_media = perda_total / max(1, total)
    acc = acertos / max(1, total)
    log(f"[OFFLINE] epoca={epoca}/{epochs} | loss={perda_media:.5f} | acc={acc:.3f}")

  cerebro._modelo.eval()
  cerebro.salvar_memoria()
  log("[OFFLINE] Treino offline concluido e checkpoint salvo.")
