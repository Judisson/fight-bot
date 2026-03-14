import os
import queue
import threading
import time

from src.ai.acoes import (
  ACAO_BLOQUEIO,
  ACAO_COMBO_SEGURO,
  ACAO_DEFENSIVO,
  ACAO_DESTREZA,
  ACAO_ESPECIAL,
  ACAO_NADA,
)
from src.bot.teclado import pressionar, segurar

TECLA_ATAQUE_LEVE = os.getenv("BOT_TECLA_ATAQUE_LEVE", "j").strip() or "j"
TECLA_ATAQUE_MEDIO = os.getenv("BOT_TECLA_ATAQUE_MEDIO", "k").strip() or "k"
TECLA_DESTREZA = os.getenv("BOT_TECLA_DESTREZA", "f").strip() or "f"
TECLA_ESPECIAL = os.getenv("BOT_TECLA_ESPECIAL", "d").strip() or "d"
TECLA_BLOQUEIO = os.getenv("BOT_TECLA_BLOQUEIO", "space").strip() or "space"

TEMPO_APOS_MEDIO = 0.300
TEMPO_ENTRE_LEVES = 0.200
TEMPO_APOS_DESTREZA = 0.200
TEMPO_BLOQUEIO_DEFENSIVO = 0.180

_FILA_ACOES = queue.Queue(maxsize=1)
_WORKER_INICIADO = False
_LOCK_WORKER = threading.Lock()
_LOCK_ESTADO_ACAO = threading.Lock()
_ACAO_EM_EXECUCAO = ACAO_NADA


def _garantir_worker():
  global _WORKER_INICIADO
  if _WORKER_INICIADO:
    return

  with _LOCK_WORKER:
    if _WORKER_INICIADO:
      return

    thread = threading.Thread(target=_worker_acoes, daemon=True, name="acao-luta-worker")
    thread.start()
    _WORKER_INICIADO = True


def _worker_acoes():
  global _ACAO_EM_EXECUCAO
  while True:
    item = _FILA_ACOES.get()
    if isinstance(item, tuple):
      acao, pode_soltar_especial = item
    else:
      acao, pode_soltar_especial = item, False
    try:
      with _LOCK_ESTADO_ACAO:
        _ACAO_EM_EXECUCAO = acao
      _executar_acao_sincrona(acao, pode_soltar_especial=pode_soltar_especial)
    finally:
      with _LOCK_ESTADO_ACAO:
        _ACAO_EM_EXECUCAO = ACAO_NADA
      _FILA_ACOES.task_done()


def _executar_acao_sincrona(acao, pode_soltar_especial=False):
  if acao == ACAO_DESTREZA:
    destreza()
    return

  if acao == ACAO_ESPECIAL:
    especial()
    return

  if acao == ACAO_BLOQUEIO:
    bloqueio()
    return

  if acao == ACAO_COMBO_SEGURO:
    combo_seguro(pode_soltar_especial=pode_soltar_especial)
    return

  if acao == ACAO_DEFENSIVO:
    defensivo()


def executar_acao(acao, pode_soltar_especial=False):
  if acao == ACAO_NADA:
    return False

  _garantir_worker()
  try:
    _FILA_ACOES.put_nowait((acao, bool(pode_soltar_especial)))
    return True
  except queue.Full:
    return False


def obter_acao_em_execucao():
  with _LOCK_ESTADO_ACAO:
    return _ACAO_EM_EXECUCAO


def ataque_leve():
  pressionar(TECLA_ATAQUE_LEVE)


def ataque_medio():
  pressionar(TECLA_ATAQUE_MEDIO)


def destreza():
  pressionar(TECLA_DESTREZA)


def especial():
  pressionar(TECLA_ESPECIAL)


def bloqueio():
  segurar(TECLA_BLOQUEIO, duracao=TEMPO_BLOQUEIO_DEFENSIVO)


def combo_seguro(pode_soltar_especial=False):
  ataque_medio()
  time.sleep(TEMPO_APOS_MEDIO)

  ataque_leve()
  time.sleep(TEMPO_ENTRE_LEVES)

  ataque_leve()
  time.sleep(TEMPO_ENTRE_LEVES)

  ataque_leve()
  time.sleep(TEMPO_ENTRE_LEVES)

  if pode_soltar_especial:
    especial()
  else:
    destreza()
  time.sleep(TEMPO_APOS_DESTREZA)


def defensivo():
  bloqueio()
  destreza()
  time.sleep(TEMPO_APOS_DESTREZA)
