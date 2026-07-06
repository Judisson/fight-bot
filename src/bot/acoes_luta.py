import os
import queue
import threading

from src.ai.acoes import (
  ACAO_ATAQUE_LEVE,
  ACAO_ATAQUE_MEDIO,
  ACAO_ATAQUE_PESADO,
  ACAO_BLOQUEIO,
  ACAO_ESQUIVA,
  ACAO_ESPERAR,
  ACAO_ESPECIAL,
)
from src.bot.teclado import pressionar, segurar


TECLA_ESQUIVA = os.getenv("BOT_TECLA_ESQUIVA", "f").strip() or "f"
TECLA_BLOQUEIO = os.getenv("BOT_TECLA_BLOQUEIO", "d").strip() or "d"
TECLA_ESPECIAL = os.getenv("BOT_TECLA_ESPECIAL", "s").strip() or "s"
TECLA_ATAQUE_LEVE = os.getenv("BOT_TECLA_ATAQUE_LEVE", "j").strip() or "j"
TECLA_ATAQUE_MEDIO = os.getenv("BOT_TECLA_ATAQUE_MEDIO", "k").strip() or "k"

TEMPO_BLOQUEIO_DEFENSIVO = 0.180
try:
  TEMPO_ATAQUE_PESADO = float(os.getenv("BOT_TEMPO_ATAQUE_PESADO", "0.25"))
except ValueError:
  TEMPO_ATAQUE_PESADO = 0.25

_FILA_ACOES = queue.Queue(maxsize=1)
_WORKER_INICIADO = False
_LOCK_WORKER = threading.Lock()
_LOCK_ESTADO_ACAO = threading.Lock()
_ACAO_EM_EXECUCAO = ACAO_ESPERAR


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
    acao = _FILA_ACOES.get()
    try:
      with _LOCK_ESTADO_ACAO:
        _ACAO_EM_EXECUCAO = acao
      _executar_acao_sincrona(acao)
    finally:
      with _LOCK_ESTADO_ACAO:
        _ACAO_EM_EXECUCAO = ACAO_ESPERAR
      _FILA_ACOES.task_done()


def _executar_acao_sincrona(acao):
  if acao == ACAO_ESQUIVA:
    esquiva()
    return

  if acao == ACAO_BLOQUEIO:
    bloqueio()
    return

  if acao == ACAO_ATAQUE_LEVE:
    ataque_leve()
    return

  if acao == ACAO_ATAQUE_MEDIO:
    ataque_medio()
    return

  if acao == ACAO_ATAQUE_PESADO:
    ataque_pesado()
    return

  if acao == ACAO_ESPECIAL:
    especial()
    return


def executar_acao(acao):
  if acao == ACAO_ESPERAR:
    return False

  _garantir_worker()
  try:
    _FILA_ACOES.put_nowait(acao)
    return True
  except queue.Full:
    return False
def limpar_fila_acoes():
  try:
    while not _FILA_ACOES.empty():
      _FILA_ACOES.get_nowait()
      _FILA_ACOES.task_done()
  except Exception:
    pass


def obter_acao_em_execucao():
  with _LOCK_ESTADO_ACAO:
    return _ACAO_EM_EXECUCAO


def esquiva():
  pressionar(TECLA_ESQUIVA)


def bloqueio():
  segurar(TECLA_BLOQUEIO, duracao=TEMPO_BLOQUEIO_DEFENSIVO)


def ataque_leve():
  pressionar(TECLA_ATAQUE_LEVE)


def ataque_medio():
  pressionar(TECLA_ATAQUE_MEDIO)


def ataque_pesado():
  segurar(TECLA_ATAQUE_LEVE, duracao=TEMPO_ATAQUE_PESADO)


def especial():
  pressionar(TECLA_ESPECIAL)
