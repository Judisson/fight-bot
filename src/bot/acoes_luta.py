import os
import queue
import threading

from src.ai.acoes import ACAO_DEFENDER, ACAO_DESTREZA, ACAO_ESPERAR
from src.bot.teclado import pressionar, segurar

TECLA_DESTREZA = os.getenv("BOT_TECLA_DESTREZA", "f").strip() or "f"
TECLA_BLOQUEIO = os.getenv("BOT_TECLA_BLOQUEIO", "space").strip() or "space"

TEMPO_BLOQUEIO_DEFENSIVO = 0.180

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
  if acao == ACAO_DESTREZA:
    destreza()
    return

  if acao == ACAO_DEFENDER:
    defender()
    return

  # FASE 2 (PENDENTE): reativar ACAO_COMBO_SEGURO com janela ofensiva.
  # if acao == ACAO_COMBO_SEGURO:
  #   combo_seguro()
  #   return

  # FASE 3 (PENDENTE): reativar ACAO_ESPECIAL com validacao de barra.
  # if acao == ACAO_ESPECIAL:
  #   especial()
  #   return


def executar_acao(acao):
  if acao == ACAO_ESPERAR:
    return False

  _garantir_worker()
  try:
    _FILA_ACOES.put_nowait(acao)
    return True
  except queue.Full:
    return False


def obter_acao_em_execucao():
  with _LOCK_ESTADO_ACAO:
    return _ACAO_EM_EXECUCAO


def destreza():
  pressionar(TECLA_DESTREZA)


def bloqueio():
  segurar(TECLA_BLOQUEIO, duracao=TEMPO_BLOQUEIO_DEFENSIVO)


def defender():
  bloqueio()


# FASE 2 (PENDENTE):
# def combo_seguro():
#   ataque_medio()
#   time.sleep(0.300)
#   ataque_leve()
#   time.sleep(0.200)
#   ataque_leve()
#   time.sleep(0.200)
#   ataque_leve()
#   time.sleep(0.200)
#   destreza()
#   time.sleep(0.200)
#
# FASE 3 (PENDENTE):
# def especial():
#   pressionar("d")
