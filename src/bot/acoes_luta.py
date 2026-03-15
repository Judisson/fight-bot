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
)
from src.bot.teclado import pressionar, segurar


def _env_tecla(nome_novo, nome_legado, padrao):
  valor = os.getenv(nome_novo, "").strip()
  if valor:
    return valor
  valor_legado = os.getenv(nome_legado, "").strip()
  if valor_legado:
    return valor_legado
  return padrao


TECLA_ESQUIVA = _env_tecla("BOT_TECLA_ESQUIVA", "BOT_TECLA_DESTREZA", "f")
TECLA_BLOQUEIO = os.getenv("BOT_TECLA_BLOQUEIO", "space").strip() or "space"
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


def esquiva():
  pressionar(TECLA_ESQUIVA)


def destreza():
  # Alias legado para a mesma tecla de esquiva.
  esquiva()


def bloqueio():
  segurar(TECLA_BLOQUEIO, duracao=TEMPO_BLOQUEIO_DEFENSIVO)


def defender():
  # Alias legado de bloqueio.
  bloqueio()


def ataque_leve():
  pressionar(TECLA_ATAQUE_LEVE)


def ataque_medio():
  pressionar(TECLA_ATAQUE_MEDIO)


def ataque_pesado():
  segurar(TECLA_ATAQUE_LEVE, duracao=TEMPO_ATAQUE_PESADO)


# FASE 3 (PENDENTE):
# def especial():
#   pressionar("d")
