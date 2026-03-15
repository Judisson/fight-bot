import json
import os
import time
from pathlib import Path

from src.bot.acoes import clicar
from src.bot.controle_fps import ControleFPS, obter_fps_alvo
from src.bot.captura import CapturaAssincrona
from src.bot.foco import focar_jogo
from src.bot.janela_jogo import encontrar_janela, obter_bbox_janela
from src.bot.luta import Luta
from src.bot.rois import ROI_JOGAR_NOVAMENTE, obter_roi
from src.bot.visao import encontrar_templates_em_paralelo
from src.utils.log import log, logs_ativos
from src.utils.visao_debug import atualizar_metricas

TITULO_JOGO = "Champions"
CAMINHO_LOG_FPS_TREINO = Path("data/fps_treino.jsonl")
BOTOES_TREINO = [
  ("jogar_novamente", "assets/jogar-novamente-button.png"),
]


def obter_monitor_jogo():
  log("Procurando janela do jogo...")
  janela = None

  while janela is None:
    janela = encontrar_janela(TITULO_JOGO)
    time.sleep(1)

  log("Janela encontrada!")

  if focar_jogo(TITULO_JOGO):
    log("Janela do jogo focada automaticamente.")
    time.sleep(0.5)  # MUITO IMPORTANTE
  else:
    log("Nao foi possivel focar automaticamente a janela.")

  return obter_bbox_janela(janela)


def processar_layout_treino(frame, monitor, roi_jogar_novamente):
  consultas = []
  for nome, caminho_template in BOTOES_TREINO:
    if nome == "jogar_novamente":
      consultas.append((nome, caminho_template, 0.8, roi_jogar_novamente))
    else:
      consultas.append((nome, caminho_template))

  resultados = encontrar_templates_em_paralelo(
    frame,
    consultas,
  )

  for nome_botao, _ in BOTOES_TREINO:
    botao = resultados.get(nome_botao)
    if not botao:
      continue

    log(f"Botao ({nome_botao}) encontrado no treino.")
    clicar(
      monitor["left"] + botao["x"],
      monitor["top"] + botao["y"],
    )
    time.sleep(2)
    return True

  return False


def iniciar_treino(monitor=None, modo_treino="destreza"):
  if monitor is None:
    monitor = obter_monitor_jogo()

  fps_alvo = obter_fps_alvo()
  controle_fps = ControleFPS(fps_alvo)
  log(f"Treino iniciado com FPS alvo: {fps_alvo} | modo={modo_treino}")

  try:
    intervalo_log_fps = max(0.5, float(os.getenv("BOT_LOG_FPS_INTERVALO", "2.0")))
  except ValueError:
    intervalo_log_fps = 2.0
  logs_habilitados = logs_ativos()
  salvar_log_fps = logs_habilitados and (os.getenv("BOT_SALVAR_LOG_FPS", "1").strip() == "1")
  ultimo_log_fps_ts = 0.0

  exibir_debug = os.getenv("BOT_DEBUG", "1").strip() == "1"
  roi_jogar_novamente = obter_roi(ROI_JOGAR_NOVAMENTE)
  log(f"ROI jogar_novamente (x1,y1,x2,y2): {roi_jogar_novamente}")
  luta = Luta(exibir_debug=exibir_debug, modo_treino=modo_treino)
  capturador = CapturaAssincrona(monitor)
  capturador.iniciar()
  modo_reinicio = False

  try:
    while True:
      inicio_ciclo = controle_fps.iniciar_ciclo()

      try:
        frame = capturador.obter_frame(timeout=0.05)
        if frame is None:
          continue

        if modo_reinicio:
          if processar_layout_treino(frame, monitor, roi_jogar_novamente):
            modo_reinicio = False
          continue

        status_luta = luta.processar_frame(frame)
        if status_luta and status_luta.get("terminal") is not None:
          modo_reinicio = True
          if logs_habilitados:
            log(
              f"Entrando em modo reinicio apos terminal={status_luta.get('terminal')}. "
              "YOLO/estado de luta pausados ate detectar jogar novamente."
            )
      finally:
        controle_fps.finalizar_ciclo(inicio_ciclo)
        metricas = controle_fps.obter_metricas()
        atualizar_metricas(metricas)

        agora = time.perf_counter()
        if (
          metricas.get("fps_real", 0.0) > 0.0
          and (agora - ultimo_log_fps_ts) >= intervalo_log_fps
        ):
          ultimo_log_fps_ts = agora
          if logs_habilitados:
            log(
              "[TREINO_FPS]",
              f"real={metricas['fps_real']:.1f}",
              f"alvo={metricas['fps_alvo']}",
              f"proc_ms={metricas['proc_ms']:.2f}",
              f"sleep_ms={metricas['sleep_ms']:.2f}",
            )

          if salvar_log_fps:
            registro = {
              "ts": round(time.time(), 3),
              "fps_real": round(float(metricas["fps_real"]), 3),
              "fps_alvo": int(metricas["fps_alvo"]),
              "proc_ms": round(float(metricas["proc_ms"]), 3),
              "sleep_ms": round(float(metricas["sleep_ms"]), 3),
            }
            try:
              CAMINHO_LOG_FPS_TREINO.parent.mkdir(parents=True, exist_ok=True)
              with CAMINHO_LOG_FPS_TREINO.open("a", encoding="utf-8") as arquivo:
                arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
            except OSError:
              log("Falha ao registrar fps_treino.jsonl.")
  finally:
    capturador.parar()
