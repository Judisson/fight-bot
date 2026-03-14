from src.calibracao.config_poder import carregar_configuracao_poder


def _clamp_bbox(bbox, largura, altura):
  x1, y1, x2, y2 = bbox
  x1 = max(0, min(x1, largura - 2))
  x2 = max(x1 + 1, min(x2, largura - 1))
  y1 = max(0, min(y1, altura - 2))
  y2 = max(y1 + 1, min(y2, altura - 1))
  return (x1, y1, x2, y2)


def obter_bboxes_poder(frame):
  altura, largura, _ = frame.shape

  config = carregar_configuracao_poder()

  y_inicio = max(0, config["y_inicio"])
  y_fim = max(y_inicio + 1, config["y_fim"])

  x1_jogador = max(0, config["x1_jogador"])
  x2_jogador = max(x1_jogador + 1, config["x2_jogador"])

  offset_direita_inimigo = max(0, config["offset_direita_inimigo"])
  largura_inimigo = max(1, config["largura_inimigo"])

  bbox_jogador = _clamp_bbox((x1_jogador, y_inicio, x2_jogador, y_fim), largura, altura)

  x2_inimigo = largura - offset_direita_inimigo
  x1_inimigo = x2_inimigo - largura_inimigo
  bbox_inimigo = _clamp_bbox((x1_inimigo, y_inicio, x2_inimigo, y_fim), largura, altura)

  return bbox_jogador, bbox_inimigo


def extrair_barras_poder(frame):
  bbox_jogador, bbox_inimigo = obter_bboxes_poder(frame)

  x1, y1, x2, y2 = bbox_jogador
  poder_jogador = frame[y1:y2, x1:x2]

  x1, y1, x2, y2 = bbox_inimigo
  poder_inimigo = frame[y1:y2, x1:x2]

  return poder_jogador, poder_inimigo
