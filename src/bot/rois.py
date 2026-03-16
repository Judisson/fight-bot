import os

from src.utils.log import log

ROI_JOGAR_NOVAMENTE = "jogar_novamente"
ROI_ACOES_PERFEITAS = "acoes_perfeitas"
ROI_SEQUENCIA_GOLPES = "sequencia_golpes"
# Alias legado para manter compatibilidade com chamadas antigas.
ROI_COMBO_SEM_PERDA = ROI_SEQUENCIA_GOLPES

# Registro central de ROIs fixos do projeto.
# Adicione novos ROIs aqui conforme forem calibrados.
ROIS_PADRAO = {
  ROI_JOGAR_NOVAMENTE: (960, 950, 1215, 1040),
  # ROI quadrado para deteccao de feedback visual de destreza/aparar.
  ROI_ACOES_PERFEITAS: (130, 480, 430, 850),
  # ROI quadrado para leitura futura do estado do combo (sem perder golpes).
  ROI_SEQUENCIA_GOLPES: (260, 340, 510, 430),
}

ROIS_ENV = {
  ROI_JOGAR_NOVAMENTE: "BOT_ROI_JOGAR_NOVAMENTE",
  ROI_ACOES_PERFEITAS: "BOT_ROI_ACOES_PERFEITAS",
  ROI_SEQUENCIA_GOLPES: "BOT_ROI_COMBO_SEM_PERDA",
}

ROI_TEMPLATES = {
  ROI_JOGAR_NOVAMENTE: "assets/jogar-novamente-button.png",
  ROI_ACOES_PERFEITAS: (
    "assets/destreza.png",
    "assets/aparar.png",
  ),
  ROI_SEQUENCIA_GOLPES: "assets/sequencia-golpes.png",
}


def obter_env_roi(nome_roi):
  return ROIS_ENV.get(nome_roi, f"BOT_ROI_{str(nome_roi).upper()}")


def _parse_roi(valor):
  partes = [p.strip() for p in str(valor).split(",") if p.strip()]
  if len(partes) != 4:
    return None
  try:
    x1, y1, x2, y2 = [int(p) for p in partes]
  except ValueError:
    return None
  if x2 <= x1 or y2 <= y1:
    return None
  return (x1, y1, x2, y2)


def obter_roi(nome_roi):
  if nome_roi not in ROIS_PADRAO:
    raise KeyError(f"ROI desconhecido: {nome_roi}")

  env_nome = obter_env_roi(nome_roi)
  valor = os.getenv(env_nome, "").strip()
  if not valor:
    return ROIS_PADRAO[nome_roi]

  roi = _parse_roi(valor)
  if roi is None:
    log(f"{env_nome} invalido. Usando ROI padrao de {nome_roi}.")
    return ROIS_PADRAO[nome_roi]

  return roi


def obter_template_roi(nome_roi):
  return ROI_TEMPLATES.get(nome_roi)
