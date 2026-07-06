import json
import copy
from pathlib import Path

CAMINHO_CONFIG = Path("data/calibracao_especial.json")

CONFIG_PADRAO = {
  "jogador": {
    "e1": {"x": 0, "y": 0, "color": [0, 0, 0]},
    "e2": {"x": 0, "y": 0, "color": [0, 0, 0]},
    "e3": {"x": 0, "y": 0, "color": [0, 0, 0]},
  },
  "inimigo": {
    "e1": {"x": 0, "y": 0, "color": [0, 0, 0]},
    "e2": {"x": 0, "y": 0, "color": [0, 0, 0]},
    "e3": {"x": 0, "y": 0, "color": [0, 0, 0]},
  },
  "tolerancia": 30,
}

_config_cache = None


def _normalizar_ponto(ponto, padrao_ponto):
  if not isinstance(ponto, dict):
    return dict(padrao_ponto)
  
  normalizado = {}
  try:
    normalizado["x"] = int(ponto.get("x", padrao_ponto["x"]))
  except (TypeError, ValueError):
    normalizado["x"] = padrao_ponto["x"]

  try:
    normalizado["y"] = int(ponto.get("y", padrao_ponto["y"]))
  except (TypeError, ValueError):
    normalizado["y"] = padrao_ponto["y"]

  cor_lida = ponto.get("color")
  if isinstance(cor_lida, list) and len(cor_lida) == 3:
    try:
      normalizado["color"] = [max(0, min(255, int(c))) for c in cor_lida]
    except (TypeError, ValueError):
      normalizado["color"] = list(padrao_ponto["color"])
  else:
    normalizado["color"] = list(padrao_ponto["color"])

  return normalizado


def _normalizar_config(config):
  if not isinstance(config, dict):
    return copy.deepcopy(CONFIG_PADRAO)

  # Migração graciosa do formato antigo (onde e1, e2, e3 ficavam na raiz)
  # para o formato novo (sob "inimigo")
  if "inimigo" not in config and ("e1" in config or "e2" in config or "e3" in config):
    config = dict(config)  # Evita mutar se não for dict próprio
    config["inimigo"] = {
      "e1": config.get("e1"),
      "e2": config.get("e2"),
      "e3": config.get("e3"),
    }

  normalizado = {}
  
  for grupo in ["jogador", "inimigo"]:
    normalizado[grupo] = {}
    grupo_lido = config.get(grupo)
    if not isinstance(grupo_lido, dict):
      grupo_lido = {}
    for chave in ["e1", "e2", "e3"]:
      normalizado[grupo][chave] = _normalizar_ponto(
        grupo_lido.get(chave),
        CONFIG_PADRAO[grupo][chave],
      )

  try:
    normalizado["tolerancia"] = int(config.get("tolerancia", CONFIG_PADRAO["tolerancia"]))
  except (TypeError, ValueError):
    normalizado["tolerancia"] = CONFIG_PADRAO["tolerancia"]

  return normalizado


def carregar_configuracao_especial(forcar_reload=False):
  global _config_cache

  if _config_cache is not None and not forcar_reload:
    return copy.deepcopy(_config_cache)

  if not CAMINHO_CONFIG.exists():
    _config_cache = copy.deepcopy(CONFIG_PADRAO)
    return copy.deepcopy(_config_cache)

  try:
    conteudo = json.loads(CAMINHO_CONFIG.read_text(encoding="utf-8"))
    _config_cache = _normalizar_config(conteudo)
  except (json.JSONDecodeError, OSError):
    _config_cache = copy.deepcopy(CONFIG_PADRAO)

  return copy.deepcopy(_config_cache)


def definir_configuracao_especial(config):
  global _config_cache
  _config_cache = _normalizar_config(config)


def salvar_configuracao_especial(config):
  normalizado = _normalizar_config(config)
  definir_configuracao_especial(normalizado)
  CAMINHO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
  CAMINHO_CONFIG.write_text(
    json.dumps(normalizado, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
