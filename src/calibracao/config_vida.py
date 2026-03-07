import json
from pathlib import Path

CAMINHO_CONFIG = Path("data/calibracao_vida.json")

CONFIG_PADRAO = {
  "y_inicio": 100,
  "y_fim": 110,
  "x1_jogador": 200,
  "x2_jogador": 358,
  "offset_direita_inimigo": 120,
  "largura_inimigo": 360,
}

_config_cache = None


def _normalizar_config(config):
  normalizado = dict(CONFIG_PADRAO)

  for chave in CONFIG_PADRAO:
    if chave in config:
      try:
        normalizado[chave] = int(config[chave])
      except (TypeError, ValueError):
        pass

  return normalizado


def carregar_configuracao_vida(forcar_reload=False):
  global _config_cache

  if _config_cache is not None and not forcar_reload:
    return dict(_config_cache)

  if not CAMINHO_CONFIG.exists():
    _config_cache = dict(CONFIG_PADRAO)
    return dict(_config_cache)

  try:
    conteudo = json.loads(CAMINHO_CONFIG.read_text(encoding="utf-8"))
    _config_cache = _normalizar_config(conteudo)
  except (json.JSONDecodeError, OSError):
    _config_cache = dict(CONFIG_PADRAO)

  return dict(_config_cache)


def definir_configuracao_vida(config):
  global _config_cache
  _config_cache = _normalizar_config(config)


def salvar_configuracao_vida(config):
  normalizado = _normalizar_config(config)
  definir_configuracao_vida(normalizado)
  CAMINHO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
  CAMINHO_CONFIG.write_text(
    json.dumps(normalizado, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
