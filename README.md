# Fight Bot

Bot de luta com visao computacional + IA por reforco em pixel puro.

## Estrutura do projeto

- `assets/`: imagens de referencia para deteccao na tela (botoes, KO, vida etc.)
- `data/`: dados persistidos localmente (checkpoint PPO, logs)
- `src/ai/`: agente `Pixel + CNN + PPO`, recompensa e deteccao de estado de luta
- `src/bot/`: captura da tela, acoes de teclado/mouse e launcher principal
- `src/utils/`: utilitarios de debug

## Setup

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Como rodar

```bash
python -m src.bot.main
```

A `main` abre um menu no terminal para escolher o modo:
- `arena`: fluxo completo de botoes da arena + luta
- `treino`: abre submenu com:
  - `treino_normal`: treino online da IA `Pixel + CNN + PPO`
  - `treino_observacao`: coleta demonstracao humana (sem interferir na gameplay)

Snapshot de filtro da CNN:
- com a janela de debug aberta (`BOT_DEBUG=1`), pressione `P` para imprimir no console:
  - kernels aprendidos da `conv1` (quantizados em `-10..10`)
  - mapa de ativacao (`0..10`) para verificar fundo baixo e regiao relevante alta
- para trocar a tecla: `BOT_DEBUG_SNAPSHOT_KEY=<tecla>`
- para imprimir mais filtros por snapshot: `BOT_DEBUG_SNAPSHOT_FILTROS=6` (exemplo)
- no treino com debug ativo, rodam 3 janelas:
  - `Visao do Bot` (tela original)
  - `Debug Pixel` (stack 4x128x128 + frame diff)
  - `Debug Perceptron` (acao, probs, valor, entropia e losses)

Calibracao ROI de acoes perfeitas:
- no modo `calibracao_roi`, use `N/P` para trocar alvo ate `acoes_perfeitas`
- ajuste o quadrado com `Q/A/W/S/E/D/R/F`
- copie o valor com `C` e salve em `BOT_ROI_ACOES_PERFEITAS`
- essa ROI passa a ser usada para validar `destreza` e `aparar` na recompensa
- alvo adicional para evolucao futura:
  - `sequencia_golpes` (template `assets/sequencia-golpes.png`) -> salvar em `BOT_ROI_COMBO_SEM_PERDA`

Dataset de demonstracoes:
- `data/demonstrations/episode_001.npz`, `episode_002.npz`, ...
- cada episodio salva:
  - `states`, `next_states`, `actions`
  - `action_labels` (inclui `esquiva/destreza`, `bloqueio/aparar`, `ataque_leve/medio/pesado`)
  - `action_base_labels` (acao base antes do resultado perfeito)
  - `rewards`, `rewards_base`, `rewards_bonus`
  - `holds_ms` (duracao da tecla do ataque leve; usado para classificar ataque pesado)
  - `destreza_perfeita`, `aparar_perfeito`, `dones`, `timestamps`

## Convencoes do projeto

- nomes de arquivos em `snake_case`
- nomes de funcoes em `snake_case`
- classes em `PascalCase`
- constantes em `UPPER_SNAKE_CASE`
- um idioma por identificador (preferencia: portugues)
