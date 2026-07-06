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

## Calibração

O bot depende de calibrações de tela para identificar corretamente os elementos do jogo (barras de vida, poder, especiais e ROIs). Ambas as calibrações podem ser acessadas pelo menu principal:
```bash
python -m src.bot.main
```

### 1. Calibração de Vida e Especial (Modo `calibracao`)
Selecione a opção **`calibracao`** no menu principal do terminal. Uma janela OpenCV com a imagem do jogo será exibida.

**Controles na Janela:**
* **Seleção de Alvo:**
  - Pressione `1`: Focar na calibração de **VIDA** (exibe caixas amarela/verde).
  - Pressione `4`: Selecionar o pixel do **Especial 1 (E1)** do jogador para calibração.
  - Pressione `5`: Selecionar o pixel do **Especial 2 (E2)** do jogador para calibração.
  - Pressione `6`: Selecionar o pixel do **Especial 3 (E3)** do jogador para calibração.
  - Pressione `7`: Selecionar o pixel do **Especial 1 (E1)** do inimigo para calibração.
  - Pressione `8`: Selecionar o pixel do **Especial 2 (E2)** do inimigo para calibração.
  - Pressione `9`: Selecionar o pixel do **Especial 3 (E3)** do inimigo para calibração.
* **Ajuste de Regiões (Apenas para Vida):**
  - `Q`/`A` ajusta a coordenada superior (`y_inicio`).
  - `W`/`S` ajusta a coordenada inferior (`y_fim`).
  - `E`/`D` ajusta a coordenada esquerda do jogador (`x1_jogador`).
  - `R`/`F` ajusta a coordenada direita do jogador (`x2_jogador`).
  - `T`/`G` ajusta o recuo direito do oponente (`offset_direita_inimigo`).
  - `Y`/`H` ajusta a largura da barra do oponente (`largura_inimigo`).
* **Calibração de Pixel do Especial (E1, E2, E3):**
  - Pressione a tecla do nível correspondente (`4`/`5`/`6` para Jogador ou `7`/`8`/`9` para Inimigo).
  - Posicione o cursor do mouse sobre o pixel na respectiva barra de especial que acende quando aquele nível está cheio e **clique com o botão esquerdo**.
  - A coordenada e a cor BGR do pixel serão salvas e exibidas em tempo real como um círculo marcador (azul para o jogador, verde para o inimigo).
* **Salvar e Sair:**
  - Pressione `P`: Salva todas as configurações em `data/calibracao_vida.json` e `data/calibracao_especial.json`.
  - Pressione `ESC`: Encerra o modo de calibração sem salvar as alterações temporárias pendentes.

### 2. Calibração de ROIs para Templates (Modo `calibracao_roi`)
Selecione a opção **`calibracao_roi`** no menu principal.
- Use `N`/`P` para navegar entre os alvos de ROI (ex: `acoes_perfeitas` para destreza/aparar ou `jogar_novamente` para o botão de restart).
- Use as teclas `Q/A/W/S/E/D/R/F` para mover e redimensionar a caixa amarela de visualização sobre o elemento correspondente.
- Pressione `C` para copiar a string formatada da ROI e salve-a na variável de ambiente correspondente (ex: `BOT_ROI_ACOES_PERFEITAS`, `BOT_ROI_JOGAR_NOVAMENTE` ou `BOT_ROI_COMBO_SEM_PERDA` no arquivo `.env`).


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
