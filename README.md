# Fight Bot

Bot de luta com visao computacional + IA por reforco simples.

## Estrutura do projeto

- `assets/`: imagens de referencia para deteccao na tela (botoes, KO, vida etc.)
- `data/`: dados persistidos localmente (memoria/estado da IA)
- `src/ai/`: decisao da IA, recompensa e deteccao de estado de luta
- `src/bot/`: captura da tela, acoes de teclado/mouse e launcher principal
- `src/core/`: recursos compartilhados (ex.: persistencia)
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
- `treino`: foco no loop de treino da IA

## Convencoes do projeto

- nomes de arquivos em `snake_case`
- nomes de funcoes em `snake_case`
- classes em `PascalCase`
- constantes em `UPPER_SNAKE_CASE`
- um idioma por identificador (preferencia: portugues)
