import time

from src.ai.cerebro import CerebroIA
from src.ai.deteccao_luta import esta_lutando, esta_nocaute
from src.ai.recompensa import obter_recompensa
from src.bot.acoes_luta import executar_acao
from src.bot.vida import desenhar_info_vida, obter_info_vida
from src.utils.visao_debug import exibir


class Luta:

  def __init__(self, exibir_debug=True):
    self.exibir_debug = exibir_debug
    self.cerebro = CerebroIA()
    self.estado = "BUSCANDO_LUTA"
    self.episodio = 0
    self.vida_jogador_anterior = None
    self.vida_inimigo_anterior = None

  def processar_frame(self, frame):
    em_luta = esta_lutando(frame)
    info_vida = None

    if self.exibir_debug:
      if em_luta:
        info_vida = obter_info_vida(frame)
        frame_debug = desenhar_info_vida(frame.copy(), info_vida)
        exibir(frame_debug)
      else:
        exibir(frame)

    if esta_nocaute(frame):
      print("K.O detectado")
      self.estado = "BUSCANDO_LUTA"
      self.episodio += 1
      self.vida_jogador_anterior = None
      self.vida_inimigo_anterior = None
      print("Episodio finalizado:", self.episodio)
      time.sleep(3)
      return

    if not em_luta:
      return

    if info_vida is None:
      info_vida = obter_info_vida(frame)

    if self.estado != "LUTANDO":
      print("Luta comecou")
      self.estado = "LUTANDO"

    estado_atual = self.cerebro.obter_estado()
    acao = self.cerebro.escolher_acao(estado_atual)
    executar_acao(acao)

    recompensa, _ = obter_recompensa(
      frame,
      vida_jogador_atual=info_vida["vida_jogador_pct"],
      vida_jogador_anterior=self.vida_jogador_anterior,
      vida_inimigo_atual=info_vida["vida_inimigo_pct"],
      vida_inimigo_anterior=self.vida_inimigo_anterior,
      colunas_escuras_jogador_finais=info_vida["colunas_escuras_jogador_finais"],
      colunas_escuras_inimigo_finais=info_vida["colunas_escuras_inimigo_finais"],
    )

    self.cerebro.aprender(estado_atual, acao, recompensa)

    print(
      "ACAO:",
      acao,
      "RECOMPENSA:",
      recompensa,
    )
    print(
      "VIDA_VOCE:",
      self._formatar_pct(info_vida["vida_jogador_pct"]),
      self._barra_vida_texto(info_vida["vida_jogador_pct"]),
    )
    print(
      "VIDA_INIMIGO:",
      self._formatar_pct(info_vida["vida_inimigo_pct"]),
      self._barra_vida_texto(info_vida["vida_inimigo_pct"]),
    )

    self.vida_jogador_anterior = info_vida["vida_jogador_pct"]
    self.vida_inimigo_anterior = info_vida["vida_inimigo_pct"]

  @staticmethod
  def _formatar_pct(valor):
    if valor is None:
      return "?"
    return f"{valor:.1f}%"

  @staticmethod
  def _barra_vida_texto(valor, largura=20):
    if valor is None:
      return "?" * largura

    preenchidos = max(0, min(largura, int(round((valor / 100.0) * largura))))
    vazios = largura - preenchidos
    return ("█" * preenchidos) + ("░" * vazios)
