import cv2
from src.calibracao.config_especial import carregar_configuracao_especial

# Cores em BGR
COR_JOGADOR = (255, 200, 80)      # Azul claro / ciano
COR_INIMIGO = (120, 120, 255)     # Vermelho claro / rosa
COR_ROXO = (255, 0, 255)          # Roxo para rastreamento de personagens
COR_ALERTA = (0, 0, 255)          # Vermelho vivo para alertas
COR_TEXTO = (255, 255, 255)       # Branco para textos informativos


def desenhar_info_hud(
    frame,
    info_vida=None,
    info_poder=None,
    info_personagens=None,
    acao_atual=None,
    slow_motion=False,
):
    """
    Desenha todas as informações de debug no frame para feedback visual completo.
    """
    if frame is None:
        return None

    frame_desenho = frame.copy()
    altura, largura = frame_desenho.shape[:2]

    # 1. Desenhar Informações de Vida
    if info_vida:
        # Vida Jogador
        bbox_jog = info_vida.get("bbox_jogador")
        if bbox_jog and len(bbox_jog) == 4:
            x1, y1, x2, y2 = bbox_jog
            cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), (0, 255, 255), 2)
            pct = info_vida.get("vida_jogador_pct")
            pct_str = f"{pct:.1f}%" if pct is not None else "?"
            cv2.putText(
                frame_desenho,
                f"Voce: {pct_str}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Vida Inimigo
        bbox_ini = info_vida.get("bbox_inimigo")
        if bbox_ini and len(bbox_ini) == 4:
            x1, y1, x2, y2 = bbox_ini
            cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), (0, 165, 255), 2)
            pct = info_vida.get("vida_inimigo_pct")
            pct_str = f"{pct:.1f}%" if pct is not None else "?"
            cv2.putText(
                frame_desenho,
                f"Inimigo: {pct_str}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

    # 1. Desenhar Calibração e Estado do Especial
    if info_poder:
        config_esp = carregar_configuracao_especial()
        
        # Desenhar círculos para os pontos calibrados do especial
        for grupo, cor_circulo in [("jogador", (255, 120, 0)), ("inimigo", (0, 255, 0))]:
            cfg_grupo = config_esp.get(grupo, {})
            pcts = info_poder.get(f"segmentos_{grupo}_pct", [0.0, 0.0, 0.0])
            
            for idx, key in enumerate(["e1", "e2", "e3"]):
                pt = cfg_grupo.get(key, {})
                px, py = pt.get("x", 0), pt.get("y", 0)
                if px > 0 and py > 0 and px < largura and py < altura:
                    ativo = pcts[idx] > 0.0
                    cor_ponto = tuple(pt.get("color", cor_circulo))
                    # Círculo externo para mostrar ponto de calibração
                    cv2.circle(frame_desenho, (px, py), 6, cor_circulo, 1)
                    # Círculo interno preenchido se ativo, senão apenas o centro
                    if ativo:
                        cv2.circle(frame_desenho, (px, py), 3, cor_ponto, -1)
                    else:
                        cv2.circle(frame_desenho, (px, py), 1, (128, 128, 128), -1)

        # Exibir Especial Textos
        lvl_jog = info_poder.get("nivel_especial_jogador", 0)
        lvl_ini = info_poder.get("nivel_especial_inimigo", 0)
        
        txt_jog = f"ESP VOCE: Lvl {lvl_jog}"
        txt_ini = f"ESP INIMIGO: Lvl {lvl_ini}"
        
        cv2.putText(
            frame_desenho,
            txt_jog,
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COR_JOGADOR,
            1,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            frame_desenho,
            txt_ini,
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 255) if lvl_ini > 0 else (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        
        # Alerta de Especial do Oponente
        if info_poder.get("tem_especial_inimigo"):
            cv2.putText(
                frame_desenho,
                "WARNING: INIMIGO TEM ESPECIAL!",
                (20, altura - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COR_ALERTA,
                2,
                cv2.LINE_AA,
            )

    # 2. Desenhar Rastreamento de Personagens
    if info_personagens:
        x, y, w, h = info_personagens.get("jogador_bbox", (0, 0, 0, 0))
        if w > 0 and h > 0:
            cv2.rectangle(frame_desenho, (x, y), (x + w, y + h), COR_ROXO, 2)
            cv2.putText(
                frame_desenho,
                "VOCE",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COR_ROXO,
                1,
                cv2.LINE_AA,
            )

        x, y, w, h = info_personagens.get("inimigo_bbox", (0, 0, 0, 0))
        if w > 0 and h > 0:
            cv2.rectangle(frame_desenho, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                frame_desenho,
                "INIMIGO",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        p1 = info_personagens.get("jogador_centro")
        p2 = info_personagens.get("inimigo_centro")
        if p1 and p2:
            cv2.line(frame_desenho, p1, p2, COR_ROXO, 1)
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            dist = info_personagens.get("distancia_px", 0)
            cv2.putText(
                frame_desenho,
                f"Dist: {dist}px",
                (mx - 30, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COR_ROXO,
                1,
                cv2.LINE_AA,
            )

    # 3. Desenhar Ação Atual
    if acao_atual is not None:
        from src.bot.acoes_luta import nome_acao
        
        cv2.putText(
            frame_desenho,
            f"ACAO: {nome_acao(acao_atual)}",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COR_TEXTO,
            1,
            cv2.LINE_AA,
        )

    # 5. Indicador de Câmera Lenta
    if slow_motion:
        # Texto destacado piscando/chamativo
        cv2.putText(
            frame_desenho,
            "* CAMERA LENTA ATIVA (Tecla L para desativar) *",
            (largura // 2 - 190, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame_desenho
