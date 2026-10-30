"""Greedy semanal para alocar o orçamento de destaque/mídia/frete."""
from __future__ import annotations

import numpy as np
import pandas as pd


def alocar_promocoes_semanais(
    df: pd.DataFrame,
    cfg: dict,
    caminho_precos: list[float],
    desconto_pct: float = 0.05,
    uplift_demanda_pct: float = 0.10,
) -> dict:
    """Seleciona promoções por benefício marginal por real investido.

    A promoção é aplicada como desconto sobre as unidades promovidas e uplift
    de demanda proporcional à intensidade (0 a 1). O enunciado menciona
    destaques/mídia/frete, mas não determina a conversão financeira; estes dois
    parâmetros tornam a hipótese visível e ajustável.
    """
    if not 0 <= desconto_pct < 1 or uplift_demanda_pct < 0:
        raise ValueError("Desconto deve estar em [0,1) e uplift não pode ser negativo.")
    semanas = int(cfg["numero_semanas"])
    if len(caminho_precos) != semanas:
        raise ValueError("A trajetória deve conter um multiplicador por semana.")
    n = len(df)
    intensidade = np.zeros((semanas, n), dtype=float)
    estoque_restante = df["estoque_inicial"].to_numpy(float).copy()
    registros = []
    taxa = df["taxa_marketplace"].to_numpy(float)
    custo = df["custo_unitario"].to_numpy(float)
    logistica = df["custo_logistico_base"].to_numpy(float)
    demanda_base = df["demanda_base_semanal"].to_numpy(float)
    elasticidade = np.abs(df["elasticidade"].to_numpy(float))
    preco_base = df["preco_atual"].to_numpy(float)
    concorrente = df["preco_concorrente_base"].to_numpy(float)
    orçamento_semanal = float(cfg["orcamento_promocional_semanal"])
    orcamento_total_gasto = 0.0

    for semana, multiplicador in enumerate(caminho_precos):
        preco = preco_base * float(multiplicador)
        demanda = demanda_base * np.power(np.maximum(preco / preco_base, 1e-9), -elasticidade)
        venda_base = np.minimum(estoque_restante, demanda)
        incremento_maximo = np.minimum(
            np.maximum(0.0, estoque_restante - venda_base), venda_base * uplift_demanda_pct
        )
        gasto_por_unidade = preco * desconto_pct
        preco_promo = preco * (1.0 - desconto_pct)
        contribuicao = preco_promo * (1.0 - taxa) - logistica - custo
        gasto_campanha = (venda_base + incremento_maximo) * gasto_por_unidade
        beneficio = incremento_maximo * contribuicao
        roi = np.divide(beneficio, gasto_campanha, out=np.zeros(n), where=gasto_campanha > 0)
        roi[(contribuicao <= 0) | (incremento_maximo <= 0)] = 0.0

        restante = orçamento_semanal
        gasto_sku = np.zeros(n)
        for i in np.argsort(-roi):
            if roi[i] <= 0 or restante <= 1e-12:
                continue
            gasto = min(float(gasto_campanha[i]), restante)
            gasto_sku[i] = gasto
            intensidade[semana, i] = gasto / gasto_campanha[i] if gasto_campanha[i] > 0 else 0.0
            restante -= gasto

        unidades_incrementais = incremento_maximo * intensidade[semana]
        estoque_restante -= venda_base + unidades_incrementais
        gasto_semana = float(gasto_sku.sum())
        orcamento_total_gasto += gasto_semana
        registros.append({
            "semana": semana + 1,
            "orcamento_disponivel": orçamento_semanal,
            "orcamento_utilizado": gasto_semana,
            "orcamento_nao_utilizado": max(0.0, orçamento_semanal - gasto_semana),
            "skus_promovidos": int(np.sum(intensidade[semana] > 0)),
            "unidades_incrementais_estimadas": float(unidades_incrementais.sum()),
        })

    return {
        "intensidade_promocional": intensidade,
        "resumo_semanal": pd.DataFrame(registros),
        "orcamento_total_utilizado": orcamento_total_gasto,
        "desconto_pct": desconto_pct,
        "uplift_demanda_pct": uplift_demanda_pct,
    }
