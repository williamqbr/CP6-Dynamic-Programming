"""Greedy para distribuir o orçamento promocional da Questão 1."""
from __future__ import annotations

import pandas as pd
import numpy as np

from .modelo import AvaliacaoPreco


def alocar_incentivos(
    df: pd.DataFrame,
    cfg: dict,
    avaliacao: AvaliacaoPreco | dict,
    desconto_pct: float = 0.05,
    aumento_demanda_pct: float = 0.10,
) -> dict:
    """Aloca orçamento por ROI marginal, permitindo alocação fracionada.

    Hipótese explícita (não detalhada numericamente no PDF): a promoção reduz
    em desconto_pct o preço das unidades incrementais e gera aumento de demanda
    de aumento_demanda_pct. Ambos são parâmetros expostos na CLI.
    """
    if not 0 <= desconto_pct < 1 or aumento_demanda_pct < 0:
        raise ValueError("Desconto deve estar em [0,1) e uplift não pode ser negativo.")
    if isinstance(avaliacao, dict):
        indices = avaliacao["indices_preco"]
        niveis = np.asarray(cfg["niveis_preco_relativos"], dtype=float)
        precos = df["preco_atual"].to_numpy(float) * niveis[np.asarray(indices, dtype=int)]
        unidades_base = np.asarray(avaliacao["unidades_estimadas"], dtype=float)
    else:
        precos = np.asarray(avaliacao.precos, dtype=float)
        unidades_base = np.asarray(avaliacao.unidades_estimadas, dtype=float)

    estoque = df["estoque"].to_numpy(float)
    capacidade = np.maximum(0.0, estoque - unidades_base)
    alvo_incremental = np.minimum(capacidade, unidades_base * aumento_demanda_pct)
    taxa = df["taxa_marketplace"].to_numpy(float)
    logistica = df["custo_logistico"].to_numpy(float)
    custo = df["custo_unitario"].to_numpy(float)
    margem_minima = float(cfg["margem_minima_percentual"])

    custo_por_unidade = precos * desconto_pct
    preco_promocional = precos * (1.0 - desconto_pct)
    margem_promocional = (preco_promocional * (1.0 - taxa) - logistica - custo) / np.maximum(preco_promocional, 1e-9)
    contribuicao = preco_promocional * (1.0 - taxa) - logistica - custo
    elegivel = (alvo_incremental > 0) & (margem_promocional >= margem_minima) & (custo_por_unidade > 0) & (contribuicao > 0)
    custo_total_campanha = alvo_incremental * custo_por_unidade
    beneficio_total = alvo_incremental * contribuicao
    roi = np.divide(beneficio_total, custo_total_campanha, out=np.zeros_like(beneficio_total), where=custo_total_campanha > 0)
    roi[~elegivel] = 0.0

    orcamento = float(cfg["orcamento_promocional"])
    restante = orcamento
    alocacoes = np.zeros(len(df), dtype=float)
    for i in np.argsort(-roi):
        if roi[i] <= 0 or restante <= 1e-12:
            continue
        gasto = min(custo_total_campanha[i], restante)
        fracao = gasto / custo_total_campanha[i] if custo_total_campanha[i] > 0 else 0.0
        alocacoes[i] = alvo_incremental[i] * fracao
        restante -= gasto

    gasto = alocacoes * custo_por_unidade
    precos_promocionais_efetivos = np.where(alocacoes > 0, preco_promocional, precos)
    receita_caixa_base = unidades_base * (precos * (1.0 - taxa) - logistica)
    caixa_incremental = alocacoes * (precos_promocionais_efetivos * (1.0 - taxa) - logistica)
    caixa_antes = float(receita_caixa_base.sum())
    caixa_depois = float(caixa_antes + caixa_incremental.sum())
    lucro_incremental = alocacoes * (
        precos_promocionais_efetivos * (1.0 - taxa) - logistica - custo
    )
    lucro_base = unidades_base * (precos * (1.0 - taxa) - logistica - custo)
    receita_bruta_base = float(np.sum(unidades_base * precos))
    receita_bruta_incremental = float(np.sum(alocacoes * precos_promocionais_efetivos))
    lucro_total_depois = float(lucro_base.sum() + lucro_incremental.sum())
    receita_total_depois = receita_bruta_base + receita_bruta_incremental
    margem_depois = lucro_total_depois / receita_total_depois if receita_total_depois > 0 else 0.0
    estoque_depois = float(np.maximum(0.0, estoque - unidades_base - alocacoes).sum())
    divida = float(cfg.get("divida_total_90d", 0.0))

    tabela = pd.DataFrame({
        "sku": df["sku"].astype(str),
        "preco_base": precos,
        "unidades_base": unidades_base,
        "unidades_promocionais_incrementais": alocacoes,
        "roi_beneficio_por_real": roi,
        "gasto_promocional": gasto,
        "caixa_incremental": caixa_incremental,
        "lucro_contribuicao_incremental": lucro_incremental,
    })
    return {
        "tabela": tabela,
        "caixa_antes_greedy": caixa_antes,
        "caixa_depois_greedy": caixa_depois,
        "orcamento_disponivel": orcamento,
        "orcamento_utilizado": float(gasto.sum()),
        "orcamento_nao_utilizado": float(max(0.0, orcamento - gasto.sum())),
        "unidades_promocionais_incrementais": float(alocacoes.sum()),
        "lucro_contribuicao_depois": lucro_total_depois,
        "margem_percentual_depois": margem_depois,
        "estoque_remanescente_depois": estoque_depois,
        "cobertura_divida_depois": caixa_depois / divida if divida > 0 else None,
        "atingiu_meta_de_caixa_depois": caixa_depois >= float(cfg["meta_caixa_90d"]),
        "desconto_pct": desconto_pct,
        "aumento_demanda_pct": aumento_demanda_pct,
    }
