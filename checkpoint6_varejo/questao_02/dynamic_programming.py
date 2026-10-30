"""Programação Dinâmica estocástica aproximada para preço semanal (Questão 2).

Estado: (semana, bucket de estoque remanescente, bucket de caixa disponível).
A função valor estima a probabilidade de pagar os compromissos restantes usando
as distribuições Monte Carlo da etapa anterior. Buckets tornam o problema
computável; o plano é reconstruído no caminho médio e validado em novos cenários.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .monte_carlo import Cenarios, calcular_demanda_e_caixa_semana


def resolver_programacao_dinamica(
    df: pd.DataFrame,
    cfg: dict,
    cenarios: Cenarios,
    stock_bins: int = 12,
    cash_bins: int = 30,
) -> dict:
    semanas = int(cfg["numero_semanas"])
    niveis = np.asarray(cfg["niveis_preco_relativos"], dtype=float)
    if stock_bins < 3 or cash_bins < 5:
        raise ValueError("A DP precisa de pelo menos 3 buckets de estoque e 5 de caixa.")
    if cenarios.semanas != semanas or cenarios.produtos != len(df):
        raise ValueError("Dimensões dos cenários não correspondem ao catálogo/configuração.")

    estoque_total = float(df["estoque_inicial"].sum())
    if estoque_total <= 0:
        raise ValueError("Estoque inicial total deve ser positivo para a DP.")
    max_mult = float(np.max(niveis))
    max_preco = df["preco_atual"].to_numpy(float) * max_mult
    max_caixa_total = float(cfg["caixa_inicial"]) + float(
        np.sum(df["estoque_inicial"].to_numpy(float) * max_preco * (1.0 - df["taxa_marketplace"].to_numpy(float)))
    )
    max_caixa_total = max(max_caixa_total, float(cfg["caixa_inicial"]) + 1.0)
    passo_caixa = max_caixa_total / (cash_bins - 1)
    dividas = {int(k.split("_")[1]): float(v) for k, v in cfg["dividas"].items()}

    # Pré-calcula resultados por semana, ação de preço e nível de estoque.
    vendas = np.zeros((semanas, len(niveis), stock_bins, cenarios.quantidade), dtype=np.float32)
    caixas = np.zeros_like(vendas)
    fracoes = np.linspace(0.0, 1.0, stock_bins)
    for semana in range(semanas):
        for acao, multiplicador in enumerate(niveis):
            for idx_estoque, fracao in enumerate(fracoes):
                unidades, caixa = calcular_demanda_e_caixa_semana(
                    df, semana, float(multiplicador), float(fracao), cenarios
                )
                vendas[semana, acao, idx_estoque] = unidades / estoque_total
                caixas[semana, acao, idx_estoque] = caixa

    valor = np.zeros((semanas + 1, stock_bins, cash_bins), dtype=np.float32)
    valor[semanas, :, :] = 1.0
    politica = np.zeros((semanas, stock_bins, cash_bins), dtype=np.int16)

    for semana in range(semanas - 1, -1, -1):
        divida = dividas.get(semana + 1, 0.0)
        for idx_estoque in range(stock_bins):
            fracao_atual = fracoes[idx_estoque]
            for idx_caixa in range(cash_bins):
                caixa_atual = idx_caixa * passo_caixa
                melhor_valor = -1.0
                melhor_acao = 0
                for acao in range(len(niveis)):
                    fracao_vendida = vendas[semana, acao, idx_estoque].astype(float)
                    fracao_proxima = np.clip(fracao_atual - fracao_vendida, 0.0, 1.0)
                    idx_estoque_proximo = np.floor(fracao_proxima * (stock_bins - 1) + 1e-12).astype(int)
                    caixa_pre_pagamento = caixa_atual + caixas[semana, acao, idx_estoque].astype(float)
                    passou = caixa_pre_pagamento >= divida
                    caixa_depois = np.maximum(0.0, caixa_pre_pagamento - divida)
                    idx_caixa_proximo = np.floor(caixa_depois / passo_caixa + 1e-12).astype(int)
                    idx_caixa_proximo = np.clip(idx_caixa_proximo, 0, cash_bins - 1)
                    valores_proximos = valor[semana + 1, idx_estoque_proximo, idx_caixa_proximo]
                    probabilidade = float(np.mean(valores_proximos * passou))
                    if probabilidade > melhor_valor:
                        melhor_valor = probabilidade
                        melhor_acao = acao
                valor[semana, idx_estoque, idx_caixa] = max(0.0, melhor_valor)
                politica[semana, idx_estoque, idx_caixa] = melhor_acao

    # Extrai uma trajetória operacional ao longo do estado médio esperado.
    idx_estoque = stock_bins - 1
    caixa_atual = float(cfg["caixa_inicial"])
    caminho = []
    detalhes = []
    for semana in range(semanas):
        idx_caixa = int(np.clip(np.floor(caixa_atual / passo_caixa), 0, cash_bins - 1))
        acao = int(politica[semana, idx_estoque, idx_caixa])
        multiplicador = float(niveis[acao])
        unidades = vendas[semana, acao, idx_estoque].astype(float) * estoque_total
        caixa_semana = float(np.mean(caixas[semana, acao, idx_estoque]))
        caminho.append(multiplicador)
        detalhes.append({
            "semana": semana + 1,
            "indice_estoque": idx_estoque,
            "caixa_estimado_inicio": caixa_atual,
            "multiplicador_preco": multiplicador,
            "unidades_estimadas_mc": float(np.mean(unidades)),
            "caixa_estimado_semana": caixa_semana,
        })
        fracao_proxima = max(0.0, fracoes[idx_estoque] - float(np.mean(vendas[semana, acao, idx_estoque])))
        idx_estoque = int(np.clip(np.floor(fracao_proxima * (stock_bins - 1) + 1e-12), 0, stock_bins - 1))
        caixa_atual += caixa_semana
        if semana + 1 in dividas and caixa_atual >= dividas[semana + 1]:
            caixa_atual -= dividas[semana + 1]

    caixa_inicial_idx = int(np.clip(np.floor(float(cfg["caixa_inicial"]) / passo_caixa), 0, cash_bins - 1))
    return {
        "algoritmo": "Programação Dinâmica",
        "probabilidade_sucesso_estimada_dp": float(valor[0, stock_bins - 1, caixa_inicial_idx]),
        "caminho_precos_multiplicadores": caminho,
        "trajetoria_estado_medio": detalhes,
        "parametros": {
            "stock_bins": stock_bins,
            "cash_bins": cash_bins,
            "passo_caixa": passo_caixa,
            "cenarios_construcao": cenarios.quantidade,
            "seed_cenarios": cenarios.seed,
            "niveis_preco": niveis.tolist(),
        },
        "politica": politica,
    }
