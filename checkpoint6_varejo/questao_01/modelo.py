"""Funções de avaliação compartilhadas pela Questão 1."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AvaliacaoPreco:
    indices_preco: list[int]
    multiplicadores: list[float]
    precos: list[float]
    unidades_estimadas: list[float]
    caixa_operacional: float
    lucro_contribuicao: float
    margem_percentual: float
    estoque_remanescente: float
    viavel: bool
    violacoes_margem: int
    atingiu_meta: bool
    folga_meta: float
    cobertura_divida: float | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def avaliar_politica(df: pd.DataFrame, cfg: dict, indices_preco: list[int]) -> AvaliacaoPreco:
    niveis = np.asarray(cfg["niveis_preco_relativos"], dtype=float)
    if len(indices_preco) != len(df):
        raise ValueError("A política deve possuir um índice de preço para cada SKU.")
    idx = np.asarray(indices_preco, dtype=int)
    if np.any(idx < 0) or np.any(idx >= len(niveis)):
        raise ValueError("Índice de nível de preço fora do intervalo permitido.")

    multiplicadores = niveis[idx]
    precos = df["preco_atual"].to_numpy(float) * multiplicadores
    elasticidade = np.abs(df["elasticidade"].to_numpy(float))
    demanda = df["demanda_base_90d"].to_numpy(float) * np.power(
        np.maximum(precos / df["preco_atual"].to_numpy(float), 1e-9), -elasticidade
    )
    unidades = np.minimum(df["estoque"].to_numpy(float), demanda)
    taxa = df["taxa_marketplace"].to_numpy(float)
    logistica = df["custo_logistico"].to_numpy(float)
    custo = df["custo_unitario"].to_numpy(float)

    caixa_unitario = precos * (1.0 - taxa) - logistica
    contribuicao_unitaria = caixa_unitario - custo
    margem_unitaria = contribuicao_unitaria / np.maximum(precos, 1e-9)
    margem_minima = float(cfg["margem_minima_percentual"])
    violacoes = int(np.sum(margem_unitaria < margem_minima - 1e-12))

    caixa = float(np.sum(unidades * caixa_unitario))
    lucro = float(np.sum(unidades * contribuicao_unitaria))
    receita_bruta = float(np.sum(unidades * precos))
    margem = lucro / receita_bruta if receita_bruta > 0 else 0.0
    estoque_remanescente = float(np.sum(df["estoque"].to_numpy(float) - unidades))
    meta = float(cfg["meta_caixa_90d"])
    divida = float(cfg["divida_total_90d"])
    return AvaliacaoPreco(
        indices_preco=idx.tolist(),
        multiplicadores=multiplicadores.tolist(),
        precos=precos.tolist(),
        unidades_estimadas=unidades.tolist(),
        caixa_operacional=caixa,
        lucro_contribuicao=lucro,
        margem_percentual=margem,
        estoque_remanescente=estoque_remanescente,
        viavel=violacoes == 0,
        violacoes_margem=violacoes,
        atingiu_meta=caixa >= meta,
        folga_meta=caixa - meta,
        cobertura_divida=(caixa / divida if divida > 0 else None),
    )


def fitness(avaliacao: AvaliacaoPreco, escala_penalidade: float = 1_000_000_000_000_000.0) -> float:
    """Fitness maximiza caixa e penaliza soluções que violam margem mínima."""
    if avaliacao.viavel:
        return avaliacao.caixa_operacional
    return avaliacao.caixa_operacional - escala_penalidade * avaliacao.violacoes_margem


def dataframe_politica(df: pd.DataFrame, avaliacao: AvaliacaoPreco) -> pd.DataFrame:
    saida = df[["sku", "categoria", "preco_atual", "estoque"]].copy()
    saida["multiplicador_escolhido"] = avaliacao.multiplicadores
    saida["preco_escolhido"] = avaliacao.precos
    saida["unidades_estimadas"] = avaliacao.unidades_estimadas
    return saida
