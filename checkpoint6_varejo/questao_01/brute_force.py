"""Busca exata de combinações de preço para um subconjunto reduzido."""
from __future__ import annotations

from itertools import product
import time
import pandas as pd

from .modelo import avaliar_politica, fitness


def resolver_brute_force(df: pd.DataFrame, cfg: dict, limite_combinacoes: int = 2_000_000) -> dict:
    if len(df) < 6:
        raise ValueError("O Brute Force exige pelo menos 6 produtos, conforme o enunciado.")
    qtd_niveis = len(cfg["niveis_preco_relativos"])
    total = qtd_niveis ** len(df)
    if total > limite_combinacoes:
        raise ValueError(
            f"Busca teria {total:,} combinações, acima do limite {limite_combinacoes:,}. "
            "Reduza o subconjunto ou a quantidade de níveis de preço."
        )
    inicio = time.perf_counter()
    melhor = None
    melhor_fitness = float("-inf")
    viavel_encontrada = False
    for genes in product(range(qtd_niveis), repeat=len(df)):
        avaliacao = avaliar_politica(df, cfg, list(genes))
        score = fitness(avaliacao)
        if score > melhor_fitness:
            melhor_fitness = score
            melhor = avaliacao
            viavel_encontrada = avaliacao.viavel
    tempo = time.perf_counter() - inicio
    if melhor is None or not viavel_encontrada:
        raise ValueError(
            "Nenhuma combinação atende ao piso de margem mínima no subconjunto. "
            "Confira os dados oficiais, níveis de preço e definição da margem."
        )
    return {
        "algoritmo": "Brute Force",
        "avaliacao": melhor.to_dict(),
        "fitness": float(melhor_fitness),
        "combinacoes_avaliadas": int(total),
        "tempo_interno_segundos": float(tempo),
    }
