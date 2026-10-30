"""Algoritmo Genético para buscar uma política de preços por SKU."""
from __future__ import annotations

import random
import time
import pandas as pd

from .modelo import avaliar_politica, fitness


def executar_algoritmo_genetico(
    df: pd.DataFrame,
    cfg: dict,
    seed: int,
    tamanho_populacao: int = 80,
    geracoes: int = 120,
    taxa_mutacao: float = 0.08,
    taxa_crossover: float = 0.85,
    elitismo: int = 2,
    torneio: int = 3,
) -> dict:
    if tamanho_populacao < 4 or geracoes < 1:
        raise ValueError("População deve ser >= 4 e gerações >= 1.")
    if not 0 <= taxa_mutacao <= 1 or not 0 <= taxa_crossover <= 1:
        raise ValueError("Taxas de mutação e crossover devem estar em [0, 1].")
    if not 1 <= elitismo < tamanho_populacao:
        raise ValueError("Elitismo deve estar entre 1 e população-1.")
    if torneio < 2:
        raise ValueError("Torneio deve ter pelo menos 2 indivíduos.")

    rng = random.Random(int(seed))
    n_genes = len(df)
    n_niveis = len(cfg["niveis_preco_relativos"])
    populacao = [[rng.randrange(n_niveis) for _ in range(n_genes)] for _ in range(tamanho_populacao)]
    # Inclui um cromossomo viável de referência quando cada SKU tem ao menos
    # um nível que respeita o piso de margem unitária.
    niveis = cfg["niveis_preco_relativos"]
    genes_viaveis = []
    margem_minima = float(cfg["margem_minima_percentual"])
    for _, linha in df.iterrows():
        opcoes = []
        for indice, multiplicador in enumerate(niveis):
            preco = float(linha["preco_atual"]) * float(multiplicador)
            margem = (preco * (1.0 - float(linha["taxa_marketplace"])) - float(linha["custo_logistico"]) - float(linha["custo_unitario"])) / preco
            if margem >= margem_minima:
                opcoes.append(indice)
        if not opcoes:
            genes_viaveis = []
            break
        genes_viaveis.append(opcoes[-1])
    if genes_viaveis:
        populacao[0] = genes_viaveis

    def pontuar(individuo):
        resultado = avaliar_politica(df, cfg, individuo)
        return fitness(resultado), resultado

    inicio = time.perf_counter()
    historico = []
    melhor_global = None
    melhor_score = float("-inf")

    for geracao in range(geracoes):
        avaliados = []
        for individuo in populacao:
            score, resultado = pontuar(individuo)
            avaliados.append((score, individuo, resultado))
        avaliados.sort(key=lambda item: item[0], reverse=True)
        score_atual, individuo_atual, avaliacao_atual = avaliados[0]
        if score_atual > melhor_score:
            melhor_score = score_atual
            melhor_global = (individuo_atual[:], avaliacao_atual)
        historico.append({
            "geracao": geracao,
            "melhor_fitness": float(score_atual),
            "melhor_caixa_viavel": float(max(
                (item[2].caixa_operacional for item in avaliados if item[2].viavel), default=0.0
            )),
        })

        nova = [item[1][:] for item in avaliados[:elitismo]]
        while len(nova) < tamanho_populacao:
            def selecionar():
                candidatos = rng.sample(avaliados, k=min(torneio, len(avaliados)))
                return max(candidatos, key=lambda item: item[0])[1][:]
            pai1, pai2 = selecionar(), selecionar()
            filho1, filho2 = pai1[:], pai2[:]
            if n_genes > 1 and rng.random() < taxa_crossover:
                ponto = rng.randrange(1, n_genes)
                filho1 = pai1[:ponto] + pai2[ponto:]
                filho2 = pai2[:ponto] + pai1[ponto:]
            for filho in (filho1, filho2):
                for posicao in range(n_genes):
                    if rng.random() < taxa_mutacao:
                        alternativas = [i for i in range(n_niveis) if i != filho[posicao]]
                        if alternativas:
                            filho[posicao] = rng.choice(alternativas)
                if len(nova) < tamanho_populacao:
                    nova.append(filho)
        populacao = nova

    tempo = time.perf_counter() - inicio
    assert melhor_global is not None
    individuo, avaliacao = melhor_global
    return {
        "algoritmo": "Algoritmo Genético",
        "avaliacao": avaliacao.to_dict(),
        "fitness": float(melhor_score),
        "viavel": bool(avaliacao.viavel),
        "indices_preco": individuo,
        "historico": historico,
        "parametros": {
            "tamanho_populacao": tamanho_populacao,
            "geracoes": geracoes,
            "taxa_mutacao": taxa_mutacao,
            "taxa_crossover": taxa_crossover,
            "elitismo": elitismo,
            "torneio": torneio,
            "seed": int(seed),
        },
        "tempo_interno_segundos": float(tempo),
    }
