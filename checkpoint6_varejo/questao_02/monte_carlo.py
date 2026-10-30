"""Monte Carlo para demanda, preço concorrente e custo logístico (Questão 2)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from metricas import variabilidade_config


@dataclass
class Cenarios:
    fator_demanda: np.ndarray       # (cenário, semana, SKU)
    fator_concorrente: np.ndarray   # (cenário, semana, SKU)
    fator_logistica: np.ndarray     # (cenário, semana, SKU)
    seed: int

    @property
    def quantidade(self) -> int:
        return int(self.fator_demanda.shape[0])

    @property
    def semanas(self) -> int:
        return int(self.fator_demanda.shape[1])

    @property
    def produtos(self) -> int:
        return int(self.fator_demanda.shape[2])


def gerar_cenarios(cfg: dict, n_scenarios: int, n_produtos: int, seed: int) -> Cenarios:
    if n_scenarios < 1 or n_produtos < 1:
        raise ValueError("Quantidade de cenários e produtos deve ser positiva.")
    semanas = int(cfg["numero_semanas"])
    sigma_demanda, sigma_concorrente, sigma_logistica = variabilidade_config(cfg)
    rng = np.random.default_rng(int(seed))
    shape = (int(n_scenarios), semanas, int(n_produtos))
    # Lognormal com média 1: evita demandas negativas e preserva a escala média.
    fator_demanda = rng.lognormal(mean=-0.5 * sigma_demanda**2, sigma=sigma_demanda, size=shape)
    fator_concorrente = rng.lognormal(mean=-0.5 * sigma_concorrente**2, sigma=sigma_concorrente, size=shape)
    fator_logistica = np.maximum(0.0, 1.0 + rng.normal(0.0, sigma_logistica, size=shape))
    return Cenarios(fator_demanda, fator_concorrente, fator_logistica, int(seed))


def calcular_demanda_e_caixa_semana(
    df: pd.DataFrame,
    semana_indice: int,
    multiplicador_preco: float,
    fracao_estoque: float,
    cenarios: Cenarios,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (unidades vendidas, caixa operacional) por cenário/semana.

    A elasticidade é tratada pelo módulo: preço próprio maior reduz demanda;
    preço do concorrente maior aumenta demanda relativa. A variabilidade no
    config é interpretada como desvio-padrão relativo (ex.: 0.10 = 10%).
    """
    if not 0 <= semana_indice < cenarios.semanas:
        raise IndexError("Semana fora dos cenários disponíveis.")
    if multiplicador_preco <= 0 or not 0 <= fracao_estoque <= 1:
        raise ValueError("Multiplicador deve ser positivo e estoque deve estar em [0,1].")
    base_preco = df["preco_atual"].to_numpy(float)
    preco = base_preco * float(multiplicador_preco)
    elasticidade = np.abs(df["elasticidade"].to_numpy(float))
    demanda_base = df["demanda_base_semanal"].to_numpy(float)
    concorrente_base = df["preco_concorrente_base"].to_numpy(float)
    concorrente_simulado = concorrente_base[None, :] * cenarios.fator_concorrente[:, semana_indice, :]
    demanda = (
        demanda_base[None, :]
        * np.power(np.maximum(preco[None, :] / base_preco[None, :], 1e-9), -elasticidade[None, :])
        * np.power(np.maximum(concorrente_simulado / concorrente_base[None, :], 1e-9), elasticidade[None, :])
        * cenarios.fator_demanda[:, semana_indice, :]
    )
    estoque_disponivel = df["estoque_inicial"].to_numpy(float)[None, :] * fracao_estoque
    unidades = np.minimum(demanda, estoque_disponivel)
    taxa = df["taxa_marketplace"].to_numpy(float)[None, :]
    logistica = (
        df["custo_logistico_base"].to_numpy(float)[None, :]
        * cenarios.fator_logistica[:, semana_indice, :]
    )
    caixa = unidades * (preco[None, :] * (1.0 - taxa) - logistica)
    return unidades.sum(axis=1), caixa.sum(axis=1)


def simular_politica(
    df: pd.DataFrame,
    cfg: dict,
    caminho_precos: list[float],
    cenarios: Cenarios,
    promocoes: np.ndarray | None = None,
    desconto_pct: float = 0.05,
    uplift_demanda_pct: float = 0.10,
) -> dict:
    """Valida uma trajetória de preços em cenários, com estoque e dívidas."""
    n, semanas, n_prod = cenarios.fator_demanda.shape
    if len(caminho_precos) != semanas:
        raise ValueError("A trajetória de preços deve ter um multiplicador por semana.")
    if promocoes is None:
        promocoes = np.zeros((semanas, n_prod), dtype=float)
    promocoes = np.asarray(promocoes, dtype=float)
    if promocoes.shape != (semanas, n_prod):
        raise ValueError("Matriz de promoções deve ter formato (semanas, produtos).")
    if np.any((promocoes < 0) | (promocoes > 1)):
        raise ValueError("Intensidade promocional deve estar em [0,1].")

    estoque = np.broadcast_to(df["estoque_inicial"].to_numpy(float), (n, n_prod)).copy()
    caixa_saldo = np.full(n, float(cfg["caixa_inicial"]), dtype=float)
    pagamentos = {4: np.ones(n, dtype=bool), 8: np.ones(n, dtype=bool), 12: np.ones(n, dtype=bool)}
    caixa_por_semana = np.zeros((n, semanas), dtype=float)
    unidades_por_semana = np.zeros((n, semanas), dtype=float)
    lucro_total = np.zeros(n, dtype=float)
    receita_bruta_total = np.zeros(n, dtype=float)
    taxa = df["taxa_marketplace"].to_numpy(float)[None, :]
    custo_unitario = df["custo_unitario"].to_numpy(float)[None, :]
    base_preco = df["preco_atual"].to_numpy(float)
    base_concorrente = df["preco_concorrente_base"].to_numpy(float)
    demanda_base = df["demanda_base_semanal"].to_numpy(float)[None, :]
    elasticidade = np.abs(df["elasticidade"].to_numpy(float))[None, :]
    custo_logistico = df["custo_logistico_base"].to_numpy(float)[None, :]
    dividas = cfg["dividas"]

    for semana in range(semanas):
        intensidade = promocoes[semana][None, :]
        preco_cheio = base_preco[None, :] * float(caminho_precos[semana])
        preco_efetivo = preco_cheio * (1.0 - desconto_pct * intensidade)
        concorrente = base_concorrente[None, :] * cenarios.fator_concorrente[:, semana, :]
        demanda = (
            demanda_base
            * np.power(np.maximum(preco_cheio / base_preco[None, :], 1e-9), -elasticidade)
            * np.power(np.maximum(concorrente / base_concorrente[None, :], 1e-9), elasticidade)
            * cenarios.fator_demanda[:, semana, :]
            * (1.0 + uplift_demanda_pct * intensidade)
        )
        unidades = np.minimum(demanda, estoque)
        logistica_real = custo_logistico * cenarios.fator_logistica[:, semana, :]
        caixa_semana_sku = unidades * (preco_efetivo * (1.0 - taxa) - logistica_real)
        lucro_semana_sku = unidades * (
            preco_efetivo * (1.0 - taxa) - logistica_real - custo_unitario
        )
        receita_bruta_total += (unidades * preco_efetivo).sum(axis=1)
        lucro_total += lucro_semana_sku.sum(axis=1)
        caixa_semana = caixa_semana_sku.sum(axis=1)
        caixa_por_semana[:, semana] = caixa_semana
        unidades_por_semana[:, semana] = unidades.sum(axis=1)
        caixa_saldo += caixa_semana
        estoque -= unidades

        numero_semana = semana + 1
        chave_divida = f"semana_{numero_semana}"
        if chave_divida in dividas:
            valor_divida = float(dividas[chave_divida])
            pagou = caixa_saldo >= valor_divida
            if numero_semana in pagamentos:
                pagamentos[numero_semana] = pagou.copy()
            # Dívida não paga permanece em aberto; caixa não fica negativo.
            caixa_saldo[pagou] -= valor_divida

    caixa_final = caixa_saldo
    falhou_alguma = ~np.logical_and.reduce(list(pagamentos.values()))
    percentil = float(cfg.get("percentil_risco", 10))
    return {
        "quantidade_cenarios": int(n),
        "seed": int(cenarios.seed),
        "probabilidade_pagamento_semana_4": float(pagamentos[4].mean()),
        "probabilidade_pagamento_semana_8": float(pagamentos[8].mean()),
        "probabilidade_pagamento_semana_12": float(pagamentos[12].mean()),
        "probabilidade_pagar_todas": float((~falhou_alguma).mean()),
        "caixa_final_esperado": float(caixa_final.mean()),
        "caixa_final_mediano": float(np.median(caixa_final)),
        "caixa_final_percentil_risco": float(np.percentile(caixa_final, percentil)),
        "percentil_risco_utilizado": percentil,
        "margem_media_percentual": float(
            np.mean(np.divide(lucro_total, receita_bruta_total, out=np.zeros_like(lucro_total), where=receita_bruta_total > 0))
        ),
        "estoque_final_medio_unidades": float(estoque.sum(axis=1).mean()),
        "caixa_final_por_cenario": caixa_final,
        "pagamentos_por_cenario": {
            semana: valores for semana, valores in pagamentos.items()
        },
        "caixa_por_semana_medio": caixa_por_semana.mean(axis=0),
        "unidades_por_semana_media": unidades_por_semana.mean(axis=0),
        "fracasso_por_cenario": falhou_alguma,
    }
