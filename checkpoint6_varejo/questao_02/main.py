"""Execução integrada da Questão 2: Monte Carlo -> DP -> Greedy -> Monte Carlo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dados import carregar_q2
from metricas import criar_pasta, medir_desempenho, salvar_json
from questao_02.dynamic_programming import resolver_programacao_dinamica
from questao_02.greedy import alocar_promocoes_semanais
from questao_02.monte_carlo import gerar_cenarios, simular_politica


def benchmark_escalabilidade(df: pd.DataFrame, cfg: dict, seed: int, stock_bins: int, cash_bins: int) -> list[dict]:
    n = len(df)
    tamanhos = sorted(set([max(1, n // 4), max(1, n // 2), n]))
    registros = []
    for pos, tamanho in enumerate(tamanhos):
        sub = df.iloc[:tamanho].reset_index(drop=True)
        def tarefa():
            cenarios = gerar_cenarios(cfg, 2000, len(sub), seed + 5000 + pos)
            dp = resolver_programacao_dinamica(sub, cfg, cenarios, stock_bins=stock_bins, cash_bins=cash_bins)
            return cenarios, dp
        _, met = medir_desempenho(tarefa)
        registros.append({
            "quantidade_skus": tamanho,
            "cenarios_construcao": 2000,
            "tempo_monte_carlo_mais_dp_s": met["tempo_segundos"],
            "pico_memoria_python_mb": met["pico_memoria_python_mb"],
        })
    return registros


def montar_trajetoria_produtos(df: pd.DataFrame, cfg: dict, caminho: list[float], promocoes: np.ndarray,
                               desconto_pct: float, uplift_pct: float) -> pd.DataFrame:
    """Gera série explicável de preço/estoque por SKU usando demanda média sem choque."""
    saldo = df["estoque_inicial"].to_numpy(float).copy()
    linhas = []
    for semana, mult in enumerate(caminho):
        intensidade = promocoes[semana]
        preco_cheio = df["preco_atual"].to_numpy(float) * mult
        preco_efetivo = preco_cheio * (1.0 - desconto_pct * intensidade)
        elasticidade = np.abs(df["elasticidade"].to_numpy(float))
        demanda = df["demanda_base_semanal"].to_numpy(float) * np.power(
            np.maximum(preco_cheio / df["preco_atual"].to_numpy(float), 1e-9), -elasticidade
        ) * (1.0 + uplift_pct * intensidade)
        unidades = np.minimum(saldo, demanda)
        for i, sku in enumerate(df["sku"].astype(str)):
            linhas.append({
                "semana": semana + 1,
                "sku": sku,
                "preco_escolhido": float(preco_efetivo[i]),
                "multiplicador_preco": float(mult),
                "estoque_inicio_estimado": float(saldo[i]),
                "demanda_estimada": float(demanda[i]),
                "unidades_vendidas_estimadas": float(unidades[i]),
                "estoque_fim_estimado": float(saldo[i] - unidades[i]),
                "intensidade_promocional": float(intensidade[i]),
            })
        saldo -= unidades
    return pd.DataFrame(linhas)


def executar(args: argparse.Namespace) -> dict:
    dados = carregar_q2(args.data_dir, args.grupo)
    df = dados.catalogo.sort_values("sku", kind="stable").reset_index(drop=True)
    cfg = dict(dados.config)
    if args.percentil_risco is not None:
        if not 0 < args.percentil_risco < 100:
            raise ValueError("Percentil de risco deve estar entre 0 e 100.")
        cfg["percentil_risco"] = float(args.percentil_risco)

    seed = int(cfg["seed_final"])
    n_construcao = max(2000, int(args.cenarios_construcao or cfg["n_simulacoes_recomendadas"]))
    n_validacao = max(5000, int(args.cenarios_validacao))
    saida = criar_pasta(ROOT / "questao_02" / "resultados")

    cenarios_func = lambda: gerar_cenarios(cfg, n_construcao, len(df), seed)
    cenarios, met_mc = medir_desempenho(cenarios_func)
    dp_func = lambda: resolver_programacao_dinamica(
        df, cfg, cenarios, stock_bins=args.stock_bins, cash_bins=args.cash_bins
    )
    dp, met_dp = medir_desempenho(dp_func)

    greedy_func = lambda: alocar_promocoes_semanais(
        df, cfg, dp["caminho_precos_multiplicadores"],
        desconto_pct=args.desconto_promocional,
        uplift_demanda_pct=args.uplift_demanda,
    )
    greedy, met_greedy = medir_desempenho(greedy_func)
    greedy["resumo_semanal"].to_csv(saida / "greedy_resumo_semanal.csv", index=False, encoding="utf-8-sig")

    # Seed separada da construção para evitar validar nos mesmos cenários.
    seed_validacao = seed + 1_000_003
    validacao_func = lambda: gerar_cenarios(cfg, n_validacao, len(df), seed_validacao)
    cenarios_validacao, met_geracao_validacao = medir_desempenho(validacao_func)
    simulacao_func = lambda: simular_politica(
        df, cfg, dp["caminho_precos_multiplicadores"], cenarios_validacao,
        promocoes=greedy["intensidade_promocional"],
        desconto_pct=args.desconto_promocional,
        uplift_demanda_pct=args.uplift_demanda,
    )
    validacao, met_validacao = medir_desempenho(simulacao_func)

    # Evidências tabulares e gráficas.
    caminho_df = pd.DataFrame(dp["trajetoria_estado_medio"])
    caminho_df.to_csv(saida / "caminho_dp.csv", index=False, encoding="utf-8-sig")
    produtos_semana = montar_trajetoria_produtos(
        df, cfg, dp["caminho_precos_multiplicadores"], greedy["intensidade_promocional"],
        args.desconto_promocional, args.uplift_demanda,
    )
    produtos_semana.to_csv(saida / "trajetoria_preco_estoque_skus.csv", index=False, encoding="utf-8-sig")
    distribuicao = pd.DataFrame({
        "cenario": np.arange(1, n_validacao + 1),
        "caixa_final": validacao["caixa_final_por_cenario"],
        "pagou_todas": ~validacao["fracasso_por_cenario"],
    })
    distribuicao.to_csv(saida / "distribuicao_caixa_final.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"compromisso": "semana_4", "probabilidade_pagamento": validacao["probabilidade_pagamento_semana_4"]},
        {"compromisso": "semana_8", "probabilidade_pagamento": validacao["probabilidade_pagamento_semana_8"]},
        {"compromisso": "semana_12", "probabilidade_pagamento": validacao["probabilidade_pagamento_semana_12"]},
        {"compromisso": "todas_as_dividas", "probabilidade_pagamento": validacao["probabilidade_pagar_todas"]},
    ]).to_csv(saida / "probabilidade_pagamentos.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    plt.hist(validacao["caixa_final_por_cenario"], bins=35)
    plt.xlabel("Caixa final após compromissos")
    plt.ylabel("Frequência de cenários")
    plt.title("Distribuição do caixa final — validação Monte Carlo")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(saida / "distribuicao_caixa_final.png", dpi=160)
    plt.close()

    escala = benchmark_escalabilidade(df, cfg, seed, args.stock_bins, args.cash_bins)
    pd.DataFrame(escala).to_csv(saida / "escalabilidade.csv", index=False, encoding="utf-8-sig")

    resumo_validacao = {k: v for k, v in validacao.items() if k not in (
        "caixa_final_por_cenario", "pagamentos_por_cenario", "caixa_por_semana_medio",
        "unidades_por_semana_media", "fracasso_por_cenario",
    )}
    risco_observado = 1.0 - validacao["probabilidade_pagar_todas"]
    resumo = {
        "grupo": dados.grupo,
        "integridade": {"status": dados.integridade.status, "resumo": dados.integridade.resumo(), "detalhes": dados.integridade.detalhes},
        "seed_construcao": seed,
        "seed_validacao_independente": seed_validacao,
        "algoritmos_utilizados": 3,
        "fluxo": "Monte Carlo estima resultados por ação de preço; DP escolhe trajetória semanal; Greedy distribui orçamento por semana; Monte Carlo independente valida o plano completo.",
        "monte_carlo_construcao": {"quantidade": n_construcao, "seed": seed},
        "programacao_dinamica": {k: v for k, v in dp.items() if k != "politica"},
        "greedy": {k: v for k, v in greedy.items() if k not in ("intensidade_promocional", "resumo_semanal")},
        "validacao_monte_carlo": resumo_validacao,
        "limite_risco_falha_configurado": cfg["limite_risco_falha"],
        "risco_falha_observado": risco_observado,
        "dentro_limite_risco": risco_observado <= float(cfg["limite_risco_falha"]),
        "desempenho": {
            "geracao_cenarios_construcao": met_mc,
            "dp": met_dp,
            "greedy": met_greedy,
            "geracao_cenarios_validacao": met_geracao_validacao,
            "simulacao_validacao": met_validacao,
            "escalabilidade": escala,
        },
        "hipoteses_modelagem": {
            "demanda": "demanda semanal multiplicada por elasticidade-preço própria, fator de preço do concorrente e choque lognormal de demanda.",
            "concorrente": "preço concorrente varia por fator lognormal de média 1; preço concorrente maior aumenta a demanda relativa do varejista.",
            "logistica": "custo logístico multiplicado por max(0, 1 + choque normal).",
            "variabilidade": "desvios relativos (frações); reconhece aliases comuns e usa defaults 15% demanda, 5% concorrente e 10% logística se as chaves não estiverem presentes.",
            "dp": "estado discretizado por semana, fração de estoque remanescente e caixa disponível; valor aproxima probabilidade de cumprir os compromissos a partir dos cenários de construção.",
            "promocao": f"Desconto {args.desconto_promocional:.1%}, uplift {args.uplift_demanda:.1%}; ambos ajustáveis por CLI.",
        },
    }
    salvar_json(saida / "resultado_q2.json", resumo)
    print(f"Q2 concluída. Resultados em: {saida}")
    print(f"Probabilidade de pagar todas: {validacao['probabilidade_pagar_todas']:.2%}")
    print(f"Risco observado: {risco_observado:.2%} | limite: {cfg['limite_risco_falha']:.2%}")
    return resumo


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 6 — Questão 2")
    parser.add_argument("--data-dir", default="data", help="Pasta com arquivos oficiais do notebook.")
    parser.add_argument("--grupo", default=None, help="Código do grupo, por exemplo G07.")
    parser.add_argument("--cenarios-construcao", type=int, default=None, help="Mínimo 2.000; padrão usa config ou 2.000.")
    parser.add_argument("--cenarios-validacao", type=int, default=5000, help="Validação independente: mínimo 5.000.")
    parser.add_argument("--stock-bins", type=int, default=12)
    parser.add_argument("--cash-bins", type=int, default=30)
    parser.add_argument("--desconto-promocional", type=float, default=0.05)
    parser.add_argument("--uplift-demanda", type=float, default=0.10)
    parser.add_argument("--percentil-risco", type=float, default=10.0)
    return parser


if __name__ == "__main__":
    executar(construir_parser().parse_args())
