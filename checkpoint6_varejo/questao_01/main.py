"""Execução integrada da Questão 1: Brute Force -> GA -> Greedy."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Permite tanto `python questao_01/main.py` quanto `python -m questao_01.main`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dados import carregar_q1
from metricas import criar_pasta, medir_desempenho, salvar_json
from questao_01.brute_force import resolver_brute_force
from questao_01.genetic_algorithm import executar_algoritmo_genetico
from questao_01.greedy import alocar_incentivos
from questao_01.modelo import avaliar_politica, dataframe_politica


def avaliar_politica_referencia(df: pd.DataFrame, cfg: dict, indices: list[int]) -> dict:
    """Versão de referência com laço Python, usada como baseline de otimização."""
    niveis = cfg["niveis_preco_relativos"]
    caixa = lucro = receita = unidades_total = 0.0
    for pos, (_, linha) in enumerate(df.iterrows()):
        preco = float(linha["preco_atual"]) * float(niveis[indices[pos]])
        demanda = float(linha["demanda_base_90d"]) * (preco / float(linha["preco_atual"])) ** (-abs(float(linha["elasticidade"])))
        unidades = min(float(linha["estoque"]), demanda)
        caixa_unit = preco * (1.0 - float(linha["taxa_marketplace"])) - float(linha["custo_logistico"])
        contribuicao_unit = caixa_unit - float(linha["custo_unitario"])
        caixa += unidades * caixa_unit
        lucro += unidades * contribuicao_unit
        receita += unidades * preco
        unidades_total += unidades
    return {"caixa": caixa, "lucro": lucro, "margem": lucro / receita if receita else 0.0, "unidades": unidades_total}


def benchmark_escalabilidade(df: pd.DataFrame, cfg: dict, indices_completos: list[int], seed: int) -> list[dict]:
    n = len(df)
    tamanhos = sorted(set([max(1, n // 4), max(1, n // 2), n]))
    registros = []
    for tamanho in tamanhos:
        sub = df.iloc[:tamanho].reset_index(drop=True)
        indices = indices_completos[:tamanho]
        ga_func = lambda: executar_algoritmo_genetico(
            sub, cfg, seed, tamanho_populacao=14, geracoes=15,
            taxa_mutacao=0.10, taxa_crossover=0.85, elitismo=2,
        )
        _, met_ga = medir_desempenho(ga_func)
        greedy_func = lambda: alocar_incentivos(sub, cfg, avaliar_politica(sub, cfg, indices))
        _, met_greedy = medir_desempenho(greedy_func)
        registros.append({
            "quantidade_skus": tamanho,
            "algoritmo_genetico_tempo_s": met_ga["tempo_segundos"],
            "algoritmo_genetico_memoria_mb": met_ga["pico_memoria_python_mb"],
            "greedy_tempo_s": met_greedy["tempo_segundos"],
            "greedy_memoria_mb": met_greedy["pico_memoria_python_mb"],
        })
    # O Brute Force é medido em 6, 7 e 8 SKUs quando a instância comporta.
    for tamanho in (6, 7, 8):
        if tamanho > n or len(cfg["niveis_preco_relativos"]) ** tamanho > 20_000:
            continue
        func = lambda t=tamanho: resolver_brute_force(df.iloc[:t].reset_index(drop=True), cfg, 20_000)
        _, met = medir_desempenho(func)
        registros.append({
            "quantidade_skus": tamanho,
            "brute_force_tempo_s": met["tempo_segundos"],
            "brute_force_memoria_mb": met["pico_memoria_python_mb"],
        })
    return registros


def executar(args: argparse.Namespace) -> dict:
    dados = carregar_q1(args.data_dir, args.grupo)
    df = dados.catalogo.sort_values("sku", kind="stable").reset_index(drop=True)
    cfg = dados.config
    seed = int(cfg["seed_final"])
    if len(df) < 6:
        raise ValueError("O catálogo oficial Q1 precisa possuir pelo menos 6 SKUs.")
    saida = criar_pasta(ROOT / "questao_01" / "resultados")

    subconjunto = df.iloc[:6].reset_index(drop=True)
    (brute, met_brute) = medir_desempenho(resolver_brute_force, subconjunto, cfg)

    ga_a_func = lambda: executar_algoritmo_genetico(
        subconjunto, cfg, seed, tamanho_populacao=35, geracoes=50,
        taxa_mutacao=0.08, taxa_crossover=0.85, elitismo=2,
    )
    ga_a, met_ga_a = medir_desempenho(ga_a_func)
    ga_b_func = lambda: executar_algoritmo_genetico(
        subconjunto, cfg, seed, tamanho_populacao=50, geracoes=70,
        taxa_mutacao=0.12, taxa_crossover=0.90, elitismo=3,
    )
    ga_b, met_ga_b = medir_desempenho(ga_b_func)

    tabela_comparacao = pd.DataFrame([
        {"metodo": "Brute Force", "caixa": brute["avaliacao"]["caixa_operacional"], "viavel": brute["avaliacao"]["viavel"], "gap_percentual_vs_brute_force": 0.0, **met_brute},
        {"metodo": "GA configuração A", "caixa": ga_a["avaliacao"]["caixa_operacional"], "viavel": ga_a["viavel"], "gap_percentual_vs_brute_force": (brute["avaliacao"]["caixa_operacional"] - ga_a["avaliacao"]["caixa_operacional"]) / max(abs(brute["avaliacao"]["caixa_operacional"]), 1e-9) * 100.0, **met_ga_a},
        {"metodo": "GA configuração B", "caixa": ga_b["avaliacao"]["caixa_operacional"], "viavel": ga_b["viavel"], "gap_percentual_vs_brute_force": (brute["avaliacao"]["caixa_operacional"] - ga_b["avaliacao"]["caixa_operacional"]) / max(abs(brute["avaliacao"]["caixa_operacional"]), 1e-9) * 100.0, **met_ga_b},
    ])
    tabela_comparacao.to_csv(saida / "brute_force_vs_ga.csv", index=False, encoding="utf-8-sig")

    # O GA completo recebe como inicialização aleatória a mesma seed oficial.
    ga_completo_func = lambda: executar_algoritmo_genetico(
        df, cfg, seed, tamanho_populacao=50, geracoes=70,
        taxa_mutacao=0.12, taxa_crossover=0.90, elitismo=3,
    )
    ga_completo, met_ga_completo = medir_desempenho(ga_completo_func)
    if not ga_completo["viavel"]:
        raise RuntimeError(
            "O GA não encontrou política viável para o catálogo completo. "
            "Revise níveis de preço/margem e aumente população/gerações."
        )

    avaliacao_ga = ga_completo["avaliacao"]
    avaliacao_obj = avaliar_politica(df, cfg, ga_completo["indices_preco"])
    greedy_func = lambda: alocar_incentivos(
        df, cfg, avaliacao_obj, args.desconto_promocional, args.uplift_demanda
    )
    greedy, met_greedy = medir_desempenho(greedy_func)
    greedy["tabela"].to_csv(saida / "greedy_alocacao.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"etapa": "Antes do Greedy", "caixa_operacional": greedy["caixa_antes_greedy"], "orcamento_utilizado": 0.0},
        {"etapa": "Depois do Greedy", "caixa_operacional": greedy["caixa_depois_greedy"], "orcamento_utilizado": greedy["orcamento_utilizado"]},
    ]).to_csv(saida / "greedy_antes_depois.csv", index=False, encoding="utf-8-sig")

    # Evidência de otimização: laço Python de referência versus cálculo vetorizado.
    indices = ga_completo["indices_preco"]
    ref, met_ref = medir_desempenho(avaliar_politica_referencia, df, cfg, indices)
    vet, met_vet = medir_desempenho(avaliar_politica, df, cfg, indices)
    if abs(ref["caixa"] - vet.caixa_operacional) > 1e-6:
        raise AssertionError("Versão vetorizada e versão de referência produziram caixas diferentes.")
    otimizacao = {
        "antes_laco_python": met_ref,
        "depois_vetorizado": met_vet,
        "speedup_tempo": met_ref["tempo_segundos"] / max(met_vet["tempo_segundos"], 1e-12),
        "nota": "Comparação de uma implementação de referência por SKU com a implementação vetorizada NumPy/Pandas.",
    }

    historico = pd.DataFrame(ga_completo["historico"])
    historico.to_csv(saida / "fitness_por_geracao.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 5))
    plt.plot(historico["geracao"], historico["melhor_fitness"], label="Melhor fitness")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Convergência do Algoritmo Genético — Questão 1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(saida / "fitness_por_geracao.png", dpi=160)
    plt.close()

    politica_df = dataframe_politica(df, avaliacao_obj)
    politica_df.to_csv(saida / "politica_precos_ga.csv", index=False, encoding="utf-8-sig")
    escala = benchmark_escalabilidade(df, cfg, indices, seed)
    pd.DataFrame(escala).to_csv(saida / "escalabilidade.csv", index=False, encoding="utf-8-sig")

    resumo = {
        "grupo": dados.grupo,
        "integridade": {"status": dados.integridade.status, "resumo": dados.integridade.resumo(), "detalhes": dados.integridade.detalhes},
        "seed_algoritmos": seed,
        "algoritmos_utilizados": 3,
        "fluxo": "Brute Force valida o GA em 6 SKUs; GA otimiza o catálogo completo; Greedy recebe a política de preços do GA e aloca incentivos.",
        "brute_force_subconjunto": brute,
        "ga_configuracao_a_subconjunto": ga_a,
        "ga_configuracao_b_subconjunto": ga_b,
        "ga_catalogo_completo": ga_completo,
        "greedy": {k: v for k, v in greedy.items() if k != "tabela"},
        "resultado_final": {
            "caixa_antes_greedy": greedy["caixa_antes_greedy"],
            "caixa_depois_greedy": greedy["caixa_depois_greedy"],
            "margem_percentual_antes_incentivos": avaliacao_obj.margem_percentual,
            "margem_percentual_depois_incentivos": greedy["margem_percentual_depois"],
            "lucro_contribuicao_depois_incentivos": greedy["lucro_contribuicao_depois"],
            "estoque_remanescente_antes_incentivos": avaliacao_obj.estoque_remanescente,
            "estoque_remanescente_depois_incentivos": greedy["estoque_remanescente_depois"],
            "cobertura_divida_depois_incentivos": greedy["cobertura_divida_depois"],
            "meta_atingida_depois_incentivos": greedy["atingiu_meta_de_caixa_depois"],
            "meta_caixa_90d": cfg["meta_caixa_90d"],
            "meta_atingida_antes_incentivos": avaliacao_obj.atingiu_meta,
            "folga_meta_antes_incentivos": avaliacao_obj.folga_meta,
            "divida_total_90d": cfg["divida_total_90d"],
            "orcamento_promocional_utilizado": greedy["orcamento_utilizado"],
        },
        "desempenho": {
            "brute_force": met_brute,
            "ga_a_subconjunto": met_ga_a,
            "ga_b_subconjunto": met_ga_b,
            "ga_catalogo_completo": met_ga_completo,
            "greedy": met_greedy,
            "otimizacao": otimizacao,
            "escalabilidade": escala,
        },
        "hipoteses_modelagem": {
            "demanda": "demanda_base_90d * (preco_escolhido/preco_atual)^(-abs(elasticidade)), limitada pelo estoque.",
            "caixa_operacional": "unidades * (preço após marketplace - custo logístico); custo unitário é considerado no cálculo da margem, não subtraído novamente do caixa operacional.",
            "margem": "(preço após marketplace - logística - custo unitário) / preço de venda.",
            "promocao": f"Desconto {args.desconto_promocional:.1%} e uplift de demanda {args.uplift_demanda:.1%}, ambos ajustáveis por CLI.",
        },
    }
    salvar_json(saida / "resultado_q1.json", resumo)
    print(f"Q1 concluída. Resultados em: {saida}")
    print(f"Caixa estimado após Greedy: {greedy['caixa_depois_greedy']:.2f}")
    print(f"Meta de caixa: {cfg['meta_caixa_90d']:.2f} | atingida antes dos incentivos: {avaliacao_obj.atingiu_meta}")
    return resumo


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint 6 — Questão 1")
    parser.add_argument("--data-dir", default="data", help="Pasta com arquivos oficiais do notebook.")
    parser.add_argument("--grupo", default=None, help="Código do grupo, por exemplo G07.")
    parser.add_argument("--desconto-promocional", type=float, default=0.05, help="Desconto promocional adotado (padrão 5%%).")
    parser.add_argument("--uplift-demanda", type=float, default=0.10, help="Aumento estimado de demanda da promoção (padrão 10%%).")
    return parser


if __name__ == "__main__":
    executar(construir_parser().parse_args())
