import numpy as np

from questao_01.brute_force import resolver_brute_force
from questao_01.genetic_algorithm import executar_algoritmo_genetico
from questao_01.greedy import alocar_incentivos
from questao_01.modelo import avaliar_politica
from questao_02.dynamic_programming import resolver_programacao_dinamica
from questao_02.greedy import alocar_promocoes_semanais
from questao_02.monte_carlo import gerar_cenarios, simular_politica


def test_brute_force_e_ga(q1_data, q1_config):
    subset = q1_data.iloc[:6].reset_index(drop=True)
    brute = resolver_brute_force(subset, q1_config)
    ga = executar_algoritmo_genetico(
        subset, q1_config, seed=123, tamanho_populacao=20,
        geracoes=12, taxa_mutacao=0.1, taxa_crossover=0.8, elitismo=2,
    )
    assert brute["combinacoes_avaliadas"] == 4 ** 6
    assert brute["avaliacao"]["viavel"] is True
    assert len(ga["historico"]) == 12
    assert len(ga["indices_preco"]) == 6


def test_greedy_q1_respeita_orcamento(q1_data, q1_config):
    indices = [2] * len(q1_data)
    avaliacao = avaliar_politica(q1_data, q1_config, indices)
    resultado = alocar_incentivos(q1_data, q1_config, avaliacao)
    assert resultado["orcamento_utilizado"] <= q1_config["orcamento_promocional"] + 1e-8
    assert resultado["orcamento_utilizado"] >= 0
    assert len(resultado["tabela"]) == len(q1_data)


def test_monte_carlo_dp_greedy_validacao(q2_data, q2_config):
    cenarios = gerar_cenarios(q2_config, 2000, len(q2_data), seed=456)
    assert cenarios.fator_demanda.shape == (2000, 12, len(q2_data))
    dp = resolver_programacao_dinamica(q2_data, q2_config, cenarios, stock_bins=6, cash_bins=12)
    caminho = dp["caminho_precos_multiplicadores"]
    assert len(caminho) == 12
    greedy = alocar_promocoes_semanais(q2_data, q2_config, caminho)
    assert greedy["intensidade_promocional"].shape == (12, len(q2_data))
    cenarios_validacao = gerar_cenarios(q2_config, 5000, len(q2_data), seed=987654)
    validacao = simular_politica(
        q2_data, q2_config, caminho, cenarios_validacao,
        promocoes=greedy["intensidade_promocional"],
    )
    assert validacao["quantidade_cenarios"] == 5000
    assert 0 <= validacao["probabilidade_pagar_todas"] <= 1
    assert np.isfinite(validacao["caixa_final_esperado"])
