import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


@pytest.fixture
def q1_data():
    return pd.DataFrame({
        "sku": [f"SKU{i:03d}" for i in range(1, 9)],
        "categoria": ["geral"] * 8,
        "custo_unitario": [40.0, 45.0, 50.0, 55.0, 60.0, 35.0, 48.0, 52.0],
        "preco_atual": [100.0, 110.0, 120.0, 130.0, 140.0, 90.0, 115.0, 125.0],
        "estoque": [100.0] * 8,
        "demanda_base_90d": [70.0, 65.0, 60.0, 55.0, 50.0, 80.0, 63.0, 58.0],
        "elasticidade": [1.2] * 8,
        "taxa_marketplace": [0.10] * 8,
        "custo_logistico": [5.0] * 8,
        "idade_estoque_dias": [30.0] * 8,
    })


@pytest.fixture
def q1_config():
    return {
        "grupo": "G99", "seed_base": 19, "seed_final": 99019,
        "horizonte_dias": 90, "meta_caixa_90d": 10000.0,
        "divida_total_90d": 8000.0, "orcamento_promocional": 500.0,
        "margem_minima_percentual": 0.10,
        "niveis_preco_relativos": [0.90, 1.0, 1.10, 1.20],
    }


@pytest.fixture
def q2_data():
    return pd.DataFrame({
        "sku": [f"SKU{i:03d}" for i in range(1, 7)],
        "categoria": ["geral"] * 6,
        "custo_unitario": [40.0, 45.0, 50.0, 55.0, 60.0, 35.0],
        "preco_atual": [100.0, 110.0, 120.0, 130.0, 140.0, 90.0],
        "estoque_inicial": [100.0] * 6,
        "demanda_base_semanal": [10.0, 9.0, 8.0, 7.0, 6.0, 11.0],
        "elasticidade": [1.2] * 6,
        "taxa_marketplace": [0.10] * 6,
        "custo_logistico_base": [5.0] * 6,
        "preco_concorrente_base": [105.0, 115.0, 125.0, 135.0, 145.0, 95.0],
    })


@pytest.fixture
def q2_config():
    return {
        "grupo": "G99", "seed_base": 19, "seed_final": 99019,
        "numero_semanas": 12, "caixa_inicial": 1000.0,
        "dividas": {"semana_4": 100.0, "semana_8": 100.0, "semana_12": 100.0},
        "variabilidade": {"demanda": 0.15, "concorrente": 0.05, "logistica": 0.10},
        "orcamento_promocional_semanal": 100.0,
        "niveis_preco_relativos": [0.90, 1.0, 1.10],
        "n_simulacoes_recomendadas": 2000,
        "limite_risco_falha": 0.20,
    }
