import pandas as pd
import pytest

from dados import ErroDeDados, validar_catalogo_q1, validar_config_q1


def test_catalogo_q1_aceita_estoque_zero(q1_data):
    q1_data.loc[0, "estoque"] = 0
    validado = validar_catalogo_q1(q1_data)
    assert validado.loc[0, "estoque"] == 0


def test_catalogo_q1_rejeita_preco_negativo(q1_data):
    q1_data.loc[0, "preco_atual"] = -2
    with pytest.raises(ErroDeDados):
        validar_catalogo_q1(q1_data)


def test_config_q1_rejeita_margem_invalida(q1_config):
    q1_config["margem_minima_percentual"] = 1.2
    with pytest.raises(ErroDeDados):
        validar_config_q1(q1_config)
