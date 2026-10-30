"""Utilitários compartilhados de desempenho, persistência e modelagem."""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np


def medir_desempenho(funcao: Callable, *args, **kwargs) -> tuple[Any, dict[str, float]]:
    """Executa uma função e mede tempo de parede e pico de memória Python.

    tracemalloc não contabiliza necessariamente toda a memória nativa usada por
    NumPy/Pandas; essa limitação é descrita na documentação do projeto.
    """
    tracemalloc.start()
    inicio = time.perf_counter()
    try:
        resultado = funcao(*args, **kwargs)
        tempo = time.perf_counter() - inicio
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return resultado, {
        "tempo_segundos": float(tempo),
        "pico_memoria_python_mb": float(pico / (1024 ** 2)),
    }


def json_seguro(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_seguro(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_seguro(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "to_dict"):
        return json_seguro(obj.to_dict(orient="records"))
    return obj


def salvar_json(caminho: Path | str, dados: Any) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(json_seguro(dados), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def criar_pasta(caminho: Path | str) -> Path:
    pasta = Path(caminho)
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def variabilidade_config(cfg: dict) -> tuple[float, float, float]:
    """Retorna desvios relativos (demanda, concorrente, logística).

    O enunciado não fixa nomes/unidades das chaves internas de variabilidade.
    Por isso, reconhecemos aliases comuns e usamos defaults documentados caso
    o gerador não forneça os fatores explicitamente.
    """
    raw = cfg.get("variabilidade", {}) or {}
    def encontrar(chaves: tuple[str, ...], padrao: float) -> float:
        for chave in chaves:
            if chave in raw:
                valor = float(raw[chave])
                # Interpretação adotada: fração (0.10 = 10%).
                return max(0.0, valor)
        return padrao
    return (
        encontrar(("demanda", "demanda_consumidor", "variacao_demanda"), 0.15),
        encontrar(("concorrente", "preco_concorrente", "movimento_concorrente"), 0.05),
        encontrar(("logistica", "custo_logistico", "atendimento"), 0.10),
    )
