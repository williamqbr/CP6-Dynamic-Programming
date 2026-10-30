"""Leitura, validação e integridade dos arquivos oficiais do Checkpoint 6.

Este módulo não altera os arquivos gerados pelo notebook oficial. A fórmula do
ID da instância reproduz a hipótese presente no módulo recebido pelo grupo;
ela deve ser confirmada com o notebook oficial antes da entrega final.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


class ErroDeDados(ValueError):
    """Erro de conteúdo, estrutura ou integridade dos arquivos de entrada."""


COLUNAS_Q1 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque",
    "demanda_base_90d", "elasticidade", "taxa_marketplace",
    "custo_logistico", "idade_estoque_dias",
]
COLUNAS_Q2 = [
    "sku", "categoria", "custo_unitario", "preco_atual", "estoque_inicial",
    "demanda_base_semanal", "elasticidade", "taxa_marketplace",
    "custo_logistico_base", "preco_concorrente_base",
]
CHAVES_CONFIG_Q1 = [
    "grupo", "seed_base", "seed_final", "horizonte_dias", "meta_caixa_90d",
    "divida_total_90d", "orcamento_promocional", "margem_minima_percentual",
    "niveis_preco_relativos",
]
CHAVES_CONFIG_Q2 = [
    "grupo", "seed_base", "seed_final", "numero_semanas", "caixa_inicial",
    "dividas", "variabilidade", "orcamento_promocional_semanal",
    "niveis_preco_relativos", "n_simulacoes_recomendadas", "limite_risco_falha",
]


def sha256_arquivo(caminho: Path | str) -> str:
    """Calcula SHA-256 em blocos, sem carregar o arquivo inteiro na memória."""
    h = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def _sha256_normalizado(caminho: Path | str) -> str:
    """Hash alternativo tolerante a CRLF do Git no Windows."""
    conteudo = Path(caminho).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(conteudo).hexdigest()


def _numero_finito(valor: Any) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(float(valor))


@dataclass
class Integridade:
    grupo: str
    seed_base: int | None
    seed_final: int | None
    id_instancia_manifesto: str | None
    id_instancia_recalculado: str | None
    status: str  # ok | ok_normalizado | divergente | sem_manifesto
    detalhes: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "ok_normalizado")

    def resumo(self) -> str:
        return (
            f"Grupo: {self.grupo} | seed_base: {self.seed_base} | "
            f"seed_final: {self.seed_final} | ID manifesto: "
            f"{self.id_instancia_manifesto} | ID recalculado: "
            f"{self.id_instancia_recalculado} | integridade: {self.status}"
        )


def localizar_grupo(data_dir: Path | str, grupo: str | None = None) -> str:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ErroDeDados(f"Pasta de dados não encontrada: {data_dir}")
    achados = sorted(
        p.stem.removeprefix("q1_catalogo_")
        for p in data_dir.glob("q1_catalogo_*.csv")
        if re.fullmatch(r"q1_catalogo_.+\.csv", p.name)
    )
    if grupo:
        if grupo not in achados:
            raise ErroDeDados(f"Arquivos do grupo {grupo!r} não encontrados. Presentes: {achados or 'nenhum'}.")
        return grupo
    if not achados:
        raise ErroDeDados(
            f"Nenhum q1_catalogo_<GRUPO>.csv em {data_dir}. Copie os quatro arquivos oficiais "
            "e manifesto_dados.json para data/."
        )
    if len(achados) > 1:
        raise ErroDeDados(f"Mais de um grupo em {data_dir}: {achados}. Informe --grupo.")
    return achados[0]


def _caminhos(data_dir: Path, grupo: str) -> dict[str, Path]:
    return {
        "q1_catalogo": data_dir / f"q1_catalogo_{grupo}.csv",
        "q1_config": data_dir / f"q1_config_{grupo}.json",
        "q2_catalogo": data_dir / f"q2_catalogo_{grupo}.csv",
        "q2_config": data_dir / f"q2_config_{grupo}.json",
        "manifesto": data_dir / "manifesto_dados.json",
    }


def _hashes_manifesto(manifesto: dict) -> dict[str, str]:
    """Aceita os nomes usuais; a estrutura exata deve ser confirmada no notebook."""
    hashes = manifesto.get("arquivos") or manifesto.get("hashes_sha256") or manifesto.get("hashes") or {}
    if not isinstance(hashes, dict):
        raise ErroDeDados("Manifesto: seção de hashes deve ser um objeto JSON.")
    normalizados: dict[str, str] = {}
    for nome, valor in hashes.items():
        if isinstance(valor, dict):
            valor = valor.get("sha256") or valor.get("hash")
        if isinstance(valor, str):
            normalizados[Path(nome).name] = valor.lower()
    return normalizados


def verificar_integridade(data_dir: Path | str, grupo: str) -> Integridade:
    data_dir = Path(data_dir)
    caminhos = _caminhos(data_dir, grupo)
    for chave in ("q1_catalogo", "q1_config", "q2_catalogo", "q2_config"):
        if not caminhos[chave].is_file():
            raise ErroDeDados(f"Arquivo oficial ausente: {caminhos[chave]}")
    if not caminhos["manifesto"].is_file():
        return Integridade(grupo, None, None, None, None, "sem_manifesto", {"manifesto": "ausente"})

    try:
        manifesto = json.loads(caminhos["manifesto"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ErroDeDados(f"Manifesto JSON inválido: {exc}") from exc
    if not isinstance(manifesto, dict):
        raise ErroDeDados("Manifesto deve conter um objeto JSON.")

    esperados = _hashes_manifesto(manifesto)
    detalhes: dict[str, str] = {}
    recalculados: dict[str, str] = {}
    usou_normalizacao = False
    for chave in ("q1_catalogo", "q1_config", "q2_catalogo", "q2_config"):
        caminho = caminhos[chave]
        bruto = sha256_arquivo(caminho)
        esperado = esperados.get(caminho.name)
        if esperado is None:
            detalhes[caminho.name] = "não consta na seção de hashes do manifesto"
            recalculados[caminho.name] = bruto
        elif bruto == esperado:
            detalhes[caminho.name] = "ok"
            recalculados[caminho.name] = bruto
        elif _sha256_normalizado(caminho) == esperado:
            detalhes[caminho.name] = "ok após normalizar CRLF para LF"
            recalculados[caminho.name] = esperado
            usou_normalizacao = True
        else:
            detalhes[caminho.name] = "HASH DIVERGENTE: arquivo não corresponde ao manifesto"
            recalculados[caminho.name] = bruto

    # Hipótese herdada do módulo recebido: hashes concatenados por nome e 12 hex.
    id_recalculado = hashlib.sha256(
        "".join(recalculados[nome] for nome in sorted(recalculados)).encode("utf-8")
    ).hexdigest()[:12]
    id_manifesto = manifesto.get("id_instancia") or manifesto.get("instance_id")
    if not id_manifesto:
        detalhes["id_instancia"] = "ID ausente no manifesto"
    elif str(id_manifesto).lower() != id_recalculado:
        detalhes["id_instancia"] = "ID divergente (confirmar fórmula com o notebook oficial)"

    divergencia = any(
        "DIVERGENTE" in valor or "não consta" in valor for valor in detalhes.values()
    ) or (not id_manifesto or str(id_manifesto).lower() != id_recalculado)
    if divergencia:
        status = "divergente"
    elif usou_normalizacao:
        status = "ok_normalizado"
    else:
        status = "ok"

    return Integridade(
        grupo=str(manifesto.get("grupo", grupo)),
        seed_base=manifesto.get("seed_base"),
        seed_final=manifesto.get("seed_final"),
        id_instancia_manifesto=str(id_manifesto) if id_manifesto else None,
        id_instancia_recalculado=id_recalculado,
        status=status,
        detalhes=detalhes,
    )


def exigir_integridade(integridade: Integridade) -> None:
    if not integridade.ok:
        raise ErroDeDados(
            "Integridade não confirmada. A execução foi interrompida para evitar usar "
            f"dados divergentes. {integridade.resumo()} Detalhes: {integridade.detalhes}"
        )


def _exigir_colunas(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    faltam = [coluna for coluna in colunas if coluna not in df.columns]
    if faltam:
        raise ErroDeDados(f"{nome}: colunas ausentes: {faltam}.")


def _validar_numericas(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    for coluna in colunas:
        valores = pd.to_numeric(df[coluna], errors="coerce")
        if valores.isna().any():
            raise ErroDeDados(f"{nome}: '{coluna}' contém valores vazios ou não numéricos.")
        if not valores.map(math.isfinite).all():
            raise ErroDeDados(f"{nome}: '{coluna}' contém valores infinitos.")
        df[coluna] = valores.astype(float)


def _validar_nao_negativos(df: pd.DataFrame, colunas: list[str], nome: str) -> None:
    for coluna in colunas:
        if (df[coluna] < 0).any():
            raise ErroDeDados(f"{nome}: '{coluna}' deve ser maior ou igual a zero.")


def _validar_catalogo_comum(df: pd.DataFrame, nome: str) -> None:
    if df.empty:
        raise ErroDeDados(f"{nome}: catálogo vazio.")
    if df["sku"].isna().any() or df["sku"].astype(str).str.strip().eq("").any():
        raise ErroDeDados(f"{nome}: SKU vazio.")
    if df["sku"].duplicated().any():
        duplicados = df.loc[df["sku"].duplicated(), "sku"].astype(str).tolist()
        raise ErroDeDados(f"{nome}: SKU duplicado: {duplicados[:10]}.")
    if df["categoria"].isna().any():
        raise ErroDeDados(f"{nome}: categoria vazia.")
    if not df["taxa_marketplace"].between(0, 1, inclusive="left").all():
        raise ErroDeDados(f"{nome}: taxa_marketplace deve estar em [0, 1).")


def validar_catalogo_q1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "idade_estoque" in df.columns and "idade_estoque_dias" not in df.columns:
        df = df.rename(columns={"idade_estoque": "idade_estoque_dias"})
    _exigir_colunas(df, COLUNAS_Q1, "Catálogo Q1")
    _validar_numericas(df, [
        "custo_unitario", "preco_atual", "estoque", "demanda_base_90d",
        "elasticidade", "taxa_marketplace", "custo_logistico", "idade_estoque_dias",
    ], "Catálogo Q1")
    _validar_catalogo_comum(df, "Catálogo Q1")
    if (df[["custo_unitario", "preco_atual"]] <= 0).any().any():
        raise ErroDeDados("Catálogo Q1: custo_unitario e preco_atual devem ser positivos.")
    _validar_nao_negativos(
        df, ["estoque", "demanda_base_90d", "elasticidade", "custo_logistico", "idade_estoque_dias"],
        "Catálogo Q1",
    )
    return df.reset_index(drop=True)


def validar_catalogo_q2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _exigir_colunas(df, COLUNAS_Q2, "Catálogo Q2")
    _validar_numericas(df, [
        "custo_unitario", "preco_atual", "estoque_inicial", "demanda_base_semanal",
        "elasticidade", "taxa_marketplace", "custo_logistico_base", "preco_concorrente_base",
    ], "Catálogo Q2")
    _validar_catalogo_comum(df, "Catálogo Q2")
    if (df[["custo_unitario", "preco_atual", "preco_concorrente_base"]] <= 0).any().any():
        raise ErroDeDados("Catálogo Q2: custos e preços de referência devem ser positivos.")
    _validar_nao_negativos(
        df, ["estoque_inicial", "demanda_base_semanal", "elasticidade", "custo_logistico_base"],
        "Catálogo Q2",
    )
    return df.reset_index(drop=True)


def _validar_niveis(niveis: Any, nome: str) -> None:
    if not isinstance(niveis, list) or not niveis:
        raise ErroDeDados(f"{nome}: niveis_preco_relativos deve ser uma lista não vazia.")
    if any(not _numero_finito(x) or x <= 0 for x in niveis):
        raise ErroDeDados(f"{nome}: níveis de preço devem ser números finitos positivos.")


def _validar_numero_config(cfg: dict, chave: str, nome: str, minimo: float | None = None) -> None:
    valor = cfg.get(chave)
    if not _numero_finito(valor) or (minimo is not None and valor < minimo):
        regra = f" maior ou igual a {minimo}" if minimo is not None else " finito"
        raise ErroDeDados(f"{nome}: '{chave}' deve ser um número{regra}.")


def validar_config_q1(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ErroDeDados("Config Q1 deve ser um objeto JSON.")
    faltam = [chave for chave in CHAVES_CONFIG_Q1 if chave not in cfg]
    if faltam:
        raise ErroDeDados(f"Config Q1: chaves ausentes {faltam}.")
    _validar_niveis(cfg["niveis_preco_relativos"], "Config Q1")
    for chave in ("seed_base", "seed_final", "horizonte_dias", "meta_caixa_90d", "divida_total_90d", "orcamento_promocional"):
        _validar_numero_config(cfg, chave, "Config Q1", 0)
    if cfg["horizonte_dias"] <= 0:
        raise ErroDeDados("Config Q1: horizonte_dias deve ser positivo.")
    margem = cfg["margem_minima_percentual"]
    if not _numero_finito(margem) or not 0 <= margem < 1:
        raise ErroDeDados("Config Q1: margem_minima_percentual deve estar em [0, 1).")
    return cfg


def validar_config_q2(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ErroDeDados("Config Q2 deve ser um objeto JSON.")
    faltam = [chave for chave in CHAVES_CONFIG_Q2 if chave not in cfg]
    if faltam:
        raise ErroDeDados(f"Config Q2: chaves ausentes {faltam}.")
    _validar_niveis(cfg["niveis_preco_relativos"], "Config Q2")
    for chave in ("seed_base", "seed_final", "numero_semanas", "caixa_inicial", "orcamento_promocional_semanal", "n_simulacoes_recomendadas"):
        _validar_numero_config(cfg, chave, "Config Q2", 0)
    if int(cfg["numero_semanas"]) < 12:
        raise ErroDeDados("Config Q2: numero_semanas deve cobrir ao menos as 12 semanas do enunciado.")
    if int(cfg["n_simulacoes_recomendadas"]) < 1:
        raise ErroDeDados("Config Q2: n_simulacoes_recomendadas deve ser >= 1.")
    if not isinstance(cfg["dividas"], dict) or not cfg["dividas"]:
        raise ErroDeDados("Config Q2: 'dividas' deve ser um dicionário não vazio.")
    for chave, valor in cfg["dividas"].items():
        if not re.fullmatch(r"semana_\d+", str(chave)) or not _numero_finito(valor) or valor < 0:
            raise ErroDeDados(f"Config Q2: dívida inválida {chave}={valor}.")
        if int(str(chave).split("_")[1]) > int(cfg["numero_semanas"]):
            raise ErroDeDados(f"Config Q2: {chave} excede o horizonte.")
    for semana in (4, 8, 12):
        if f"semana_{semana}" not in cfg["dividas"]:
            raise ErroDeDados(f"Config Q2: compromisso semana_{semana} não encontrado.")
    if not isinstance(cfg["variabilidade"], dict):
        raise ErroDeDados("Config Q2: variabilidade deve ser um objeto JSON.")
    for chave, valor in cfg["variabilidade"].items():
        if not _numero_finito(valor) or valor < 0:
            raise ErroDeDados(f"Config Q2: variabilidade '{chave}' inválida.")
    risco = cfg["limite_risco_falha"]
    if not _numero_finito(risco) or not 0 < risco < 1:
        raise ErroDeDados("Config Q2: limite_risco_falha deve estar em (0, 1).")
    return cfg


@dataclass
class DadosOficiais:
    grupo: str
    catalogo: pd.DataFrame
    config: dict
    integridade: Integridade


def _ler(data_dir: Path, grupo: str, questao: str) -> tuple[pd.DataFrame, dict]:
    caminhos = _caminhos(data_dir, grupo)
    try:
        df = pd.read_csv(caminhos[f"{questao}_catalogo"], encoding="utf-8-sig", dtype={"sku": "string"})
    except pd.errors.EmptyDataError as exc:
        raise ErroDeDados(f"Catálogo {questao.upper()} vazio ou inválido.") from exc
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ErroDeDados(f"Não foi possível ler o CSV {questao.upper()}: {exc}") from exc
    try:
        cfg = json.loads(caminhos[f"{questao}_config"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ErroDeDados(f"Config {questao.upper()} inválida: {exc}") from exc
    return df, cfg


def _validar_manifesto_config(grupo: str, cfg: dict, integridade: Integridade) -> None:
    if integridade.grupo != grupo:
        raise ErroDeDados("Código do grupo no manifesto diverge do nome dos arquivos.")
    if cfg.get("grupo") != grupo:
        raise ErroDeDados("Código do grupo na configuração diverge do nome dos arquivos.")
    if integridade.seed_base is not None and cfg.get("seed_base") != integridade.seed_base:
        raise ErroDeDados("seed_base da configuração diverge do manifesto.")
    if integridade.seed_final is not None and cfg.get("seed_final") != integridade.seed_final:
        raise ErroDeDados("seed_final da configuração diverge do manifesto.")


def carregar_q1(data_dir: Path | str = "data", grupo: str | None = None, permitir_sem_integridade: bool = False) -> DadosOficiais:
    data_dir = Path(data_dir)
    grupo = localizar_grupo(data_dir, grupo)
    integridade = verificar_integridade(data_dir, grupo)
    if not permitir_sem_integridade:
        exigir_integridade(integridade)
    df, cfg = _ler(data_dir, grupo, "q1")
    cfg = validar_config_q1(cfg)
    _validar_manifesto_config(grupo, cfg, integridade)
    return DadosOficiais(grupo, validar_catalogo_q1(df), cfg, integridade)


def carregar_q2(data_dir: Path | str = "data", grupo: str | None = None, permitir_sem_integridade: bool = False) -> DadosOficiais:
    data_dir = Path(data_dir)
    grupo = localizar_grupo(data_dir, grupo)
    integridade = verificar_integridade(data_dir, grupo)
    if not permitir_sem_integridade:
        exigir_integridade(integridade)
    df, cfg = _ler(data_dir, grupo, "q2")
    cfg = validar_config_q2(cfg)
    _validar_manifesto_config(grupo, cfg, integridade)
    return DadosOficiais(grupo, validar_catalogo_q2(df), cfg, integridade)


def validar_identidade_q1_q2(q1: DadosOficiais, q2: DadosOficiais) -> None:
    if q1.grupo != q2.grupo:
        raise ErroDeDados("Q1 e Q2 pertencem a grupos diferentes.")
    for chave in ("grupo", "seed_base", "seed_final"):
        if q1.config.get(chave) != q2.config.get(chave):
            raise ErroDeDados(f"Q1 e Q2 possuem valores diferentes em '{chave}'.")
    for dados in (q1, q2):
        integ = dados.integridade
        if integ.grupo != dados.grupo:
            raise ErroDeDados("Grupo do manifesto diverge do grupo dos arquivos.")
        if integ.seed_base is not None and integ.seed_base != dados.config["seed_base"]:
            raise ErroDeDados("seed_base da configuração diverge do manifesto.")
        if integ.seed_final is not None and integ.seed_final != dados.config["seed_final"]:
            raise ErroDeDados("seed_final da configuração diverge do manifesto.")


def carregar_json(caminho: Path | str) -> dict:
    try:
        conteudo = json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ErroDeDados(f"JSON inválido em {caminho}: {exc}") from exc
    if not isinstance(conteudo, dict):
        raise ErroDeDados(f"JSON em {caminho} deve conter um objeto.")
    return conteudo
