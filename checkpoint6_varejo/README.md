# Checkpoint 6 — Recuperação financeira de varejistas por precificação online

**FIAP | Dynamic Programming**  
**Integrantes**

| Nome | RM |
|---|---:|
| Kauê de Almeida Pena | 564211 |
| Eduardo Delorenzo Moraes | 561749 |
| Ronaldo Aparecido Monteiro Almeida | 565017 |
| William Queiroz | 565032 |

**Seed base calculada a partir dos RMs:** `1 + 9 + 7 + 2 = 19`.  
**Código do grupo:** preencher com o código efetivamente atribuído pelo professor/notebook oficial (`GXX`).  
**Seed final:** `1000 × número_do_grupo + 19`, obtida exclusivamente da configuração gerada pelo notebook. Não escolher manualmente.

## 1. Objetivo e escopo

O projeto implementa as duas cadeias algorítmicas pedidas no enunciado:

- **Questão 1:** Brute Force valida uma instância reduzida; Algoritmo Genético busca preços para o catálogo completo; Greedy distribui o orçamento promocional usando a política de preços do GA.
- **Questão 2:** Monte Carlo estima resultados sob incerteza; Programação Dinâmica escolhe uma trajetória semanal; Greedy distribui o orçamento promocional semanal; um segundo Monte Carlo, com seed diferente, valida o plano.

Os arquivos de entrada oficiais não são criados ou modificados por este projeto. Devem vir do notebook oficial da avaliação.

## 2. Estrutura

```text
checkpoint6_varejo/
├── README.md
├── requirements.txt
├── IA_USO.md
├── dados.py
├── metricas.py
├── data/
│   └── LEIA-ME.md
├── questao_01/
│   ├── brute_force.py
│   ├── genetic_algorithm.py
│   ├── greedy.py
│   ├── modelo.py
│   ├── main.py
│   └── resultados/
├── questao_02/
│   ├── monte_carlo.py
│   ├── dynamic_programming.py
│   ├── greedy.py
│   ├── main.py
│   └── resultados/
├── tests/
└── docs/
    ├── arquitetura.png
    ├── arquitetura.md
    ├── complexidade.md
    ├── desempenho.md
    ├── decisoes.md
    └── testes.md
```

## 3. Preparação do ambiente

Requisito: Python 3.10 ou superior.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Arquivos oficiais de entrada

Execute o notebook oficial integralmente, alterando somente o Bloco 02 (código do grupo e RMs). Copie para `data/` os cinco arquivos gerados, sem editar seu conteúdo:

```text
q1_catalogo_GXX.csv
q1_config_GXX.json
q2_catalogo_GXX.csv
q2_config_GXX.json
manifesto_dados.json
```

Substitua `GXX` pelo código real do grupo. O arquivo `data/LEIA-ME.md` contém a lista. Não use os fixtures de `tests/` como dados da avaliação.

**Importante sobre o manifesto:** o enunciado exige hashes SHA-256 e ID da instância, mas não especifica a fórmula do ID nem a chave JSON que contém os hashes. `dados.py` aceita algumas formas comuns e implementa provisoriamente a fórmula que estava no módulo recebido (hash da concatenação dos hashes ordenados pelo nome, truncado em 12 caracteres). Antes da execução oficial, compare isso com o notebook gerador. Se a fórmula/chave do notebook for diferente, ajuste apenas a leitura/verificação em `dados.py`, nunca os arquivos oficiais.

## 5. Execução

Na raiz do repositório:

```bash
python questao_01/main.py --data-dir data --grupo GXX
python questao_02/main.py --data-dir data --grupo GXX
```

Também funciona com `python -m questao_01.main` e `python -m questao_02.main`.

Parâmetros principais:

```bash
python questao_01/main.py --help
python questao_02/main.py --help
```

A Questão 1 utiliza a `seed_final` oficial para a aleatoriedade do GA. A Questão 2 usa a mesma seed para os cenários de construção e `seed_final + 1.000.003` para gerar cenários independentes de validação. Não passe uma seed inventada no lugar da seed oficial.

## 6. Resultados produzidos

### Questão 1 — `questao_01/resultados/`

- `resultado_q1.json`: resumo, metas, caixa, margem, estoque, cobertura da dívida, orçamento e medições.
- `brute_force_vs_ga.csv`: comparação entre ótimo exato e duas configurações do GA no subconjunto de seis SKUs.
- `fitness_por_geracao.csv` e `.png`: convergência do GA no catálogo completo.
- `greedy_alocacao.csv` e `greedy_antes_depois.csv`: alocação do orçamento e comparação antes/depois.
- `politica_precos_ga.csv`: política final por SKU.
- `escalabilidade.csv`: medições para tamanhos distintos de catálogo.

### Questão 2 — `questao_02/resultados/`

- `resultado_q2.json`: resumo de risco, caixa, margem, estoque e desempenho.
- `caminho_dp.csv`: trajetória semanal reconstruída pela DP.
- `trajetoria_preco_estoque_skus.csv`: preços e estoques estimados por semana/SKU.
- `greedy_resumo_semanal.csv`: alocação semanal do orçamento promocional.
- `distribuicao_caixa_final.csv` e `.png`: distribuição do caixa na validação independente.
- `probabilidade_pagamentos.csv`: probabilidade de pagar nas semanas 4, 8 e 12.
- `escalabilidade.csv`: comparação com três tamanhos de entrada.

## 7. Testes

```bash
pytest -q
```

Os testes usam dados sintéticos criados em `tests/conftest.py`. Eles verificam as funções e não substituem uma execução com os arquivos oficiais. O projeto também interrompe a execução se a integridade dos arquivos não for confirmada.

## 8. Premissas que devem ser conhecidas na defesa

O PDF não define a fórmula exata de demanda, margem/caixa, intensidade do incentivo, unidades da variabilidade ou discretização da DP. Para tornar o programa executável, foram adotadas as hipóteses registradas em `docs/decisoes.md`. Elas devem ser comparadas com qualquer orientação complementar do professor/notebook e explicadas pelo grupo. Não apresente resultados sintéticos como resultados oficiais.

## 9. Defesa técnica

O grupo deve conseguir explicar, alterar por parâmetro e executar novamente:

- preços relativos e piso de margem;
- população, gerações, mutação, crossover e elitismo do GA;
- orçamento e retorno marginal do Greedy;
- número de cenários, seed e três fontes de incerteza;
- estado/transição da DP e discretização de caixa/estoque;
- Big O, medições reais e trade-offs de tempo/memória.

O histórico Git deve refletir o trabalho real do grupo, com commits incrementais e autoria verdadeira. Não crie commits retroativos para simular histórico.
