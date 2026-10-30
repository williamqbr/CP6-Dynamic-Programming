# Análise de complexidade — tempo e memória

Definições:

- `n`: número de SKUs/produtos.
- `p`: número de níveis discretos de preço.
- `k`: número de SKUs no subconjunto da busca exata.
- `P`: tamanho da população do GA.
- `G`: número de gerações.
- `S`: número de cenários Monte Carlo.
- `W`: número de semanas.
- `A`: número de ações/níveis de preço.
- `I`: número de buckets de estoque (`stock_bins`).
- `C`: número de buckets de caixa (`cash_bins`).

## Questão 1

| Algoritmo/rotina | Tempo | Memória auxiliar | Justificativa no código |
|---|---:|---:|---|
| Avaliação de política | `O(n)` | `O(n)` | `avaliar_politica` percorre vetores de SKU com operações vetorizadas; cria vetores de preço, demanda, vendas e margem. |
| Brute Force | `O(p^k × k)` | `O(k)` | `itertools.product` enumera `p^k` cromossomos; cada combinação chama a avaliação linear em `k` produtos. |
| Algoritmo Genético | `O(G × P × n)` | `O(P × n + G)` | Em cada geração são avaliados `P` indivíduos com `n` genes; população e histórico são mantidos. |
| Greedy promocional | `O(n log n)` | `O(n)` | Calcula ROI por SKU e ordena os índices uma vez; a alocação percorre a ordenação. |

**Viabilidade:** Brute Force torna-se inviável rapidamente com o crescimento de `k`, pois o número de combinações é exponencial. Por isso é utilizado apenas nos seis produtos exigidos para benchmark. Para o catálogo completo é utilizado o GA, que oferece busca aproximada com custo configurável por `P` e `G`.

## Questão 2

| Algoritmo/rotina | Tempo | Memória auxiliar | Justificativa no código |
|---|---:|---:|---|
| Geração Monte Carlo | `O(S × W × n)` | `O(S × W × n)` | Gera três arrays de fatores aleatórios com dimensões cenário × semana × SKU. |
| Avaliar ação Monte Carlo | `O(S × n)` | `O(S × n)` temporária | Calcula demanda, venda limitada pelo estoque e caixa para um par semana/ação/estado de estoque. |
| Pré-cálculo para DP | `O(W × A × I × S × n)` | `O(W × A × I × S)` | Pré-calcula vendas e caixa de cada ação para todos os estados discretos de estoque. |
| Programação Dinâmica | `O(W × I × C × A × S)` após pré-cálculo | `O(W × I × C + W × A × I × S)` | Cada estado testa todas as ações e calcula a média sobre `S` transições amostradas. |
| Greedy semanal | `O(W × n log n)` | `O(W × n)` | Para cada semana calcula o ROI e ordena SKUs para distribuir o orçamento. |
| Validação Monte Carlo | `O(S × W × n)` | `O(S × n + W × n)` | Cada cenário percorre as semanas e atualiza vendas, caixa e estoque. |

**Viabilidade:** O custo da DP cresce com buckets (`I`, `C`), ações (`A`) e cenários (`S`). A discretização do estoque e caixa limita o estado. Para catálogos muito grandes, o pré-cálculo Monte Carlo é a principal pressão de memória; pode-se reduzir o número de buckets ou processar cenários em lotes, registrando a perda/alteração de precisão.

## Medição prática

As rotinas usam `time.perf_counter()` e `tracemalloc`. `tracemalloc` mede alocações rastreadas pelo alocador Python, podendo não refletir toda a memória nativa de NumPy/Pandas. Por isso os valores são identificados como pico de memória Python, não como RSS total do processo.
