# Arquitetura e fluxo dos algoritmos

## Entrada comum

`dados.py` localiza os arquivos oficiais, lê CSV/JSON, verifica SHA-256/ID do manifesto e valida colunas, tipos, faixas e coerência básica. Se a integridade não estiver confirmada, a execução padrão é interrompida.

## Questão 1

```text
Arquivos oficiais Q1
       |
       v
Validação + integridade
       |
       v
Subconjunto determinístico de 6 SKUs
       |
       +----> Brute Force: enumera todas as combinações -> ótimo de referência
       |
       +----> GA no mesmo subconjunto -> gap contra o ótimo
       |
       v
GA no catálogo completo -> vetor de preços por SKU
       |
       v
Greedy -> ordena oportunidades por benefício marginal/orçamento
       |
       v
Resultados: caixa, margem, estoque, dívida, orçamento e desempenho
```

O Brute Force e o GA do subconjunto utilizam a mesma função de avaliação, de forma que o benchmark é comparável. O GA do catálogo completo entrega diretamente a política consumida pelo Greedy.

## Questão 2

```text
Arquivos oficiais Q2
       |
       v
Monte Carlo (>= 2.000 cenários)
       |  demanda / preço concorrente / logística
       v
Programação Dinâmica
       |  estado = semana, estoque agregado discretizado, caixa discretizado
       v
Trajetória semanal de multiplicadores de preço
       |
       v
Greedy semanal -> orçamento distribuído por SKU
       |
       v
Monte Carlo independente (>= 5.000 cenários)
       |
       v
Probabilidades de pagamento nas semanas 4, 8 e 12 + distribuição do caixa
```

## Decisão de estado na DP

A DP utiliza `semana`, `estoque remanescente` e `caixa disponível`. O estoque é agregado como fração do estoque total inicial e discretizado em `stock_bins`; o caixa é discretizado em `cash_bins`. A transição usa resultados dos cenários Monte Carlo de construção. O vetor de política é reconstruído em um caminho de estado médio para gerar uma trajetória semanal explicável, que depois é validada em novos cenários. A reutilização de subproblemas ocorre porque cada estado `(semana, estoque, caixa)` calcula seu valor uma vez na tabela `valor`; diferentes estados anteriores consultam esses mesmos estados futuros já calculados, em vez de reexplorar todas as sequências de preço.

A discretização é uma aproximação computacional. O número de buckets afeta o equilíbrio entre memória/tempo e precisão; deve ser discutido na defesa.
