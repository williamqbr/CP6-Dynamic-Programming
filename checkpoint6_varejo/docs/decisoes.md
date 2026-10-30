# Decisões e hipóteses de modelagem

O PDF define os algoritmos, dados e entregas, mas não determina todas as fórmulas operacionais. As hipóteses abaixo foram adotadas para que exista uma implementação executável. Elas **não devem ser apresentadas como fórmulas textualmente determinadas pelo professor**; devem ser confrontadas com o notebook/detalhes passados em aula.

## Questão 1

1. **Demanda:** `demanda_base_90d × (preço_escolhido/preço_atual)^(-abs(elasticidade))`.
2. **Unidades vendidas:** mínimo entre demanda estimada e estoque disponível. A demanda é tratada como valor esperado contínuo, sem arredondamento de unidade.
3. **Caixa operacional:** unidades vendidas × (preço após taxa de marketplace − custo logístico). O custo unitário é abatido no cálculo da margem/contribuição, não novamente do caixa operacional, pois o modelo interpreta o custo econômico do estoque como custo já incorrido. Confirmar se essa interpretação corresponde à convenção financeira esperada pelo professor.
4. **Margem unitária:** (preço × (1 − taxa marketplace) − custo logístico − custo unitário) / preço de venda. Cada SKU selecionado precisa respeitar o piso informado.
5. **Brute Force:** os primeiros seis SKUs em ordem estável de SKU formam o subconjunto; todas as combinações de níveis são avaliadas. O GA é executado no mesmo subconjunto para cálculo do gap.
6. **Algoritmo Genético:** cromossomo contém um índice de nível de preço por SKU; usa seleção por torneio, crossover de um ponto, mutação por gene e elitismo.
7. **Promoção Q1:** padrão de 5% de desconto e 10% de uplift de demanda, ambos configuráveis por CLI. O Greedy calcula benefício de contribuição por real de incentivo e aloca orçamento em ordem decrescente de ROI. O desconto é aplicado às unidades incrementais promovidas.

## Questão 2

1. **Incerteza:** demanda usa choque lognormal de média 1; preço do concorrente usa fator lognormal de média 1; logística usa `max(0, 1 + choque normal)`. Os três desvios relativos são extraídos de `variabilidade` por aliases comuns. Se o JSON não trouxer essas chaves, defaults de 15%, 5% e 10% são usados e explicitados no resultado.
2. **Elasticidade:** módulo da elasticidade própria é aplicado para reduzir demanda quando o preço próprio cresce; o aumento do preço concorrente aumenta a demanda relativa do varejista.
3. **Programação Dinâmica:** estado é (semana, fração agregada de estoque remanescente, caixa disponível); estoque e caixa são discretizados. O valor da DP aproxima a probabilidade de cumprir os compromissos com as transições amostradas pelo Monte Carlo.
4. **Reconstrução:** a tabela de política é convertida em uma trajetória fixa por meio do estado médio esperado. É essa trajetória que é encaminhada ao Greedy e validada pelo Monte Carlo independente.
5. **Promoção Q2:** padrão de 5% de desconto e 10% de uplift, configuráveis por CLI. A intensidade varia de 0 a 1. O orçamento estimado cobre o desconto sobre a venda prevista do SKU promovido; o desconto e o uplift são aplicados proporcionalmente à intensidade na simulação.
6. **Seeds:** Monte Carlo de construção utiliza `seed_final`; validação usa `seed_final + 1.000.003`. O fluxo de validação tem 5.000 cenários ou mais.
7. **Percentil de risco:** por padrão, o resultado também reporta o percentil 10 do caixa final; isso é uma escolha de apresentação, e pode ser alterada por `--percentil-risco`.

## Integridade

A verificação SHA-256 é baseada nos hashes dos quatro arquivos CSV/JSON de Q1/Q2 listados pelo manifesto. A fórmula do ID truncado usada no módulo é uma hipótese herdada do código inicial e precisa ser conferida contra o notebook oficial. A validação deve ser ajustada se a implementação real do gerador tiver outra estrutura, mantendo sempre os arquivos de entrada intactos.
