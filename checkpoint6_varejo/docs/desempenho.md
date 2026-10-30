# Desempenho e otimização

As execuções oficiais gravam medições de tempo (`time.perf_counter`) e pico de memória Python (`tracemalloc`) nos arquivos `resultado_q1.json`, `resultado_q2.json` e `escalabilidade.csv`.

## Otimização demonstrada na Q1

`questao_01/main.py` mede duas implementações equivalentes da avaliação de uma política:

- **Antes:** versão de referência com iteração Python por SKU (`avaliar_politica_referencia`).
- **Depois:** versão vetorizada com NumPy/Pandas (`avaliar_politica`).

O programa verifica que o caixa produzido pelas duas versões é igual antes de registrar o speedup. Isso demonstra a troca de laço interpretado por operações vetorizadas.

## Tamanhos de entrada

- Q1: GA e Greedy são medidos em três tamanhos distintos (6 SKUs, metade do catálogo e catálogo completo, removendo tamanhos repetidos). O Brute Force também é medido em tamanhos pequenos enquanto o total de combinações permanece dentro do limite computacional configurado; tamanhos maiores são omitidos por inviabilidade exponencial.
- Q2: Monte Carlo + DP são medidos em três tamanhos de catálogo (aproximadamente 25%, 50% e 100%), com 2.000 cenários para cada medição.

Os valores numéricos só devem ser preenchidos/interpretados depois da execução com os cinco arquivos oficiais do grupo. Testes com fixtures ou amostras sintéticas não são resultados da instância oficial.

## Limitação da memória

`tracemalloc` pode não rastrear todo o uso de memória em buffers nativos das bibliotecas numéricas. Para medição do RSS total, o grupo pode acrescentar `psutil` ou ferramenta equivalente, documentando o ambiente e a metodologia. Não misturar as duas medidas como se fossem iguais.
