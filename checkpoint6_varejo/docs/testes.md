# Testes automatizados

Executar a partir da raiz:

```bash
pytest -q
```

Os testes cobrem:

- validação de estoque zero e rejeição de preço negativo/configuração inválida;
- enumeração Brute Force no subconjunto reduzido;
- execução reprodutível do GA em pequena instância;
- limite de orçamento no Greedy da Q1;
- geração de pelo menos 2.000 cenários, DP, Greedy e validação com 5.000 cenários na Q2.

A execução de desenvolvimento desta versão retornou `6 passed`. Os testes utilizam fixtures sintéticas definidas em `tests/conftest.py`; não leem nem alteram os dados oficiais.

Além dos testes unitários, as duas rotinas `main.py` foram executadas de ponta a ponta em uma pasta temporária com dados sintéticos e manifesto de teste válido. Isso confirmou a conexão entre os módulos e a geração dos artefatos, mas **não** é uma execução oficial nem fornece números para a entrega do grupo.
