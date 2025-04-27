# Otimização de Carteiras com Programação Funcional

Este projeto em Python implementa uma simulação e otimização de carteiras de ações do índice Dow Jones, com objetivo de maximizar o Sharpe Ratio. Utiliza princípios de programação funcional e paralelismo para explorar eficientemente grandes combinações de ativos sob as seguintes restrições:
  - Soma dos pesos igual a 1
  - Posição apenas comprada (long-only)
  - Peso máximo por ativo definido

## Índice
1. [Recursos](#recursos)
2. [Tecnologias](#tecnologias)
3. [Pré-requisitos](#prérequisitos)
4. [Instalação](#instalação)
5. [Uso](#uso)
6. [Configuração](#configuração)
7. [Estrutura do Projeto](#estrutura-do-projeto)
8. [Exemplo de Saída](#exemplo-de-saída)
9. [Contribuição](#contribuição)

## Recursos
- Otimização de carteiras via simulação Monte Carlo
- Cálculo de retornos, volatilidade e métricas financeiras com funções puras
- Paralelismo para aceleração com `concurrent.futures.ProcessPoolExecutor`
- Reprodutibilidade garantida por seed configurável

## Tecnologias
- Python 3.13+
- yfinance (download de dados históricos)
- concurrent.futures (paralelismo)
- uv (gerenciador de pacotes)

## Pré-requisitos
- Git
- Python 3.13 ou superior
- uv (gerenciador de pacotes Python)

## Instalação
```bash
git clone https://github.com/rafaelcl292/portfolio-optimization-progfunc.git
cd portfolio-optimization-progfunc
uv sync
```

## Uso
Após a instalação, execute o script principal com as opções desejadas:

- Otimização padrão:
  ```bash
  uv run python main.py --select 25 --samples 1000
  ```
- Reprodutibilidade (seed fixa):
  ```bash
  uv run python main.py --select 25 --samples 1000 --seed 42
  ```
- Benchmark (comparar execução serial vs. paralelo):
  ```bash
  uv run python main.py --benchmark 5
  ```

## Configuração
- Periodo de análise (01/08/2024 a 31/12/2024) e lista de tickers: ajuste em `main.py`
- Parâmetros de execução:
  - `--select`: número de ativos na carteira
  - `--samples`: número de simulações por combinação
  - `--free-rate`: taxa livre de risco (padrão 0.0)
  - `--seed`: semente para reprodutibilidade
  - `--benchmark`: quantidade de execuções para benchmark

## Estrutura do Projeto
- **data_loader/**: download e carregamento de dados históricos via yfinance
- **utils/**: cálculos de retornos, volatilidade e métricas financeiras
- **simulate/**: geração de carteiras e cálculo de Sharpe Ratio
- **main.py**: orquestra todo o pipeline de otimização

## Exemplo de Saída
```bash
$ uv run python main.py --select 25 --samples 1000
Carregando dados para 30 tickers de 2024-08-01 a 2024-12-31...
Iniciando otimização com 142506 combinações, 1000 simulações cada, r_free=0.00
Processando 100% | █████████████████ | 142506 combos em 16s
Melhor Sharpe Ratio: 3.1585
Carteira Ótima:
  Tickers: [AAPL, AXP, BA, …]
  Pesos:   [0.010, 0.003, 0.011, …]
```
