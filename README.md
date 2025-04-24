# Otimização de Carteiras com Programação Funcional

Projeto em Python que simula e otimiza carteiras de ações do índice Dow Jones, maximizando o Sharpe Ratio sob restrições de soma de pesos, posição long-only e peso máximo por ativo. O código faz uso de programação funcional e paralelismo para processar grandes combinações de ativos.

## Estrutura do Projeto

- data_loader/: módulo para download e carregamento de dados históricos via yfinance
- utils/: funções puras para cálculo de retornos e volatilidade
- simulate/: lógica de simulação de carteiras e cálculo de Sharpe Ratio
- main.py: script principal que executa todo o pipeline
  
## Pré-requisitos
- Python 3.13 ou superior
- uv (Python package manager) 

## Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/rafaelcl292/portfolio-optimization-progfunc.git
   cd otimiza-carteiras
   ```
2. Inicialize o projeto e instale dependências:
   ```bash
   uv sync
   ```

## Uso
- Para executar a otimização:
  ```bash
  uv run python main.py
  ```

## Configurações
- Período de análise (start/end) e lista de tickers definidos em `main.py`.
- Número de ativos selecionados (25 de 30), número de simulações (1000 por combinação) e taxa livre de risco também podem ser ajustados ali.

## Extras
- Paralelismo via `concurrent.futures.ProcessPoolExecutor` para acelerar as simulações.
