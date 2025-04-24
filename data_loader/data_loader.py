import pandas as pd
import yfinance as yf


def load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Baixa preços ajustados de fechamento para os tickers no período dado.
    Retorna DataFrame com datas como índice e colunas como tickers.
    """
    # Download de dados via yfinance
    # Baixar preços ajustados (auto_adjust=True retorna preços ajustados em 'Close')
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    # Extrair coluna de fechamento ajustado
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        # se for DataFrame simples ou Series
        data = raw["Close"] if hasattr(raw, "columns") else raw
    # Converter Series em DataFrame se necessário
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # Remover linhas sem dados
    data = data.dropna(how="all")
    return data
