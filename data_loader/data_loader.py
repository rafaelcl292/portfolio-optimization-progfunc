import pandas as pd
import yfinance as yf


def load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Baixa preços ajustados de fechamento para os tickers no período dado.
    Retorna DataFrame com datas como índice e colunas como tickers.
    """
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw["Close"] if hasattr(raw, "columns") else raw
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    return data
