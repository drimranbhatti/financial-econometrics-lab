"""Refresh the data snapshots in data/ (Yahoo Finance + FRED, both free, no key)."""
import io, pathlib, requests, pandas as pd, yfinance as yf

OUT = pathlib.Path(__file__).resolve().parents[1] / "data"
TICKERS = {"GLD": "gold_etf", "C40.PA": "cac40_etf", "BTC-USD": "bitcoin", "^GSPC": "sp500"}
FRED = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]

def main(start="2018-01-01", end="2026-10-31"):
    OUT.mkdir(exist_ok=True)
    px = yf.download(list(TICKERS), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px.index.name = "date"
    px.rename(columns=TICKERS).round(4).to_csv(OUT / "prices.csv")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(FRED) + f"&cosd={start}&coed={end}"
    y = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text), index_col=0, parse_dates=True, na_values=".")
    y.index.name = "date"
    y = y.loc[start:end]
    y.dropna(how="all").to_csv(OUT / "treasury_yields.csv")
    print("saved", px.shape, y.shape)

if __name__ == "__main__":
    main()
