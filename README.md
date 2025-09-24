# FinBERT Stock Price Prediction

This project integrates **FinBERT** (finance-specific BERT) with **ATR (Average True Range)**-based volatility signals to predict short-term stock market movements. Developed for **EECS 545 (Machine Learning)** at the University of Michigan.

---

## Features
- 📰 Use financial news (FNSPID dataset) + Yahoo Finance stock data  
- 📈 ATR-based labeling for bullish/bearish signals  
- 🤖 FinBERT embeddings + Transformer for temporal modeling  
- 📊 Backtested trading strategy with ~22% improved return vs. baseline:contentReference[oaicite:0]{index=0}  
- 🧪 Reproducible with `venv`, `pytest`, and a **one-command runner**  

---

## Quickstart

```bash
# create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
