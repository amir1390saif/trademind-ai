# TradeMind AI (prototype)

## Install
```bash
python -m pip install streamlit pandas numpy
```

## Run
```bash
streamlit run trademind_ai.py
```

Open the local URL shown by Streamlit.

## CSV format
Required columns: Open, High, Low, Close
Optional: Date, Volume

This is an educational analysis prototype only. It does not connect to a broker,
place orders, or provide personalized financial advice. Historical tests are
simplified and exclude fees, slippage, taxes, and market-specific constraints.
