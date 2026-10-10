# 🔬 Quant Efficiency Lab

Quant Efficiency Lab is an open-source quantitative analysis engine built with Python and Streamlit. It is designed to evaluate the true efficiency of mutual funds and ETFs, cutting through marketing narratives to reveal whether an active manager is truly generating alpha or simply charging high fees for a closet-indexed portfolio.

**Live Demo:** Quant_Efficiency_Lab

## 🌟 Key Features

* **🤖 Returns-Based Style Analysis (RBSA):** The engine mathematically deconstructs a fund's historical returns against standard proxies (Global Equity, Bonds, Cash) to build a highly correlated "Shadow ETF". This reveals the fund's true DNA and prevents managers from hiding behind easy-to-beat benchmarks.

* **👨‍💼 Investor Dashboard (Intuitive):** Translates complex econometrics into tangible retail metrics:

  * **Index Overlap:** Detects "closet indexers".

  * **Active Fee Proxy:** Calculates the real cost you are paying for the *active* portion of the portfolio.

  * **The Airbag Test:** Measures Downside Capture during actual market crashes.

  * **3-Year Rolling Win Rate:** Evaluates the long-term consistency of the manager's edge.

* **🔬 Quant Laboratory (Statistical):** Provides raw statistical decomposition, Pure Alpha calculations, and monthly return linear regression scatter plots.
  

## ⚠️ Limitations & Best Practices

This engine relies on linear regression and constrained Returns-Based Style Analysis (RBSA). To get accurate verdicts, the inputs must align with the model's fundamental assumptions.

* **Currency and Regional Alignment:**
  The engine compares the fund's returns to the selected proxies. If you analyze a European UCITS ETF priced in EUR, you **must** use EUR-denominated proxies (e.g., `EUNL.DE`, `DBZB.DE`, `XEON.DE`). Mixing a EUR fund with USD proxies will introduce severe currency exchange noise, completely invalidating the Alpha and Tracking Error calculations.

* **Testing Active Factor Funds (e.g., Small Cap Value, Quality):**
  Using the "Shadow ETF (RBSA)" mode against a global broad market proxy (like `VT`) will penalize pure factor funds if their specific factor is out of favor in the current macroeconomic regime. To test the true *manager skill* of a factor fund (e.g., `AVUV`), switch to **Single Benchmark (Direct)** and test it against a passive index ETF of the exact same category (e.g., `VBR` or `IJS`).

* **Capital Efficiency and Leveraged Portfolios:**
  The RBSA optimizer uses a long-only constraint (`weights >= 0` and `sum = 100%`). It cannot mathematically replicate funds with implicit leverage or return stacking (e.g., a 90/60 portfolio like `NTSX` or `NTSG`). The engine will classify the extra volatility from the bond leverage as unrewarded active risk, resulting in a poor score.

* **Non-Linear and Convex Strategies:**
  Metrics like Information Ratio and R-Squared assume symmetrical, linear market exposure. Strategies designed to be highly convex—such as Managed Futures, Trend-Following, or Tail Risk hedging (e.g., `DBMF`, `TAIL`)—will break the linear regression model. These instruments generate returns precisely during extreme market dislocations, making standard statistical decomposition ineffective.

## ⚖️ Disclaimer

*This tool is for educational and informational purposes only. It does not constitute financial advice. Historical performance is not indicative of future results. Always do your own research or consult a certified financial advisor before making investment decisions.*
