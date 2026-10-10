import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.optimize import minimize
import requests

def resolve_ticker(query):
    """Uses Yahoo Finance internal search to translate ISIN to Ticker"""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            return data['quotes'][0]['symbol']
    except:
        pass
    return query 

st.set_page_config(page_title="Quant Efficiency Lab", layout="wide", page_icon="🔬")

# --- CORE MATH & RBSA ---
def calc_drawdown(wealth_index):
    peak = wealth_index.cummax()
    return (wealth_index - peak) / peak

def style_analysis(fund_rets, factors_rets):
    num_factors = factors_rets.shape[1]
    init_weights = np.ones(num_factors) / num_factors
    bounds = tuple((0, 1) for _ in range(num_factors))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    def objective(w):
        tracking_error = fund_rets - np.dot(factors_rets, w)
        return np.sum(tracking_error**2)
        
    res = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def get_retail_status(metric_name, value):
    if metric_name == "Fotocopia": 
        if value > 0.85: return "🔴", "Closet Indexer", "It's a clone. Tracks the index but charges active management fees."
        if value > 0.60: return "🟡", "Hybrid Management", "Deviates partially, but the core engine remains the base index."
        return "🟢", "Autonomous Fund", "Makes truly distinct choices, justifying active management."
    
    if metric_name == "Active_Fee":
        if value > 0.04: return "🔴", "Disproportionate Cost", "Requires unrealistic skill to recover this fee drag."
        if value > 0.025: return "🟡", "High Burden", "The active bet is expensive."
        return "🟢", "Fair Pricing", "You are paying a fair price for the active risk taken."
        
    if metric_name == "IR":
        if value < 0: return "🔴", "Value Destructor", "Higher volatility than the index and lower returns."
        if value < 0.5: return "🟡", "Inefficient", "Active risk is not adequately rewarded."
        return "🟢", "Efficient", "The active bet pays off significantly."
        
    if metric_name == "Airbag":
        if value > 0.95: return "🔴", "No Airbag", "When it rains, you get just as wet as with a passive ETF."
        if value > 0.80: return "🟡", "Weak Protection", "Only partially cushions market crashes."
        return "🟢", "True Protection", "Anti-crisis fund, sharply cuts market drawdowns."
        
    if metric_name == "P_Value":
        if value > 0.10: return "🔴", "Pure Luck", "No mathematical proof that the manager can consistently beat the market."
        if value <= 0.05: return "🟢", "Solid Skill", "Excellent statistical significance."
        return "🟡", "Gray Zone", "Positive results, but the doubt of luck persists."
        
    if metric_name == "Consistency":
        if value is None: return "⚪", "Not Enough Data", "Requires at least 3 years of history to calculate rolling win rate."
        if value > 0.75: return "🟢", "Consistent Winner", "Reliably beats the benchmark across rolling 3-year periods."
        if value < 0.40: return "🔴", "Chronic Underperformer", "Holding this for 3 years historically results in losing to the index."
        return "🟡", "Coin Flip", "Chances of beating the index over 3 years are near 50/50."

def get_global_score(ir, p_value, down_cap, active_fee, r_squared, ter, win_rate):
    # --- PASSIVE ETF BYPASS ---
    if r_squared > 0.95 and ter <= 0.003:
        return 85.0, "P", "🔵 PASSIVE ETF: Honest and efficient tracker. Excellent as a core portfolio engine."

    score = 50.0  
    
    # 1. Base Engine: Information Ratio (Extra ritorno per rischio attivo)
    score += ir * 25 
    
    # 2. Skill or Luck?
    if p_value > 0.10: score -= 10
    elif p_value <= 0.05: score += 10
    
    # 3. The Airbag Test
    if down_cap > 1.0: score -= 10
    elif down_cap < 0.80: score += 10
    
    # 4. Costo dell'implementazione
    if active_fee > 0.04: 
        score -= 15 # Penalità pesante, ma superabile dalle performance
    
    # 5. Consistency Test (Il premio al lungo termine)
    if win_rate is not None:
        if win_rate >= 0.70: score += 15
        elif win_rate < 0.40: score -= 15
    
    score = max(0, min(100, score))
    
    if score >= 80: return score, "S", "🟢 OUTSTANDING: Rare excellence. Fully justifies the fees."
    if score >= 65: return score, "A", "🟢 GREAT: Generates real, solid, and statistically valid value."
    if score >= 50: return score, "B", "🟡 FAIR: Does its job, but consider if a passive ETF is cheaper."
    if score >= 35: return score, "C", "🟠 MEDIOCRE: Inefficient. You are paying for results mostly dictated by chance."
    if score >= 20: return score, "D", "🔴 POOR: Destroys value, costs too much, and doesn't protect in crashes."
    return score, "F", "☠️ TOXIC: Sell immediately. Capital bleed guaranteed."

def analyze_instrument(fund_prices, bench_data_df, ter, mode):
    df = pd.concat([fund_prices, bench_data_df], axis=1).dropna()
    if len(df) < 252: return None
    
    f_rets = df.iloc[:, 0].pct_change().dropna()
    bench_rets_raw = df.iloc[:, 1:].pct_change().dropna()
    
    if mode == "RBSA":
        f_rets_weekly = df.iloc[:, 0].resample('W').last().pct_change().dropna()
        bench_rets_weekly = df.iloc[:, 1:].resample('W').last().pct_change().dropna()
        
        weights = style_analysis(f_rets_weekly.values, bench_rets_weekly.values)
        synthetic_bench_rets = np.dot(bench_rets_raw.values, weights)
        weight_dict = {col: w for col, w in zip(bench_data_df.columns, weights)}
        
        dynamic_rf_rate_daily = bench_rets_raw.iloc[:, 2].mean()
    else:
        synthetic_bench_rets = bench_rets_raw.iloc[:, 0].values
        weights = np.array([1.0])
        weight_dict = {bench_data_df.columns[0]: 1.0}
        
        dynamic_rf_rate_daily = 0.02 / 252

    synthetic_prices = (1 + synthetic_bench_rets).cumprod() * df.iloc[0, 1] 
    
    ann_ret = (df.iloc[-1, 0] / df.iloc[0, 0]) ** (252 / len(df)) - 1
    b_ann_ret = (synthetic_prices[-1] / synthetic_prices[0]) ** (252 / len(synthetic_prices)) - 1
    
    excess_f = f_rets - dynamic_rf_rate_daily
    excess_b = synthetic_bench_rets - dynamic_rf_rate_daily
    
    slope, intercept, r_value, _, _ = stats.linregress(excess_b, excess_f)
    alpha_ann = intercept * 252
    r_squared = r_value**2
    
    active_fee = ter / (1 - r_squared) if (1 - r_squared) > 0.01 else 9.99 
    active_return = f_rets - synthetic_bench_rets
    te = active_return.std() * np.sqrt(252)
    ir = (ann_ret - b_ann_ret) / te if te > 0 else 0

    # Calcolo Core Metrics
    volatility = f_rets.std() * np.sqrt(252)
    sharpe_ratio = (ann_ret - (dynamic_rf_rate_daily * 252)) / volatility if volatility > 0 else 0
    f_wealth_index = (1 + f_rets).cumprod()
    max_dd = calc_drawdown(f_wealth_index).min()

    n = len(excess_b)
    resid = excess_f - (intercept + slope * excess_b)
    s2_e = np.sum(resid**2) / (n - 2) if n > 2 else 0
    se_alpha = np.sqrt(s2_e * (1/n + np.mean(excess_b)**2 / np.sum((excess_b - np.mean(excess_b))**2))) if s2_e > 0 else 0
    t_stat_alpha = intercept / se_alpha if se_alpha > 0 else 0
    p_value_alpha = 2 * (1 - stats.t.cdf(np.abs(t_stat_alpha), df=n-2)) if n > 2 else 1.0
    
    # ... (tutto il codice precedente rimane uguale fino al calcolo del down_cap)
    up_mask = bench_rets_raw.iloc[:, 0] > 0
    down_mask = bench_rets_raw.iloc[:, 0] < 0
    down_cap = (f_rets[down_mask].mean() / bench_rets_raw.iloc[:, 0][down_mask].mean()) if bench_rets_raw.iloc[:, 0][down_mask].mean() < 0 else 0
    
    # --- SPOSTATO QUI: 3-YEAR ROLLING WIN RATE ---
    f_monthly = f_rets.resample('ME').apply(lambda x: (1+x).prod()-1)
    b_series = pd.Series(synthetic_bench_rets, index=f_rets.index)
    b_monthly = b_series.resample('ME').apply(lambda x: (1+x).prod()-1)
    
    rolling_window = 36
    if len(f_monthly) >= rolling_window:
        f_roll = f_monthly.rolling(window=rolling_window).apply(lambda x: (1+x).prod()-1)
        b_roll = b_monthly.rolling(window=rolling_window).apply(lambda x: (1+x).prod()-1)
        rolling_alpha = (f_roll - b_roll).dropna()
        win_rate = (rolling_alpha > 0).mean()
    else:
        win_rate = None
    
    # --- ORA CHIAMIAMO LO SCORE PASSANDO IL WIN RATE ---
    score_num, score_letter, score_verdict = get_global_score(ir, p_value_alpha, down_cap, active_fee, r_squared, ter, win_rate)
    return {
        "Grade Letter": score_letter, "Grade Num": score_num, "Global Verdict": score_verdict,
        "Annualized Return": ann_ret, "TER": ter,
        "Volatility": volatility, "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_dd,
        "RBSA Weights": weight_dict,
        "R² vs Benchmark": r_squared, "Active Fee Proxy": active_fee,
        "Alpha (Ann)": alpha_ann, "Info Ratio": ir,
        "P-Value": p_value_alpha, "Down Capture": down_cap,
        "Win Rate": win_rate,
        "Analysis Start": df.index[0].strftime('%Y-%m-%d'),
        "F_Rets": f_rets, 
        "B_Rets": pd.Series(synthetic_bench_rets, index=f_rets.index)
    }

# --- UI ---
st.title("🔬 Quant Efficiency Lab")

with st.sidebar:
    st.header("⚙️ Analysis Engine")
    
    currency_preset = st.selectbox("Currency / Region Setup", ["🇺🇸 USD (Default - US/Global)", "🇪🇺 EUR (European UCITS)"])
    if "USD" in currency_preset:
        def_eq, def_bd, def_cs = "VT", "AGG", "BIL"
    else:
        def_eq, def_bd, def_cs = "EUNL.DE", "DBZB.DE", "XEON.DE"
        
    analysis_mode = st.radio("Comparison Method:", ["🤖 Shadow ETF (RBSA)", "🎯 Single Benchmark (Direct)"])
    
    if "RBSA" in analysis_mode:
        st.caption("The engine deconstructs the fund and builds a synthetic benchmark.")
        st.info("ℹ️ **RBSA Constraints:** The model uses constrained optimization to build a realistic, long-only shadow portfolio (weights must be ≥ 0 and sum exactly to 100%).")
        eq_ticker = st.text_input("Equity Proxy", def_eq)
        bd_ticker = st.text_input("Bond Proxy", def_bd)
        cs_ticker = st.text_input("Cash Proxy", def_cs)
        target_bench_tickers = [eq_ticker.strip().upper(), bd_ticker.strip().upper(), cs_ticker.strip().upper()]
    else:
        st.caption("1-vs-1 Match: Fund against your chosen Index.")
        single_bench = st.text_input("Direct Benchmark Ticker", def_eq)
        target_bench_tickers = [single_bench.strip().upper()]

    st.caption("📉 *Risk-Free rate is dynamically calculated based on the selected Cash Proxy to reflect different market regimes.*")
    
    st.divider()
    st.header("📦 Instruments to Evaluate")
    df_input = pd.DataFrame({
        "Ticker": ["FCNTX", "ARKK", "AMCPX"],
        "TER (%)": [0.77, 0.75, 0.74]
    })
    edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

if st.button("Run X-Ray Analysis", type="primary", use_container_width=True):
    valid_tickers = edited_df.dropna(subset=["Ticker"]).copy()
    raw_inputs = valid_tickers["Ticker"].str.strip().tolist()
    
    with st.spinner("Translating ISINs, aligning dates, and running quantitative analysis..."):
        try:
            fund_mapping = {raw: resolve_ticker(raw) for raw in raw_inputs}
            bench_tickers = [resolve_ticker(t) for t in target_bench_tickers]
            
            all_t = list(set(list(fund_mapping.values()) + bench_tickers))
            names_dict = {}
            df_dict = {}
            
            for t in all_t:
                temp = yf.download(t, period="max", progress=False)
                if not temp.empty:
                    names_dict[t] = yf.Ticker(t).info.get("longName", t)
                    if isinstance(temp.columns, pd.MultiIndex):
                        temp.columns = temp.columns.droplevel(1)
                        
                    if temp.index.tz is not None:
                        temp.index = temp.index.tz_localize(None)
                    temp.index = temp.index.normalize()
                        
                    if "Adj Close" in temp.columns:
                        df_dict[t] = temp["Adj Close"].dropna()
                    elif "Close" in temp.columns:
                        df_dict[t] = temp["Close"].dropna()
                        
            if not df_dict:
                st.error("❌ No data found. Yahoo Finance is unresponsive or tickers are invalid.")
                st.stop()
                
            data = pd.DataFrame(df_dict)
            
            missing_bench = [t for t in bench_tickers if t not in data.columns]
            if missing_bench:
                st.error(f"❌ Cannot analyze: Yahoo Finance missing benchmark data for {', '.join(missing_bench)}")
                st.stop()
                
            bench_data_df = data[bench_tickers].dropna()
            if "RBSA" in analysis_mode:
                bench_data_df.columns = ["Global Equity", "Bonds", "Cash / Liquidity"]
                
            results = {}
            skipped_funds = []
            
            for raw_input, row_ter in zip(raw_inputs, valid_tickers["TER (%)"]):
                t_resolved = fund_mapping[raw_input]
                ter = row_ter / 100 if pd.notna(row_ter) else 0.0
                
                if t_resolved not in data.columns:
                    skipped_funds.append(f"**{raw_input}**: ISIN/Ticker not found in Yahoo Finance database.")
                    continue
                    
                m = analyze_instrument(data[t_resolved], bench_data_df, ter, mode="RBSA" if "RBSA" in analysis_mode else "Direct")
                
                if m:
                    results[raw_input] = m
                    names_dict[raw_input] = names_dict.get(t_resolved, t_resolved)
                else:
                    skipped_funds.append(f"**{raw_input}**: Time series too short or fragmented (min 1 year required).")
                    
            if skipped_funds:
                st.warning("⚠️ **Some instruments were skipped:**\n" + "\n".join([f"- {sf}" for sf in skipped_funds]))
                
            if not results:
                st.error("No valid instruments for analysis.")
            else:
                st.session_state.results = results
                st.session_state.names = names_dict
                st.session_state.mode = analysis_mode
                
        except Exception as e:
            st.error(f"Critical Error: {str(e)}")

if "results" in st.session_state:
    res = st.session_state.results
    names = st.session_state.names
    mode_used = st.session_state.get("mode", "🤖 Shadow ETF (RBSA)")
    
    st.divider()
    selected_t = st.selectbox("Select the instrument to inspect:", list(res.keys()), format_func=lambda x: f"{names.get(x, x)} ({x})")
    f_data = res[selected_t]
    
    tab_retail, tab_quant = st.tabs(["👨‍💼 Investor Dashboard (Intuitive)", "🔬 Quant Laboratory (Statistical)"])
    
    with tab_retail:
        st.markdown(f"## Global Score: **{f_data['Grade Letter']}** ({f_data['Grade Num']:.0f}/100)")
        
        box_color = "#1e3a8a" if f_data['Grade Letter'] == "P" else "#052e16" if f_data['Grade Num'] >= 65 else "#451a03" if f_data['Grade Num'] >= 35 else "#450a0a"
        text_color = "#60a5fa" if f_data['Grade Letter'] == "P" else "#34d399" if f_data['Grade Num'] >= 65 else "#fbbf24" if f_data['Grade Num'] >= 35 else "#f87171"
        
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; border: 1px solid {text_color}; margin-bottom: 25px;">
            <h4 style="color: {text_color}; margin: 0;">{f_data['Global Verdict']}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(f"Real-World Performance Profile")
        bench_desc = "its real 'Shadow ETF'" if "RBSA" in mode_used else "your selected benchmark"
        st.markdown(f"Historical data validated since **{f_data['Analysis Start']}**. The fund was analyzed against {bench_desc}.")
        
        # --- BLOCCO CORE METRICS ---
        st.markdown("---")
        st.markdown("### 📊 Absolute Performance")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Annual Return", f"{f_data['Annualized Return']*100:.1f}%")
        col_m2.metric("Volatility (Ann)", f"{f_data['Volatility']*100:.1f}%")
        col_m3.metric("Sharpe Ratio", f"{f_data['Sharpe Ratio']:.2f}")
        col_m4.metric("Max Drawdown", f"{f_data['Max Drawdown']*100:.1f}%")
        
        if len(f_data['F_Rets']) < 252 * 3:
            st.warning("⚠️ **Short Track Record:** The instrument has less than 3 years of historical data. Statistical metrics like P-Value and Rolling Win Rate may not be fully reliable and could be driven by market noise.")
        
        if "RBSA" in mode_used:
            st.markdown("---")
            st.markdown("### 🧬 Fund DNA (Under the Hood)")
            w_eq = f_data["RBSA Weights"].get("Global Equity", 0)
            w_bd = f_data["RBSA Weights"].get("Bonds", 0)
            w_cs = f_data["RBSA Weights"].get("Cash / Liquidity", 0)
            
            col_w1, col_w2, col_w3 = st.columns(3)
            col_w1.metric("📈 Equity Weight", f"{w_eq*100:.1f}%")
            col_w2.metric("🛡️ Bond Weight", f"{w_bd*100:.1f}%")
            col_w3.metric("💶 Cash / Liquidity", f"{w_cs*100:.1f}%")
            
            if w_eq >= 0.75:
                st.warning(f"⚠ **WARNING:** This instrument is **{w_eq*100:.0f}%** driven by Global Equities. Do not be fooled by thematic narratives: if global stock markets crash, this fund will crash with them.")
            elif w_eq <= 0.30:
                st.success("✅ **UNCORRELATED:** This instrument operates with an engine profoundly different from standard equities.")
                
        st.markdown("---")
        # ROW 1 OF METRICS
        c1, c2, c3 = st.columns(3)
        
        icon, title, desc = get_retail_status("Fotocopia", f_data["R² vs Benchmark"])
        with c1:
            st.markdown(f"### {icon} Index Overlap")
            st.markdown(f"**How much is this fund copying the market instead of actively managing it?**")
            st.metric("Correlation to Index", f"{f_data['R² vs Benchmark']*100:.1f}%")
            st.markdown(f"> **Verdict:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("Active_Fee", f_data["Active Fee Proxy"])
        with c2:
            st.markdown(f"### {icon} Active Fee Proxy")
            st.markdown(f"**On the truly active portion, what hidden commission rate are you paying?**")
            st.metric("Estimated Active Fee", f"{f_data['Active Fee Proxy']*100:.1f}%" if f_data['Active Fee Proxy'] < 9 else "> 10%")
            st.markdown(f"> **Verdict:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("Airbag", f_data["Down Capture"])
        with c3:
            st.markdown(f"### {icon} The Airbag Test")
            st.markdown(f"**During market sell-offs, does the fund avoid the crash or fall with everyone else?**")
            st.metric("Drawdown Capture", f"{f_data['Down Capture']*100:.1f}%")
            st.markdown(f"> **Verdict:** {title}. {desc}")

        st.markdown("---")
        # ROW 2 OF METRICS
        c4, c5, c6 = st.columns(3)
        
        icon, title, desc = get_retail_status("IR", f_data["Info Ratio"])
        with c4:
            st.markdown(f"### {icon} Efficiency Index")
            st.markdown(f"**For every sleepless night caused by active risk, how much extra return was generated?**")
            st.metric("Information Ratio", f"{f_data['Info Ratio']:.2f}")
            st.markdown(f"> **Verdict:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("P_Value", f_data["P-Value"])
        with c5:
            st.markdown(f"### {icon} Skill or Luck?")
            st.markdown(f"**Probability that the positive performance is just a fortunate coincidence.**")
            st.metric("Luck Risk (P-Value)", f"{f_data['P-Value']*100:.1f}%")
            st.markdown(f"> **Verdict:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("Consistency", f_data["Win Rate"])
        with c6:
            st.markdown(f"### {icon} 3Y Consistency")
            st.markdown(f"**How often does it beat the market over a 3-year holding period?**")
            val_str = f"{f_data['Win Rate']*100:.1f}%" if f_data["Win Rate"] is not None else "N/A"
            st.metric("Rolling Win Rate", val_str)
            st.markdown(f"> **Verdict:** {title}. {desc}")

        st.markdown("---")
        st.markdown("### 🎢 Compounding Path & Drawdowns")
        
        f_wealth = (1 + f_data["F_Rets"]).cumprod()
        b_wealth = (1 + f_data["B_Rets"]).cumprod()
        
        f_dd = calc_drawdown(f_wealth)
        b_dd = calc_drawdown(b_wealth)
        
        fig_wealth = go.Figure()
        fig_wealth.add_trace(go.Scatter(x=f_wealth.index, y=f_wealth, mode='lines', name='Fund', line=dict(color='#34d399', width=2)))
        fig_wealth.add_trace(go.Scatter(x=b_wealth.index, y=b_wealth, mode='lines', name='Shadow Benchmark', line=dict(color='#9ca3af', width=2, dash='dot')))
        fig_wealth.update_layout(template="plotly_dark", height=400, title="Growth of 1 Unit", margin=dict(t=40, b=10, l=10, r=10), yaxis_tickformat=".2f")
        st.plotly_chart(fig_wealth, use_container_width=True)
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=f_dd.index, y=f_dd, mode='lines', fill='tozeroy', name='Fund Drawdown', line=dict(color='#f87171', width=1), fillcolor='rgba(248, 113, 113, 0.3)'))
        fig_dd.add_trace(go.Scatter(x=b_dd.index, y=b_dd, mode='lines', name='Benchmark Drawdown', line=dict(color='#9ca3af', width=1, dash='dot')))
        fig_dd.update_layout(template="plotly_dark", height=300, title="Underwater Chart (Drawdowns)", margin=dict(t=40, b=10, l=10, r=10), yaxis_tickformat=".1%")
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.markdown("### 🔍 The Spread (Relative Wealth)")
        st.markdown("This graph shows the fund's compounding growth divided by the benchmark's growth. **Pay attention to the slope, not the height.**")
        st.markdown("- 📈 **Rising:** The fund is actively creating extra value.\n- 📉 **Falling:** The fund is bleeding alpha.\n- ➖ **Flat (even if high up):** The fund has stopped adding value and is just mirroring the benchmark, cruising on past success.")
        
        relative_wealth = f_wealth / b_wealth
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=relative_wealth.index, 
            y=relative_wealth, 
            mode='lines', 
            name='Relative Edge', 
            line=dict(color='#8b5cf6', width=2)
        ))
        fig_spread.update_layout(
            template="plotly_dark", height=300, 
            margin=dict(t=10, b=10, l=10, r=10), 
            yaxis_tickformat=".2f"
        )
        fig_spread.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig_spread, use_container_width=True)

    with tab_quant:
        st.subheader("Statistical Decomposition and Pure Alpha")
        c_q1, c_q2 = st.columns([1, 2])
        with c_q1:
            st.markdown("#### Benchmark Composition")
            for k, v in f_data["RBSA Weights"].items():
                if v > 0.01:
                    st.markdown(f"- **{names.get(k, k)[:20]}**: `{v*100:.1f}%`")
                    
        with c_q2:
            st.markdown("#### Econometric Metrics")
            df_q = pd.DataFrame({
                "Annualized Return": [f"{f_data['Annualized Return']:.2%}"],
                "Pure Alpha (Ann)": [f"{f_data['Alpha (Ann)']:.2%}"],
                "Info Ratio": [f"{f_data['Info Ratio']:.2f}"],
                "R² Adjusted": [f"{f_data['R² vs Benchmark']:.2f}"]
            })
            st.dataframe(df_q, hide_index=True, use_container_width=True)
            
        st.markdown("---")
        st.markdown("#### 📉 Linear Regression (Monthly Returns)")
        st.markdown("<small>Scattered points indicate high active risk. The intersection of the red trendline with the vertical axis represents pure monthly Alpha.</small>", unsafe_allow_html=True)
        
        st.info("⚠️ **Note on Non-Linearity:** This linear regression model assumes symmetrical market exposure. The metrics will naturally penalize highly convex strategies (e.g., trend-following managed futures) as inefficient, since these strategies profit mainly during strong market trends, creating convexity during extreme regimes.")
        
        f_monthly = f_data["F_Rets"].resample('ME').apply(lambda x: (1+x).prod()-1)
        b_series = pd.Series(f_data["B_Rets"], index=f_data["F_Rets"].index)
        b_monthly = b_series.resample('ME').apply(lambda x: (1+x).prod()-1)
        
        m, b = np.polyfit(b_monthly, f_monthly, 1)
        trendline_x = np.array([b_monthly.min(), b_monthly.max()])
        trendline_y = m * trendline_x + b

        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=b_monthly, y=f_monthly, 
            mode='markers', name='Historical Months',
            marker=dict(color='#3B82F6', size=8, opacity=0.7),
            hovertemplate="Benchmark: %{x:.2%}<br>Fund: %{y:.2%}<extra></extra>"
        ))
        
        fig_scatter.add_trace(go.Scatter(
            x=trendline_x, y=trendline_y, 
            mode='lines', name=f'Regression (Beta: {m:.2f})',
            line=dict(color='#F87171', width=3, dash='dot')
        ))
        
        fig_scatter.update_layout(
            template="plotly_dark", height=500,
            xaxis_title="Shadow Benchmark Return",
            yaxis_title="Fund Return",
            xaxis_tickformat=".1%", yaxis_tickformat=".1%",
            margin=dict(t=30, b=10, l=10, r=10),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        fig_scatter.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        fig_scatter.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        
        st.plotly_chart(fig_scatter, use_container_width=True)