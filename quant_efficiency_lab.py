import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.optimize import minimize
import requests
import re

st.set_page_config(page_title="Quant Efficiency Lab", layout="wide", page_icon="🔬")

# Regex per identificare un ISIN
ISIN_REGEX = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")

@st.cache_data(ttl=3600)
def resolve_ticker(query):
    """Traduzione robusta da ISIN a Ticker tramite Yahoo Finance Search"""
    q = str(query).strip().upper()
    if not ISIN_REGEX.match(q):
        return q
    
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, params={"q": q}, headers=headers, timeout=5).json()
        quotes = res.get("quotes", [])
        for x in quotes:
            if x.get("quoteType") in ("ETF", "MUTUALFUND"):
                return x["symbol"]
        return next((x["symbol"] for x in quotes if "symbol" in x), q)
    except Exception:
        return q

@st.cache_data(ttl=3600)
def fetch_price_series(ticker):
    """Scarica un singolo ticker, rimuove il fuso orario e restituisce (Serie prezzi, Nome)"""
    try:
        tk_obj = yf.Ticker(ticker)
        hist = tk_obj.history(period="max", auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            return pd.Series(dtype=float), ticker
        
        s = hist["Close"].dropna()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep='last')]
        
        try:
            name = tk_obj.info.get("shortName") or tk_obj.info.get("longName") or ticker
        except Exception:
            name = ticker
            
        return s, name
    except Exception:
        return pd.Series(dtype=float), ticker

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp { background-color: #0E1117; }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with st.sidebar:
    lang_choice = st.radio("🌐 Language / Lingua", ["🇮🇹 Italiano", "🇬🇧 English"], horizontal=True)
    is_it = "Italiano" in lang_choice

def t(en, it): return it if is_it else en

# --- CORE MATH ---
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
        if value > 0.85: return "🔴", t("Closet Indexer", "Finto Attivo"), t("It's a clone. Tracks the index but charges active fees.", "Copia l'indice ma fa pagare commissioni da gestione attiva.")
        if value > 0.60: return "🟡", t("Hybrid", "Gestione Ibrida"), t("Deviates partially, but core engine is the base index.", "Devia parzialmente, ma il motore resta l'indice.")
        return "🟢", t("Autonomous", "Fondo Autonomo"), t("Makes distinct choices, justifying active management.", "Compie scelte distinte, giustificando la gestione attiva.")
    
    if metric_name == "Active_Fee":
        if value > 0.04: return "🔴", t("Disproportionate Cost", "Costo Spropositato"), t("Requires unrealistic skill to recover this fee drag.", "Richiede abilità irrealistica per recuperare i costi.")
        if value > 0.025: return "🟡", t("High Burden", "Fardello Elevato"), t("The active bet is expensive.", "La scommessa attiva è costosa.")
        return "🟢", t("Fair Pricing", "Prezzo Equo"), t("Fair price for the active risk taken.", "Stai pagando un prezzo giusto per il rischio attivo.")
        
    if metric_name == "Airbag":
        if value > 0.95: return "🔴", t("No Airbag", "Nessun Airbag"), t("When it rains, you get just as wet as with a passive ETF.", "Ti bagni esattamente come con un passivo.")
        if value > 0.80: return "🟡", t("Weak Protection", "Protezione Debole"), t("Only partially cushions market crashes.", "Attutisce solo in parte i crolli.")
        return "🟢", t("True Protection", "Vera Protezione"), t("Anti-crisis fund, cuts market drawdowns.", "Fondo anti-crisi, taglia i crolli del mercato.")

def get_global_score(ir, p_value, down_cap, active_fee, r_squared, ter, win_rate, alpha_ann):
    if r_squared > 0.90 and ter <= 0.005:
        return 85.0, "P", t("🔵 PASSIVE ETF: Honest and efficient tracker.", "🔵 ETF PASSIVO: Tracker onesto ed efficiente.")

    score = 50.0  
    score += ir * 20 
    
    if alpha_ann > 0:
        if p_value <= 0.05: score += 15
        elif p_value > 0.10: score -= 5
    else:
        if p_value <= 0.05: score -= 15
        elif p_value > 0.10: score -= 5
    
    if down_cap > 1.0: score -= 10
    elif down_cap < 0.80: score += 10
    
    capped_active_fee = min(active_fee, 0.05)
    if capped_active_fee > 0.04: score -= 15 
    
    if win_rate is not None:
        if win_rate >= 0.70: score += 15
        elif win_rate < 0.40: score -= 15
    
    score = max(0, min(100, score))
    
    if score >= 80: return score, "S", t("🟢 OUTSTANDING: Rare excellence.", "🟢 ECCELLENTE: Rara eccellenza.")
    if score >= 65: return score, "A", t("🟢 GREAT: Generates real, solid value.", "🟢 OTTIMO: Genera valore reale solido.")
    if score >= 50: return score, "B", t("🟡 FAIR: Does its job, but consider passives.", "🟡 DISCRETO: Fa il suo lavoro, ma valuta un passivo.")
    if score >= 35: return score, "C", t("🟠 MEDIOCRE: Inefficient management.", "🟠 MEDIOCRE: Gestione inefficiente.")
    if score >= 20: return score, "D", t("🔴 POOR: Destroys value, high costs.", "🔴 SCARSO: Distrugge valore, costa troppo.")
    return score, "F", t("☠️ TOXIC: Consistent wealth destroyer.", "☠️ TOSSICO: Distruttore costante di ricchezza.")

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
        weight_dict = {"Global Equity": weights[0], "Bonds": weights[1], "Cash / Liquidity": weights[2]}
        dynamic_rf_rate_daily = bench_rets_raw.iloc[:, 2].mean()
    else:
        synthetic_bench_rets = bench_rets_raw.iloc[:, 0].values
        weight_dict = {"Global Equity": 1.0, "Bonds": 0.0, "Cash / Liquidity": 0.0}
        dynamic_rf_rate_daily = 0.02 / 252

    n_days = len(f_rets)
    ann_ret = (1 + f_rets).prod() ** (252 / n_days) - 1
    b_ann_ret = (1 + pd.Series(synthetic_bench_rets)).prod() ** (252 / n_days) - 1
    
    excess_f = f_rets - dynamic_rf_rate_daily
    excess_b = synthetic_bench_rets - dynamic_rf_rate_daily
    
    reg_result = stats.linregress(excess_b, excess_f)
    intercept = reg_result.intercept
    alpha_ann = intercept * 252
    r_squared = reg_result.rvalue**2

    active_share_proxy = max(1 - r_squared, 0.05) 
    active_fee = ter / active_share_proxy
    
    te = (f_rets - synthetic_bench_rets).std() * np.sqrt(252)
    ir = (ann_ret - b_ann_ret) / te if te > 0.0005 else 0.0
    
    volatility = f_rets.std() * np.sqrt(252)
    b_volatility = pd.Series(synthetic_bench_rets).std() * np.sqrt(252)
    
    f_wealth_index = (1 + f_rets).cumprod()
    max_dd = calc_drawdown(f_wealth_index).min()
    
    b_wealth_index = (1 + pd.Series(synthetic_bench_rets)).cumprod()
    b_max_dd = calc_drawdown(b_wealth_index).min()

    n = len(excess_b)
    if hasattr(reg_result, 'intercept_stderr') and reg_result.intercept_stderr > 0:
        t_stat_alpha = intercept / reg_result.intercept_stderr
        p_value_alpha = 2 * (1 - stats.t.cdf(np.abs(t_stat_alpha), df=n-2))
    else:
        p_value_alpha = 1.0
    
    down_mask = synthetic_bench_rets < 0
    if synthetic_bench_rets[down_mask].mean() < 0:
        down_cap = f_rets[down_mask].mean() / synthetic_bench_rets[down_mask].mean()
    else:
        down_cap = 0
    
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
        rolling_alpha = pd.Series(dtype=float)
        win_rate = None
        
    return {
        "Annualized Return": ann_ret,
        "Bench Annualized Return": b_ann_ret,
        "TER": ter,
        "Volatility": volatility,
        "Bench Volatility": b_volatility,
        "Max Drawdown": max_dd,
        "Bench Max DD": b_max_dd,
        "RBSA Weights": weight_dict,
        "R² vs Benchmark": r_squared, 
        "Active Fee Proxy": active_fee,
        "Alpha (Ann)": alpha_ann, 
        "Info Ratio": ir,
        "P-Value": p_value_alpha, 
        "Down Capture": down_cap,
        "Win Rate": win_rate,
        "Rolling Alpha": rolling_alpha,
        "Analysis Start": df.index[0].strftime('%Y-%m-%d'),
        "F_Rets": f_monthly, 
        "B_Rets": b_monthly
    }

# --- UI MAIN ---
st.title(t("🔬 Quant Efficiency Lab", "🔬 Laboratorio di Efficienza Quant"))

# SEZIONE INPUT
st.header(t("📦 Evaluate Instruments", "📦 Strumenti da Valutare"))

if is_it:
    with st.expander("💡 Spunti: Fondi famosi da testare"):
        st.markdown("""
        Copia questi Ticker e il loro TER nella tabella sottostante:
        * 🟢 **Pilastri Passivi:** `VT` (0.07%), `SPY` (0.09%).
        * 🟠 **Colossi Attivi:** `FCNTX` (0.77%), `AMCPX` (0.64%).
        * 🔴 **Scommesse Tematiche:** `ARKK` (0.75%), `ICLN` (0.40%).
        * 🟡 **Dividendi:** `JEPI` (0.35%).
        """)
else:
    with st.expander("💡 Ideas: Famous Funds to test"):
        st.markdown("""
        Copy these Tickers and their TER into the table below:
        * 🟢 **Passive Pillars:** `VT` (0.07%), `SPY` (0.09%).
        * 🟠 **Active Giants:** `FCNTX` (0.77%), `AMCPX` (0.64%).
        * 🔴 **Thematic Bets:** `ARKK` (0.75%), `ICLN` (0.40%).
        * 🟡 **Dividends:** `JEPI` (0.35%).
        """)
        
df_input = pd.DataFrame({
    "Ticker": ["VT", "AMCPX", "ARKK"],
    "TER (%)": [0.07, 0.64, 0.75]
})
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

run_btn = st.button(t("Run Analysis", "Avvia Analisi"), type="primary", use_container_width=True)

# SIDEBAR
with st.sidebar:
    st.header(t("⚙️ Engine Specs", "⚙️ Motore di Analisi"))
    
    currency_preset = st.selectbox(t("Currency / Region", "Valuta / Area Geografica"), ["🇺🇸 USD (Default)", "🇪🇺 EUR (UCITS)"])
    if "USD" in currency_preset:
        def_eq, def_bd, def_cs = "VT", "AGG", "BIL"
    else:
        def_eq, def_bd, def_cs = "EUNL.DE", "DBZB.DE", "XEON.DE"
        
    analysis_mode = st.radio(t("Comparison Method:", "Metodo di Confronto:"), [t("🤖 Shadow ETF (RBSA)", "🤖 Shadow ETF (RBSA)"), t("🎯 Direct Index", "🎯 Indice Singolo")])
    
    if "RBSA" in analysis_mode:
        eq_ticker = st.text_input(t("Equity Proxy", "Proxy Azionario"), def_eq)
        bd_ticker = st.text_input(t("Bond Proxy", "Proxy Obbligazionario"), def_bd)
        cs_ticker = st.text_input(t("Cash Proxy", "Proxy Liquidità"), def_cs)
        target_bench_tickers = [eq_ticker.strip().upper(), bd_ticker.strip().upper(), cs_ticker.strip().upper()]
    else:
        single_bench = st.text_input(t("Direct Benchmark", "Ticker Benchmark"), def_eq)
        target_bench_tickers = [single_bench.strip().upper()]

    st.markdown("---")
    st.caption(t("🚨 **Methodology Disclaimer:** Historical data evaluated via OLS Regression and SLSQP Optimization. Past performance does not guarantee future results. Educational tool only.", "🚨 **Disclaimer Metodologico:** Dati storici valutati tramite Regressione OLS e ottimizzazione SLSQP. I rendimenti passati non garantiscono risultati futuri. Solo a scopo educativo."))

# ESECUZIONE
if run_btn:
    valid_tickers = edited_df.dropna(subset=["Ticker"]).copy()
    valid_tickers["Ticker"] = valid_tickers["Ticker"].astype(str).str.strip().str.upper()
    valid_tickers = valid_tickers[valid_tickers["Ticker"] != ""]
    raw_inputs = valid_tickers["Ticker"].tolist()
    
    with st.spinner(t("Fetching data & computing...", "Download dati e calcolo quantitativo in corso...")):
        try:
            fund_mapping = {raw: resolve_ticker(raw) for raw in raw_inputs}
            bench_tickers = [resolve_ticker(tk) for tk in target_bench_tickers]
            
            all_t = list(set(list(fund_mapping.values()) + bench_tickers))
            names_dict = {}
            df_dict = {}
            
            for tk in all_t:
                s, long_name = fetch_price_series(tk)
                names_dict[tk] = long_name
                if not s.empty:
                    df_dict[tk] = s

            if not df_dict:
                st.error(t("❌ No data found from Yahoo Finance.", "❌ Nessun dato ricevuto da Yahoo Finance."))
                st.stop()

            data = pd.DataFrame(df_dict)
            
            missing_bench = [tk for tk in bench_tickers if tk not in data.columns]
            if missing_bench:
                st.error(t(f"❌ Missing benchmark data for {', '.join(missing_bench)}", f"❌ Dati benchmark mancanti per {', '.join(missing_bench)}"))
                st.stop()
                
            bench_data_df = data[bench_tickers].dropna()
            results = {}
            skipped_funds = []
            
            for raw_input, row_ter in zip(raw_inputs, valid_tickers["TER (%)"]):
                t_resolved = fund_mapping[raw_input]
                ter = row_ter / 100 if pd.notna(row_ter) else 0.0
                
                if t_resolved not in data.columns:
                    skipped_funds.append(f"**{raw_input}**")
                    continue
                    
                m = analyze_instrument(data[t_resolved], bench_data_df, ter, mode="RBSA" if "RBSA" in analysis_mode else "Direct")
                
                if m:
                    results[raw_input] = m
                    names_dict[raw_input] = names_dict.get(t_resolved, t_resolved)
                else:
                    skipped_funds.append(f"**{raw_input}** (Time series < 1 year)")
                    
            if skipped_funds:
                st.warning(t("⚠️ **Skipped (No data):** ", "⚠️ **Ignorati (Senza dati):** ") + ", ".join(skipped_funds))
                
            if not results:
                st.error(t("No valid instruments.", "Nessuno strumento valido."))
            else:
                st.session_state.results = results
                st.session_state.names = names_dict
                st.session_state.mode = analysis_mode
                
        except Exception as e:
            st.error(f"Critical Error: {str(e)}")

# RESULTS
if "results" in st.session_state:
    res = st.session_state.results
    names = st.session_state.names
    mode_used = st.session_state.get("mode", "🤖 Shadow ETF (RBSA)")
    
    st.divider()
    selected_t = st.selectbox(
        t("Select instrument:", "Seleziona strumento:"),
        list(res.keys()),
        format_func=lambda x: f"{names.get(x, x)} ({x})"
    )
    f_data = res[selected_t]
    f_monthly = f_data["F_Rets"]
    b_monthly = f_data["B_Rets"]
    
    tab_retail, tab_quant = st.tabs([t("👨‍💼 Retail Dashboard", "👨‍💼 Dashboard Retail"), t("🔬 Quant Lab", "🔬 Lab Quantitativo")])
    
    with tab_retail:
        score_num, score_letter, score_verdict = get_global_score(
            f_data['Info Ratio'], f_data['P-Value'], f_data['Down Capture'], 
            f_data['Active Fee Proxy'], f_data['R² vs Benchmark'], f_data['TER'], f_data['Win Rate'], f_data['Alpha (Ann)']
        )
        
        # HERO CARD UI
        st.markdown(f"## {t('Score', 'Voto')}: **{score_letter}** ({score_num:.0f}/100)")
        box_color = "#1e3a8a" if score_letter == "P" else "#052e16" if score_num >= 65 else "#451a03" if score_num >= 35 else "#450a0a"
        text_color = "#60a5fa" if score_letter == "P" else "#34d399" if score_num >= 65 else "#fbbf24" if score_num >= 35 else "#f87171"
        st.markdown(f"""<div style="background-color: {box_color}; padding: 15px; border-radius: 10px; border: 1px solid {text_color}; margin-bottom: 25px;">
            <h4 style="color: {text_color}; margin: 0;">{score_verdict}</h4></div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        icon_f, title_f, desc_f = get_retail_status("Fotocopia", f_data["R² vs Benchmark"])
        with c1:
            st.metric(t("Closet Indexing (Overlap)", "Sovrapposizione Indice"), f"{f_data['R² vs Benchmark']*100:.1f}%")
            st.caption(f"{icon_f} **{title_f}**")
            
        icon_c, title_c, desc_c = get_retail_status("Active_Fee", f_data["Active Fee Proxy"])
        with c2:
            st.metric(t("Active Fee Proxy", "Stima Costo Attivo"), f"{f_data['Active Fee Proxy']*100:.1f}%" if f_data['Active Fee Proxy'] < 9 else "> 10%")
            st.caption(f"{icon_c} **{title_c}**")
            
        icon_a, title_a, desc_a = get_retail_status("Airbag", f_data["Down Capture"])
        with c3:
            st.metric(t("Downside Capture", "Cattura Ribassi (Airbag)"), f"{f_data['Down Capture']*100:.1f}%")
            st.caption(f"{icon_a} **{title_a}**")

        if len(f_monthly) < 36:
            st.warning(t("⚠️ **Short Track Record:** Less than 3 years of data.", "⚠️ **Storico Breve:** Meno di 3 anni di dati."))

        st.markdown("---")
        st.markdown(t("### 📊 Absolute Performance (Fund vs Benchmark)", "### 📊 Performance Assoluta (Fondo vs Benchmark)"))
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(t("Annual Return", "Rendimento Annuo"), f"{f_data['Annualized Return']*100:.1f}%", f"Bench: {f_data['Bench Annualized Return']*100:.1f}%", delta_color="off")
        col_m2.metric(t("Volatility (Ann)", "Volatilità (Annua)"), f"{f_data['Volatility']*100:.1f}%", f"Bench: {f_data['Bench Volatility']*100:.1f}%", delta_color="off")
        col_m3.metric(t("Max Drawdown", "Max Drawdown (Crollo Max)"), f"{f_data['Max Drawdown']*100:.1f}%", f"Bench: {f_data['Bench Max DD']*100:.1f}%", delta_color="off")

        if "RBSA" in mode_used:
            st.markdown("---")
            st.markdown(t("### 🧬 Fund DNA", "### 🧬 DNA del Fondo"))
            w_eq = round(f_data["RBSA Weights"].get("Global Equity", 0), 4)
            w_bd = round(f_data["RBSA Weights"].get("Bonds", 0), 4)
            w_cs = round(f_data["RBSA Weights"].get("Cash / Liquidity", 0), 4)
            
            col_dna_text, col_dna_chart = st.columns([1, 1])
            with col_dna_text:
                st.metric(t("📈 Equity", "📈 Azioni"), f"{w_eq*100:.1f}%")
                st.metric(t("🛡️ Bonds", "🛡️ Obbligazioni"), f"{w_bd*100:.1f}%")
                st.metric(t("💶 Cash", "💶 Liquidità"), f"{w_cs*100:.1f}%")
            with col_dna_chart:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=[t("Equity", "Azioni"), t("Bonds", "Obbligazioni"), t("Cash", "Liquidità")],
                    values=[w_eq, w_bd, w_cs], hole=.6, marker_colors=['#3B82F6', '#8B5CF6', '#10B981']
                )])
                fig_donut.update_layout(template="plotly_dark", height=200, margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
                st.plotly_chart(fig_donut, use_container_width=True)

        # --- GRAFICO COSTI ---
        st.markdown("---")
        st.markdown(t("### 💸 The Cost of Management (Capital Eaten by Fees)", "### 💸 Il Peso dei Costi (Capitale Mangiato)"))
        
        initial_capital = 10000
        ter_monthly = f_data["TER"] / 12
        f_gross_rets = f_monthly + ter_monthly
        
        wealth_net = (1 + f_monthly).cumprod() * initial_capital
        wealth_gross = (1 + f_gross_rets).cumprod() * initial_capital
        lost_capital = wealth_gross.iloc[-1] - wealth_net.iloc[-1]
        
        fig_fees = go.Figure()
        fig_fees.add_trace(go.Scatter(x=wealth_net.index, y=wealth_net, mode='lines', name=t('Net Wealth (Actual)', 'Capitale Netto (Reale)'), line=dict(color='#34d399', width=2)))
        fig_fees.add_trace(go.Scatter(x=wealth_gross.index, y=wealth_gross, mode='lines', name=t('Gross Wealth (No Fees)', 'Capitale Lordo (Senza Costi)'), line=dict(color='#9ca3af', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(248, 113, 113, 0.4)'))
        fig_fees.update_layout(template="plotly_dark", height=350, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickformat=",.0f", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        
        st.metric(t("Total Capital Eaten by Fees", "Capitale Totale Mangiato dai Costi"), f"- €{lost_capital:,.0f}")
        st.plotly_chart(fig_fees, use_container_width=True)

        # --- GRAFICO ANNI SOLARI ---
        st.markdown("---")
        st.markdown(t("### 📅 Year-by-Year Performance", "### 📅 Performance Anno per Anno"))
        
        f_yearly = f_monthly.resample('YE').apply(lambda x: (1+x).prod()-1)
        b_yearly = b_monthly.resample('YE').apply(lambda x: (1+x).prod()-1)
        years = f_yearly.index.year
        
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Bar(x=years, y=f_yearly, name=t('Fund', 'Fondo'), marker_color=np.where(f_yearly >= 0, '#34d399', '#f87171')))
        fig_yearly.add_trace(go.Bar(x=years, y=b_yearly, name='Shadow Benchmark', marker_color='#9ca3af', opacity=0.6))
        fig_yearly.update_layout(template="plotly_dark", height=350, barmode='group', margin=dict(t=30, b=10, l=10, r=10), yaxis_tickformat=".1%", xaxis=dict(type='category'), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_yearly, use_container_width=True)

    with tab_quant:
        st.subheader(t("Econometric Metrics", "Metriche Econometriche"))
        df_q = pd.DataFrame({
            t("Pure Alpha (Ann)", "Alpha Puro (Annuo)"): [f"{f_data['Alpha (Ann)']:.2%}"],
            "Info Ratio": [f"{f_data['Info Ratio']:.2f}"],
            t("P-Value", "P-Value (Rischio Fortuna)"): [f"{f_data['P-Value']:.3f}"],
            "R²": [f"{f_data['R² vs Benchmark']:.2f}"]
        })
        st.dataframe(df_q, hide_index=True, use_container_width=True)
            
        # --- SCATTER PLOT ---
        st.markdown("---")
        st.markdown(t("#### 📉 Regression Scatter", "#### 📉 Regressione Lineare"))
        m, b = np.polyfit(b_monthly, f_monthly, 1)
        trendline_x = np.array([b_monthly.min(), b_monthly.max()])
        trendline_y = m * trendline_x + b

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=b_monthly, y=f_monthly, mode='markers', marker=dict(color='#3B82F6', size=6, opacity=0.6)))
        fig_scatter.add_trace(go.Scatter(x=trendline_x, y=trendline_y, mode='lines', line=dict(color='#F87171', width=2, dash='dot')))
        fig_scatter.update_layout(template="plotly_dark", height=400, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        fig_scatter.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        fig_scatter.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- ROLLING ALPHA ---
        st.markdown("---")
        st.markdown(t("#### 🔄 3-Year Rolling Alpha", "#### 🔄 Alpha Rolling a 3 Anni"))
        rolling_alpha = f_data.get("Rolling Alpha")
        if rolling_alpha is not None and not rolling_alpha.empty:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=rolling_alpha.index, y=rolling_alpha.clip(lower=0), mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(52, 211, 153, 0.4)'))
            fig_roll.add_trace(go.Scatter(x=rolling_alpha.index, y=rolling_alpha.clip(upper=0), mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='rgba(248, 113, 113, 0.4)'))
            fig_roll.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, yaxis_tickformat=".1%")
            fig_roll.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.5)")
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("No data.")
            
        # --- HISTOGRAM ---
        st.markdown("---")
        st.markdown(t("#### 📊 Return Distribution (Tails & Skewness)", "#### 📊 Distribuzione dei Rendimenti (Asimmetria e Code)"))
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=f_monthly, name=t('Fund', 'Fondo'), marker_color='#34d399', opacity=0.7, nbinsx=50))
        fig_hist.add_trace(go.Histogram(x=b_monthly, name='Shadow Benchmark', marker_color='#9ca3af', opacity=0.5, nbinsx=50))
        fig_hist.update_layout(template="plotly_dark", height=350, barmode='overlay', margin=dict(t=10, b=10, l=10, r=10), xaxis_tickformat=".1%", legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_hist, use_container_width=True)
