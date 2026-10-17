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

# --- CSS INJECTION (Nasconde elementi di default per un look professionale) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp {
                background-color: #0E1117;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- LINGUA E TRADUZIONI ---
with st.sidebar:
    lang_choice = st.radio("🌐 Language / Lingua", ["🇮🇹 Italiano", "🇬🇧 English"], horizontal=True)
    is_it = "Italiano" in lang_choice

def t(en, it):
    return it if is_it else en

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
        if value > 0.85: return "🔴", t("Closet Indexer", "Finto Attivo (Closet Indexer)"), t("It's a clone. Tracks the index but charges active management fees.", "È un clone. Copia l'indice ma fa pagare commissioni da gestione attiva.")
        if value > 0.60: return "🟡", t("Hybrid Management", "Gestione Ibrida"), t("Deviates partially, but the core engine remains the base index.", "Devia parzialmente, ma il motore principale resta l'indice di base.")
        return "🟢", t("Autonomous Fund", "Fondo Autonomo"), t("Makes truly distinct choices, justifying active management.", "Compie scelte nettamente distinte, giustificando la gestione attiva.")
    
    if metric_name == "Active_Fee":
        if value > 0.04: return "🔴", t("Disproportionate Cost", "Costo Spropositato"), t("Requires unrealistic skill to recover this fee drag.", "Richiede un'abilità irrealistica per recuperare questa zavorra di costi.")
        if value > 0.025: return "🟡", t("High Burden", "Fardello Elevato"), t("The active bet is expensive.", "La scommessa attiva è costosa.")
        return "🟢", t("Fair Pricing", "Prezzo Equo"), t("You are paying a fair price for the active risk taken.", "Stai pagando un prezzo giusto per il rischio attivo che il gestore si prende.")
        
    if metric_name == "IR":
        if value < 0: return "🔴", t("Value Destructor", "Distruttore di Valore"), t("Higher volatility than the index and lower returns.", "Volatilità più alta dell'indice e rendimenti inferiori.")
        if value < 0.5: return "🟡", t("Inefficient", "Inefficiente"), t("Active risk is not adequately rewarded.", "Il rischio attivo preso non è adeguatamente ricompensato.")
        return "🟢", t("Efficient", "Efficiente"), t("The active bet pays off significantly.", "La scommessa attiva ripaga in modo significativo.")
        
    if metric_name == "Airbag":
        if value > 0.95: return "🔴", t("No Airbag", "Nessun Airbag"), t("When it rains, you get just as wet as with a passive ETF.", "Quando piove, ti bagni esattamente come con un ETF passivo.")
        if value > 0.80: return "🟡", t("Weak Protection", "Protezione Debole"), t("Only partially cushions market crashes.", "Attutisce solo in minima parte i crolli di mercato.")
        return "🟢", t("True Protection", "Vera Protezione"), t("Anti-crisis fund, sharply cuts market drawdowns.", "Fondo anti-crisi, taglia drasticamente i crolli del mercato.")
        
    if metric_name == "P_Value":
        if value > 0.10: return "🔴", t("Pure Luck", "Pura Fortuna"), t("No mathematical proof that the manager can consistently beat the market.", "Nessuna prova matematica che il gestore sappia battere il mercato con costanza.")
        if value <= 0.05: return "🟢", t("Solid Skill", "Abilità Solida"), t("Excellent statistical significance.", "Eccellente significatività statistica.")
        return "🟡", t("Gray Zone", "Zona Grigia"), t("Positive results, but the doubt of luck persists.", "Risultati positivi, ma resta il dubbio che sia stato un caso.")
        
    if metric_name == "Consistency":
        if value is None: return "⚪", t("Not Enough Data", "Dati Insufficienti"), t("Requires at least 3 years of history.", "Richiede almeno 3 anni di storico.")
        if value > 0.75: return "🟢", t("Consistent Winner", "Vincente Sistematico"), t("Reliably beats the benchmark across rolling 3-year periods.", "Batte sistematicamente il benchmark su orizzonti di 3 anni.")
        if value < 0.40: return "🔴", t("Chronic Underperformer", "Perdente Cronico"), t("Holding this for 3 years historically results in losing to the index.", "Tenere questo fondo per 3 anni porta storicamente a perdere contro l'indice.")
        return "🟡", t("Coin Flip", "Lancio della Moneta"), t("Chances of beating the index over 3 years are near 50/50.", "Le probabilità di battere l'indice su 3 anni sono vicine al 50/50.")

def get_global_score(ir, p_value, down_cap, active_fee, r_squared, ter, win_rate):
    if r_squared > 0.80 and ter <= 0.005:
        return 85.0, "P", t("🔵 PASSIVE ETF: Honest and efficient tracker. Excellent as a core portfolio engine.", "🔵 ETF PASSIVO: Tracker onesto ed efficiente. Eccellente come motore base del portafoglio.")

    score = 50.0  
    score += ir * 25 
    
    if p_value > 0.10: score -= 10
    elif p_value <= 0.05: score += 10
    
    if down_cap > 1.0: score -= 10
    elif down_cap < 0.80: score += 10
    
    if active_fee > 0.04: score -= 15 
    
    if win_rate is not None:
        if win_rate >= 0.70: score += 15
        elif win_rate < 0.40: score -= 15
    
    score = max(0, min(100, score))
    
    if score >= 80: return score, "S", t("🟢 OUTSTANDING: Rare excellence. Fully justifies the fees.", "🟢 ECCELLENTE: Rara eccellenza. Giustifica pienamente le commissioni.")
    if score >= 65: return score, "A", t("🟢 GREAT: Generates real, solid, and statistically valid value.", "🟢 OTTIMO: Genera valore reale, solido e statisticamente valido.")
    if score >= 50: return score, "B", t("🟡 FAIR: Does its job, but consider if a passive ETF is cheaper.", "🟡 DISCRETO: Fa il suo lavoro, ma valuta se un ETF passivo ti costerebbe meno.")
    if score >= 35: return score, "C", t("🟠 MEDIOCRE: Inefficient. You are paying for results mostly dictated by chance.", "🟠 MEDIOCRE: Inefficiente. Paghi caro per risultati spesso dettati dal caso.")
    if score >= 20: return score, "D", t("🔴 POOR: Destroys value, costs too much, and doesn't protect in crashes.", "🔴 SCARSO: Distrugge valore, costa troppo e non protegge nei crolli.")
    return score, "F", t("☠️ TOXIC: Sell immediately. Capital bleed guaranteed.", "☠️ TOSSICO: Da vendere subito. Sanguinamento di capitale garantito.")

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
    
    ir = (ann_ret - b_ann_ret) / te if te > 0.0005 else 0.0
    
    volatility = f_rets.std() * np.sqrt(252)
    b_volatility = pd.Series(synthetic_bench_rets).std() * np.sqrt(252)
    
    sharpe_ratio = (ann_ret - (dynamic_rf_rate_daily * 252)) / volatility if volatility > 0 else 0
    b_sharpe_ratio = (b_ann_ret - (dynamic_rf_rate_daily * 252)) / b_volatility if b_volatility > 0 else 0
    
    f_wealth_index = (1 + f_rets).cumprod()
    max_dd = calc_drawdown(f_wealth_index).min()
    
    b_wealth_index = (1 + pd.Series(synthetic_bench_rets)).cumprod()
    b_max_dd = calc_drawdown(b_wealth_index).min()

    n = len(excess_b)
    resid = excess_f - (intercept + slope * excess_b)
    s2_e = np.sum(resid**2) / (n - 2) if n > 2 else 0
    se_alpha = np.sqrt(s2_e * (1/n + np.mean(excess_b)**2 / np.sum((excess_b - np.mean(excess_b))**2))) if s2_e > 0 else 0
    t_stat_alpha = intercept / se_alpha if se_alpha > 0 else 0
    p_value_alpha = 2 * (1 - stats.t.cdf(np.abs(t_stat_alpha), df=n-2)) if n > 2 else 1.0
    
    up_mask = bench_rets_raw.iloc[:, 0] > 0
    down_mask = bench_rets_raw.iloc[:, 0] < 0
    down_cap = (f_rets[down_mask].mean() / bench_rets_raw.iloc[:, 0][down_mask].mean()) if bench_rets_raw.iloc[:, 0][down_mask].mean() < 0 else 0
    
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
        "Sharpe Ratio": sharpe_ratio,
        "Bench Sharpe": b_sharpe_ratio,
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
        "F_Rets": f_rets, 
        "B_Rets": pd.Series(synthetic_bench_rets, index=f_rets.index)
    }

# --- UI MAIN PORTION ---
st.title(t("🔬 Quant Efficiency Lab", "🔬 Laboratorio di Efficienza Quant"))

# --- SEZIONE STRUMENTI AL CENTRO DELLA PAGINA ---
st.header(t("📦 Instruments to Evaluate", "📦 Strumenti da Valutare"))

if is_it:
    with st.expander("💡 Spunti: Fondi famosi da testare"):
        st.markdown("""
        Copia questi Ticker nella tabella sottostante per mettere alla prova l'algoritmo:
        * 🟢 **I Pilastri Passivi:** `VT` (Total World), `SPY` (S&P 500). *Tracker globali e statunitensi tra i più capitalizzati al mondo.*
        * 🟠 **I Colossi Attivi:** `FCNTX` (Fidelity), `AMCPX` (American Funds). *Fondi a gestione attiva storici e con enormi masse in gestione.*
        * 🔴 **Le Scommesse Tematiche:** `ARKK` (Ark Innovation), `ICLN` (Clean Energy). *Fondi ad alta volatilità legati a specifici settori o mega-trend.*
        * 🟡 **I Fondi da Dividendo:** `JEPI` (JPMorgan Equity Premium). *Molto popolari per la strategia basata sulla distribuzione di alte cedole.*
        """)
else:
    with st.expander("💡 Ideas: Famous Funds to test"):
        st.markdown("""
        Copy these famous Tickers into the table below to stress-test the algorithm:
        * 🟢 **The Passive Pillars:** `VT` (Total World), `SPY` (S&P 500). *Some of the most capitalized global and US trackers in the world.*
        * 🟠 **The Active Giants:** `FCNTX` (Fidelity), `AMCPX` (American Funds). *Historical actively managed funds with massive AUM.*
        * 🔴 **The Thematic Bets:** `ARKK` (Ark Innovation), `ICLN` (Clean Energy). *High-volatility funds tied to specific sectors or mega-trends.*
        * 🟡 **The Dividend Funds:** `JEPI` (JPMorgan Equity Premium). *Highly popular for their high-yield income distribution strategies.*
        """)
        
df_input = pd.DataFrame({
    "Ticker": ["VT", "ARKK", "JEPI"],
    "TER (%)": [0.07, 0.75, 0.35]
})
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

if st.button(t("Run X-Ray Analysis", "Avvia Analisi a Raggi X"), type="primary", use_container_width=True):
    st.session_state.run_analysis = True
else:
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False

# --- SIDEBAR (Solo Impostazioni Tecniche) ---
with st.sidebar:
    st.header(t("⚙️ Analysis Engine", "⚙️ Motore di Analisi"))
    
    currency_preset = st.selectbox(t("Currency / Region Setup", "Valuta / Area Geografica"), ["🇺🇸 USD (Default - US/Global)", "🇪🇺 EUR (European UCITS)"])
    if "USD" in currency_preset:
        def_eq, def_bd, def_cs = "VT", "AGG", "BIL"
    else:
        def_eq, def_bd, def_cs = "EUNL.DE", "DBZB.DE", "XEON.DE"
        
    analysis_mode = st.radio(t("Comparison Method:", "Metodo di Confronto:"), [t("🤖 Shadow ETF (RBSA)", "🤖 Shadow ETF (RBSA Sintetico)"), t("🎯 Single Benchmark (Direct)", "🎯 Benchmark Singolo (Diretto)")])
    
    if "RBSA" in analysis_mode:
        st.caption(t("The engine deconstructs the fund and builds a synthetic benchmark.", "Il motore decostruisce il fondo e crea un benchmark sintetico su misura."))
        st.info(t("ℹ️ **RBSA Constraints:** The model uses constrained optimization to build a realistic, long-only shadow portfolio (weights must be ≥ 0 and sum exactly to 100%).", "ℹ️ **Vincoli RBSA:** Il modello utilizza un'ottimizzazione vincolata per costruire un portafoglio ombra long-only realistico (i pesi devono essere ≥ 0 e sommare al 100%)."))
        eq_ticker = st.text_input(t("Equity Proxy", "Proxy Azionario"), def_eq)
        bd_ticker = st.text_input(t("Bond Proxy", "Proxy Obbligazionario"), def_bd)
        cs_ticker = st.text_input(t("Cash Proxy", "Proxy Liquidità"), def_cs)
        target_bench_tickers = [eq_ticker.strip().upper(), bd_ticker.strip().upper(), cs_ticker.strip().upper()]
    else:
        st.caption(t("1-vs-1 Match: Fund against your chosen Index.", "Scontro 1-vs-1: Il fondo contro l'indice che hai scelto."))
        single_bench = st.text_input(t("Direct Benchmark Ticker", "Ticker del Benchmark"), def_eq)
        target_bench_tickers = [single_bench.strip().upper()]

    st.caption(t("📉 *Risk-Free rate is dynamically calculated based on the selected Cash Proxy to reflect different market regimes.*", "📉 *Il tasso Risk-Free è calcolato dinamicamente sulla base del Proxy Liquidità scelto per riflettere i diversi regimi di mercato.*"))
    
    if is_it:
        with st.expander("📚 Come funziona l'algoritmo (Metodologia RBSA)"):
            st.markdown("""
            **Returns-Based Style Analysis (RBSA)** è una tecnica quantitativa introdotta dal premio Nobel William Sharpe. 
            Il software utilizza l'ottimizzazione quadratica sequenziale (SLSQP) per calcolare il mix esatto di indici passivi (vincolato tra 0% e 100%) che minimizza la varianza dell'errore di tracciamento rispetto al fondo. 
            - **Active Fee Proxy:** Stima il vero costo percentuale caricato solo sulla parte di portafoglio gestita attivamente ($1 - R^2$).
            - **P-Value:** Derivato dalla Statistica-T dell'intercetta della regressione (OLS), valuta se l'Alpha prodotto è statisticamente diverso da zero o se è puro rumore.
            """)
    else:
        with st.expander("📚 How the engine works (RBSA Methodology)"):
            st.markdown("""
            **Returns-Based Style Analysis (RBSA)** is a quantitative technique introduced by Nobel laureate William Sharpe. 
            The software uses Sequential Least Squares Programming (SLSQP) to find the exact asset mix (constrained between 0% and 100%) 
            that minimizes the variance of the tracking error compared to the analyzed fund. 
            - **Active Fee Proxy:** Calculated by isolating the total expense ratio (TER) strictly on the active share of the portfolio ($1 - R^2$).
            - **P-Value:** Derived from the T-Statistic of the OLS regression intercept, testing the null hypothesis that Alpha equals zero.
            """)

# --- ESECUZIONE ANALISI E RISULTATI ---
if st.session_state.run_analysis:
    valid_tickers = edited_df.dropna(subset=["Ticker"]).copy()
    raw_inputs = valid_tickers["Ticker"].str.strip().tolist()
    
    with st.spinner(t("Translating ISINs, aligning dates, and running quantitative analysis...", "Traduzione ISIN, allineamento date e calcolo quantitativo in corso...")):
        try:
            fund_mapping = {raw: resolve_ticker(raw) for raw in raw_inputs}
            bench_tickers = [resolve_ticker(t) for t in target_bench_tickers]
            
            all_t = list(set(list(fund_mapping.values()) + bench_tickers))
            names_dict = {}
            df_dict = {}
            
            for tk in all_t:
                temp = yf.download(tk, period="max", progress=False)
                if not temp.empty:
                    names_dict[tk] = yf.Ticker(tk).info.get("longName", tk)
                    if isinstance(temp.columns, pd.MultiIndex):
                        temp.columns = temp.columns.droplevel(1)
                        
                    if temp.index.tz is not None:
                        temp.index = temp.index.tz_localize(None)
                    temp.index = temp.index.normalize()
                        
                    if "Adj Close" in temp.columns:
                        df_dict[tk] = temp["Adj Close"].dropna()
                    elif "Close" in temp.columns:
                        df_dict[tk] = temp["Close"].dropna()
                        
            if not df_dict:
                st.error(t("❌ No data found. Yahoo Finance is unresponsive or tickers are invalid.", "❌ Nessun dato trovato. Yahoo Finance non risponde o i ticker non sono validi."))
                st.stop()
                
            data = pd.DataFrame(df_dict)
            
            missing_bench = [tk for tk in bench_tickers if tk not in data.columns]
            if missing_bench:
                st.error(t(f"❌ Cannot analyze: Yahoo Finance missing benchmark data for {', '.join(missing_bench)}", f"❌ Impossibile analizzare: dati benchmark mancanti per {', '.join(missing_bench)}"))
                st.stop()
                
            bench_data_df = data[bench_tickers].dropna()
            
            results = {}
            skipped_funds = []
            
            for raw_input, row_ter in zip(raw_inputs, valid_tickers["TER (%)"]):
                t_resolved = fund_mapping[raw_input]
                ter = row_ter / 100 if pd.notna(row_ter) else 0.0
                
                if t_resolved not in data.columns:
                    skipped_funds.append(f"**{raw_input}**: " + t("ISIN/Ticker not found in Yahoo Finance database.", "ISIN/Ticker non trovato nel database di Yahoo Finance."))
                    continue
                    
                m = analyze_instrument(data[t_resolved], bench_data_df, ter, mode="RBSA" if "RBSA" in analysis_mode else "Direct")
                
                if m:
                    results[raw_input] = m
                    names_dict[raw_input] = names_dict.get(t_resolved, t_resolved)
                else:
                    skipped_funds.append(f"**{raw_input}**: " + t("Time series too short or fragmented (min 1 year required).", "Serie storica troppo breve o frammentata (richiesto minimo 1 anno)."))
                    
            if skipped_funds:
                st.warning(t("⚠️ **Some instruments were skipped:**\n", "⚠️ **Alcuni strumenti sono stati ignorati:**\n") + "\n".join([f"- {sf}" for sf in skipped_funds]))
                
            if not results:
                st.error(t("No valid instruments for analysis.", "Nessuno strumento valido per l'analisi."))
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
    selected_t = st.selectbox(t("Select the instrument to inspect:", "Seleziona lo strumento da ispezionare:"), list(res.keys()), format_func=lambda x: f"{names.get(x, x)} ({x})")
    f_data = res[selected_t]
    
    tab_retail, tab_quant = st.tabs([t("👨‍💼 Investor Dashboard (Intuitive)", "👨‍💼 Dashboard Investitore (Intuitiva)"), t("🔬 Quant Laboratory (Statistical)", "🔬 Laboratorio Quantitativo (Statistico)")])
    
    with tab_retail:
        score_num, score_letter, score_verdict = get_global_score(
            f_data['Info Ratio'], f_data['P-Value'], f_data['Down Capture'], 
            f_data['Active Fee Proxy'], f_data['R² vs Benchmark'], f_data['TER'], f_data['Win Rate']
        )
        
        st.markdown(f"## {t('Global Score', 'Punteggio Globale')}: **{score_letter}** ({score_num:.0f}/100)")
        
        box_color = "#1e3a8a" if score_letter == "P" else "#052e16" if score_num >= 65 else "#451a03" if score_num >= 35 else "#450a0a"
        text_color = "#60a5fa" if score_letter == "P" else "#34d399" if score_num >= 65 else "#fbbf24" if score_num >= 35 else "#f87171"
        
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; border: 1px solid {text_color}; margin-bottom: 25px;">
            <h4 style="color: {text_color}; margin: 0;">{score_verdict}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(t("Real-World Performance Profile", "Profilo di Performance Reale"))
        bench_desc = t("its real 'Shadow ETF'", "il suo 'Shadow ETF' reale") if "RBSA" in mode_used else t("your selected benchmark", "il benchmark selezionato")
        st.markdown(t(f"Historical data validated since **{f_data['Analysis Start']}**. The fund was analyzed against {bench_desc}.", f"Dati storici validati dal **{f_data['Analysis Start']}**. Il fondo è stato analizzato contro {bench_desc}."))
        
        st.markdown("---")
        st.markdown(t("### 📊 Absolute Performance (Fund vs Benchmark)", "### 📊 Performance Assoluta (Fondo vs Benchmark)"))
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        col_m1.metric(t("Annual Return", "Rendimento Annuo"), 
                      f"{f_data['Annualized Return']*100:.1f}%", 
                      f"Bench: {f_data['Bench Annualized Return']*100:.1f}%", delta_color="off")
                      
        col_m2.metric(t("Volatility (Ann)", "Volatilità (Annua)"), 
                      f"{f_data['Volatility']*100:.1f}%", 
                      f"Bench: {f_data['Bench Volatility']*100:.1f}%", delta_color="off")
                      
        col_m3.metric("Sharpe Ratio", 
                      f"{f_data['Sharpe Ratio']:.2f}", 
                      f"Bench: {f_data['Bench Sharpe']:.2f}", delta_color="off")
                      
        col_m4.metric(t("Max Drawdown", "Max Drawdown (Crollo Max)"), 
                      f"{f_data['Max Drawdown']*100:.1f}%", 
                      f"Bench: {f_data['Bench Max DD']*100:.1f}%", delta_color="off")
        
        if len(f_data['F_Rets']) < 252 * 3:
            st.warning(t("⚠️ **Short Track Record:** The instrument has less than 3 years of historical data. Statistical metrics like P-Value and Rolling Win Rate may not be fully reliable and could be driven by market noise.", "⚠️ **Track Record Breve:** Lo strumento ha meno di 3 anni di storico. Metriche statistiche come il P-Value e il Win Rate potrebbero non essere affidabili e derivare dal rumore di mercato."))
        
        if "RBSA" in mode_used:
            st.markdown("---")
            st.markdown(t("### 🧬 Fund DNA (Under the Hood)", "### 🧬 DNA del Fondo (Sotto il Cofano)"))
            
            w_eq = round(f_data["RBSA Weights"].get("Global Equity", 0), 4)
            w_bd = round(f_data["RBSA Weights"].get("Bonds", 0), 4)
            w_cs = round(f_data["RBSA Weights"].get("Cash / Liquidity", 0), 4)
            
            col_dna_text, col_dna_chart = st.columns([1, 1])
            
            with col_dna_text:
                st.write("") 
                st.metric(t("📈 Equity Weight", "📈 Peso Azionario"), f"{w_eq*100:.1f}%")
                st.metric(t("🛡️ Bond Weight", "🛡️ Peso Obbligazionario"), f"{w_bd*100:.1f}%")
                st.metric(t("💶 Cash / Liquidity", "💶 Peso Liquidità"), f"{w_cs*100:.1f}%")
                
                if w_eq >= 0.75:
                    st.warning(t(f"⚠ **WARNING:** This instrument is **{w_eq*100:.0f}%** driven by Global Equities. Do not be fooled by thematic narratives: if global stock markets crash, this fund will crash with them.", f"⚠ **ATTENZIONE:** Questo strumento è guidato per il **{w_eq*100:.0f}%** dall'Azionario Globale. Non farti ingannare dalle narrazioni: se le borse crollano, questo fondo crollerà con loro."))
                elif w_eq <= 0.30:
                    st.success(t("✅ **UNCORRELATED:** This instrument operates with an engine profoundly different from standard equities.", "✅ **DECORRELATO:** Questo strumento opera con un motore profondamente diverso dalle classiche azioni."))
            
            with col_dna_chart:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=[t("Equity", "Azioni"), t("Bonds", "Obbligazioni"), t("Cash", "Liquidità")],
                    values=[w_eq, w_bd, w_cs],
                    hole=.6,
                    marker_colors=['#3B82F6', '#8B5CF6', '#10B981'],
                    hovertemplate="%{label}: %{value:.1%}<extra></extra>"
                )])
                fig_donut.update_layout(
                    template="plotly_dark", height=250, margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False, annotations=[dict(text='DNA', x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(color="white"))]
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        
        icon, title, desc = get_retail_status("Fotocopia", f_data["R² vs Benchmark"])
        with c1:
            st.markdown(f"### {icon} {t('Index Overlap', 'Sovrapposizione all\'Indice')}")
            st.markdown(t("**How much is this fund copying the market instead of actively managing it?**", "**Quanto questo fondo copia il mercato invece di gestirlo attivamente?**"))
            st.metric(t("Correlation to Index", "Correlazione all'Indice"), f"{f_data['R² vs Benchmark']*100:.1f}%")
            st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("Active_Fee", f_data["Active Fee Proxy"])
        with c2:
            st.markdown(f"### {icon} {t('Active Fee Proxy', 'Stima Costo Attivo')}")
            st.markdown(t("**On the truly active portion, what hidden commission rate are you paying?**", "**Sulla porzione puramente attiva, che commissione nascosta stai pagando?**"))
            st.metric(t("Estimated Active Fee", "Costo Attivo Stimato"), f"{f_data['Active Fee Proxy']*100:.1f}%" if f_data['Active Fee Proxy'] < 9 else "> 10%")
            st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("Airbag", f_data["Down Capture"])
        with c3:
            st.markdown(f"### {icon} {t('The Airbag Test', 'Il Test dell\'Airbag')}")
            st.markdown(t("**During market sell-offs, does the fund avoid the crash or fall with everyone else?**", "**Durante i crolli di mercato, il fondo li evita o cade insieme a tutti gli altri?**"))
            st.metric("Drawdown Capture", f"{f_data['Down Capture']*100:.1f}%")
            st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {title}. {desc}")

        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        
        icon, title, desc = get_retail_status("IR", f_data["Info Ratio"])
        with c4:
            st.markdown(f"### {icon} {t('Efficiency Index', 'Indice di Efficienza')}")
            st.markdown(t("**For every sleepless night caused by active risk, how much extra return was generated?**", "**Per ogni notte insonne causata dal rischio attivo, quanto extra rendimento è stato generato?**"))
            st.metric("Information Ratio", f"{f_data['Info Ratio']:.2f}")
            st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("P_Value", f_data["P-Value"])
        with c5:
            st.markdown(f"### {icon} {t('Skill or Luck?', 'Abilità o Fortuna?')}")
            st.markdown(t("**Probability that the positive performance is just a fortunate coincidence.**", "**Probabilità che le performance positive siano solo una fortunata coincidenza.**"))
            st.metric(t("Luck Risk (P-Value)", "Rischio Fortuna (P-Value)"), f"{f_data['P-Value']*100:.1f}%")
            st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {title}. {desc}")
            
        icon, title, desc = get_retail_status("Consistency", f_data["Win Rate"])
        with c6:
            st.markdown(f"### {icon} {t('3Y Consistency', 'Consistenza a 3 Anni')}")
            st.markdown(t("**How often does it beat the market over a 3-year holding period?**", "**Quante volte batte il mercato se lo tieni in portafoglio per 3 anni?**"))
            val_str = f"{f_data['Win Rate']*100:.1f}%" if f_data["Win Rate"] is not None else "N/A"
            st.metric(t("Rolling Win Rate", "Win Rate Rolling"), val_str)
            st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {title}. {desc}")

        st.markdown("---")
        st.markdown(t("### 📅 Year-by-Year Performance", "### 📅 Performance Anno per Anno"))
        st.markdown(t("<small>Shows the calendar year returns. A quick way to see if the fund manages to protect capital during crisis years (e.g., 2022) and keep up during bull markets.</small>", "<small>Mostra i rendimenti per anno solare. Un modo rapido per vedere se il fondo protegge il capitale negli anni di crisi (es. 2022) e se tiene il passo nei mercati toro.</small>"), unsafe_allow_html=True)
        
        f_yearly = f_data["F_Rets"].resample('YE').apply(lambda x: (1+x).prod()-1)
        b_series = pd.Series(f_data["B_Rets"], index=f_data["F_Rets"].index)
        b_yearly = b_series.resample('YE').apply(lambda x: (1+x).prod()-1)
        
        years = f_yearly.index.year
        
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Bar(
            x=years, y=f_yearly, name=t('Fund', 'Fondo'),
            marker_color=np.where(f_yearly >= 0, '#34d399', '#f87171'),
            hovertemplate=t("Year: %{x}<br>Return: %{y:.1%}<extra></extra>", "Anno: %{x}<br>Rendimento: %{y:.1%}<extra></extra>")
        ))
        fig_yearly.add_trace(go.Bar(
            x=years, y=b_yearly, name='Shadow Benchmark',
            marker_color='#9ca3af', opacity=0.6,
            hovertemplate=t("Year: %{x}<br>Return: %{y:.1%}<extra></extra>", "Anno: %{x}<br>Rendimento: %{y:.1%}<extra></extra>")
        ))
        
        fig_yearly.update_layout(
            template="plotly_dark", height=350, barmode='group',
            margin=dict(t=30, b=10, l=10, r=10), yaxis_tickformat=".1%",
            xaxis=dict(type='category'), 
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

    with tab_quant:
        st.subheader(t("Statistical Decomposition and Pure Alpha", "Scomposizione Statistica e Alpha Puro"))
        c_q1, c_q2 = st.columns([1, 2])
        with c_q1:
            st.markdown(t("#### Benchmark Composition", "#### Composizione Benchmark"))
            for k, v in f_data["RBSA Weights"].items():
                if v > 0.01:
                    display_key = t(k, "Azionario Globale" if "Equity" in k else "Obbligazionario" if "Bond" in k else "Liquidità")
                    st.markdown(f"- **{display_key}**: `{v*100:.1f}%`")
                    
        with c_q2:
            st.markdown(t("#### Econometric Metrics", "#### Metriche Econometriche"))
            df_q = pd.DataFrame({
                t("Annualized Return", "Rendimento Annuo"): [f"{f_data['Annualized Return']:.2%}"],
                "Pure Alpha (Ann)": [f"{f_data['Alpha (Ann)']:.2%}"],
                "Info Ratio": [f"{f_data['Info Ratio']:.2f}"],
                "R² Adjusted": [f"{f_data['R² vs Benchmark']:.2f}"]
            })
            st.dataframe(df_q, hide_index=True, use_container_width=True)
            
        st.markdown("---")
        st.markdown(t("#### 📉 Linear Regression (Monthly Returns)", "#### 📉 Regressione Lineare (Rendimenti Mensili)"))
        st.markdown(t("<small><b>How to read this:</b> This is the fund's fingerprint. If dots form a tight straight line, the fund is just copying the index. Scattered dots mean the manager is making active bets. The red line shows the trend: if it crosses the vertical axis above zero, the manager is creating pure 'Alpha'.</small>", "<small><b>Come leggerlo:</b> Questa è l'impronta digitale del fondo. Se i punti formano una retta stretta, il fondo sta copiando l'indice. Punti sparsi indicano che il gestore fa scommesse attive. La linea rossa indica la tendenza: se incrocia l'asse verticale sopra lo zero, il gestore crea 'Alpha' puro.</small>"), unsafe_allow_html=True)
        
        f_monthly = f_data["F_Rets"].resample('ME').apply(lambda x: (1+x).prod()-1)
        b_series = pd.Series(f_data["B_Rets"], index=f_data["F_Rets"].index)
        b_monthly = b_series.resample('ME').apply(lambda x: (1+x).prod()-1)
        
        m, b = np.polyfit(b_monthly, f_monthly, 1)
        trendline_x = np.array([b_monthly.min(), b_monthly.max()])
        trendline_y = m * trendline_x + b

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=b_monthly, y=f_monthly, mode='markers', name=t('Historical Months', 'Mesi Storici'),
            marker=dict(color='#3B82F6', size=8, opacity=0.7), hovertemplate="Benchmark: %{x:.2%}<br>Fund: %{y:.2%}<extra></extra>"
        ))
        fig_scatter.add_trace(go.Scatter(
            x=trendline_x, y=trendline_y, mode='lines', name=f'Regression (Beta: {m:.2f})',
            line=dict(color='#F87171', width=3, dash='dot')
        ))
        fig_scatter.update_layout(
            template="plotly_dark", height=500, xaxis_title=t("Shadow Benchmark Return", "Rendimento Shadow Benchmark"),
            yaxis_title=t("Fund Return", "Rendimento Fondo"), xaxis_tickformat=".1%", yaxis_tickformat=".1%",
            margin=dict(t=30, b=10, l=10, r=10), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig_scatter.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        fig_scatter.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.markdown(t("#### 🔄 3-Year Rolling Alpha (Consistency)", "#### 🔄 Alpha Rolling a 3 Anni (Consistenza)"))
        st.markdown(t("<small><b>How to read this:</b> This answers the question 'If I held the fund for exactly 3 years, would I have beaten the market?'. Green mountains show periods where the manager consistently won. Red valleys show periods where they lost. This removes the illusion of lucky start dates.</small>", "<small><b>Come leggerlo:</b> Risponde alla domanda 'Se avessi tenuto il fondo per 3 anni esatti, avrei battuto il mercato?'. Le montagne verdi mostrano i cicli in cui il gestore ha vinto con costanza. I burroni rossi indicano le sconfitte. Questo elimina totalmente l'illusione ottica di una fortunata data di inizio.</small>"), unsafe_allow_html=True)

        rolling_alpha = f_data.get("Rolling Alpha")
        if rolling_alpha is not None and not rolling_alpha.empty:
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=rolling_alpha.index, y=rolling_alpha.clip(lower=0),
                mode='lines', line=dict(color='#34d399', width=0),
                fill='tozeroy', fillcolor='rgba(52, 211, 153, 0.4)', name='Outperformance'
            ))
            fig_roll.add_trace(go.Scatter(
                x=rolling_alpha.index, y=rolling_alpha.clip(upper=0),
                mode='lines', line=dict(color='#f87171', width=0),
                fill='tozeroy', fillcolor='rgba(248, 113, 113, 0.4)', name='Underperformance'
            ))
            fig_roll.add_trace(go.Scatter(
                x=rolling_alpha.index, y=rolling_alpha,
                mode='lines', line=dict(color='#ffffff', width=1.5), showlegend=False,
                hovertemplate=t("Date: %{x}<br>Rolling Alpha: %{y:.2%}<extra></extra>", "Data: %{x}<br>Alpha Rolling: %{y:.2%}<extra></extra>")
            ))
            fig_roll.update_layout(
                template="plotly_dark", height=350,
                margin=dict(t=10, b=10, l=10, r=10), yaxis_tickformat=".1%", hovermode="x unified"
            )
            fig_roll.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.5)")
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info(t("Not enough historical data to compute 3-Year Rolling Alpha.", "Storico insufficiente per calcolare l'Alpha Rolling a 3 anni."))

        st.markdown("---")
        st.markdown(t("#### 📊 Return Distribution (Tails & Skewness)", "#### 📊 Distribuzione dei Rendimenti (Asimmetria e Code)"))
        st.markdown(t("<small><b>How to read this:</b> Imagine putting all past months into 'buckets' based on their returns. A good fund should have fewer months in the extreme left buckets (crashes) and more months on the right side compared to the benchmark.</small>", "<small><b>Come leggerlo:</b> Immagina di mettere tutti i mesi passati in dei 'secchi' in base a quanto hanno reso. Un buon fondo dovrebbe avere meno mesi nei secchi di estrema sinistra (i crolli) e più mesi nella parte destra rispetto al benchmark.</small>"), unsafe_allow_html=True)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=f_monthly, name=t('Fund', 'Fondo'), marker_color='#34d399', opacity=0.7, nbinsx=50,
            hovertemplate=t("Return: %{x:.1%}<br>Count: %{y}<extra></extra>", "Rendimento: %{x:.1%}<br>Mesi: %{y}<extra></extra>")
        ))
        fig_hist.add_trace(go.Histogram(
            x=b_monthly, name='Shadow Benchmark', marker_color='#9ca3af', opacity=0.5, nbinsx=50,
            hovertemplate=t("Return: %{x:.1%}<br>Count: %{y}<extra></extra>", "Rendimento: %{x:.1%}<br>Mesi: %{y}<extra></extra>")
        ))
        fig_hist.update_layout(
            template="plotly_dark", height=350, barmode='overlay',
            margin=dict(t=10, b=10, l=10, r=10), xaxis_tickformat=".1%",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
