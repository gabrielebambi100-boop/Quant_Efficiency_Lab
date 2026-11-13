import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.optimize import minimize
import requests

st.set_page_config(page_title="Quant Efficiency Lab", layout="wide", page_icon="🔬")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {background-color: #0E1117;}
</style>
""", unsafe_allow_html=True)

# --- LINGUA ---
with st.sidebar:
    lang_choice = st.radio("🌐 Language / Lingua", ["🇮🇹 Italiano", "🇬🇧 English"], horizontal=True)
    is_it = "Italiano" in lang_choice


def t(en, it):
    return it if is_it else en


# =====================================================================
# CORE MATH
# =====================================================================
TRADING_DAYS = 252


def resolve_ticker(query):
    """Usa la ricerca interna di Yahoo Finance per tradurre un ISIN in ticker."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    try:
        data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5).json()
        if data.get('quotes'):
            return data['quotes'][0]['symbol']
    except Exception as e:
        print(f"Errore risoluzione ticker {query}: {e}")
    return query


def ann_return(rets):
    return (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1


def compound(rets, freq):
    """Rendimento composto per periodo ('ME' = mensile, 'YE' = annuale)."""
    return (1 + rets).resample(freq).prod() - 1


def calc_drawdown(wealth_index):
    peak = wealth_index.cummax()
    return (wealth_index - peak) / peak


def risk_stats(rets, rf_daily):
    ret = ann_return(rets)
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret - rf_daily * TRADING_DAYS) / vol if vol > 0 else 0
    max_dd = calc_drawdown((1 + rets).cumprod()).min()
    return ret, vol, sharpe, max_dd


def style_analysis(fund_rets, factors_rets):
    """RBSA: pesi long-only che sommano a 1 e minimizzano l'errore quadratico di tracciamento."""
    n = factors_rets.shape[1]
    res = minimize(
        lambda w: np.sum((fund_rets - factors_rets @ w) ** 2),
        np.ones(n) / n, method='SLSQP', bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    )
    return res.x


def analyze_instrument(fund_prices, bench_df, ter, rbsa):
    df = pd.concat([fund_prices, bench_df], axis=1).dropna()
    if len(df) < TRADING_DAYS:
        return None

    rets = df.pct_change().dropna()
    f_rets, bench_rets = rets.iloc[:, 0], rets.iloc[:, 1:]

    if rbsa:
        weekly = df.resample('W').last().pct_change().dropna()
        weights = style_analysis(weekly.iloc[:, 0].values, weekly.iloc[:, 1:].values)
        rf_daily = bench_rets.iloc[:, 2].mean()
    else:
        weights = np.array([1.0])
        rf_daily = 0.02 / TRADING_DAYS

    b_rets = bench_rets.dot(weights)  # Serie con lo stesso indice di f_rets
    weight_dict = dict(zip(["equity", "bonds", "cash"], np.pad(weights, (0, 3 - len(weights)))))

    ann_ret, vol, sharpe, max_dd = risk_stats(f_rets, rf_daily)
    b_ann_ret, b_vol, b_sharpe, b_max_dd = risk_stats(b_rets, rf_daily)

    # Regressione OLS sugli excess returns: l'intercetta è l'alpha
    reg = stats.linregress(b_rets - rf_daily, f_rets - rf_daily)
    r_squared = reg.rvalue ** 2
    t_stat = reg.intercept / reg.intercept_stderr
    p_value = 2 * stats.t.sf(abs(t_stat), df=len(f_rets) - 2)

    active_fee = ter / (1 - r_squared) if (1 - r_squared) > 0.01 else 9.99
    te = (f_rets - b_rets).std() * np.sqrt(TRADING_DAYS)
    ir = (ann_ret - b_ann_ret) / te if te > 0.0005 else 0.0

    # Down capture rispetto al primo proxy (azionario)
    eq = bench_rets.iloc[:, 0]
    down = eq < 0
    down_cap = f_rets[down].mean() / eq[down].mean() if down.any() else 0

    # Rolling 36 mesi: somma dei log-rendimenti invece di apply(lambda)
    f_monthly, b_monthly = compound(f_rets, 'ME'), compound(b_rets, 'ME')
    roll = lambda s: np.expm1(np.log1p(s).rolling(36).sum())
    rolling_alpha = (roll(f_monthly) - roll(b_monthly)).dropna()
    win_rate = (rolling_alpha > 0).mean() if len(rolling_alpha) else None

    return {
        "ann_ret": ann_ret, "b_ann_ret": b_ann_ret,
        "vol": vol, "b_vol": b_vol,
        "sharpe": sharpe, "b_sharpe": b_sharpe,
        "max_dd": max_dd, "b_max_dd": b_max_dd,
        "ter": ter, "weights": weight_dict,
        "r2": r_squared, "active_fee": active_fee,
        "alpha": reg.intercept * TRADING_DAYS, "ir": ir,
        "p_value": p_value, "down_cap": down_cap,
        "win_rate": win_rate, "rolling_alpha": rolling_alpha,
        "start": df.index[0].strftime('%Y-%m-%d'), "n_days": len(f_rets),
        "f_monthly": f_monthly, "b_monthly": b_monthly,
        "f_yearly": compound(f_rets, 'YE'), "b_yearly": compound(b_rets, 'YE'),
    }


# =====================================================================
# SOGLIE E VERDETTI (dati, non if a catena)
# Ogni regola: (condizione, icona, (titolo EN, titolo IT), (descrizione EN, descrizione IT)).
# Vale la prima regola vera; l'ultima è il caso di default.
# =====================================================================
STATUS_RULES = {
    "overlap": [
        (lambda v: v > 0.85, "🔴", ("Closet Indexer", "Finto Attivo (Closet Indexer)"),
         ("It's a clone. Tracks the index but charges active management fees.", "È un clone. Copia l'indice ma fa pagare commissioni da gestione attiva.")),
        (lambda v: v > 0.60, "🟡", ("Hybrid Management", "Gestione Ibrida"),
         ("Deviates partially, but the core engine remains the base index.", "Devia parzialmente, ma il motore principale resta l'indice di base.")),
        (lambda v: True, "🟢", ("Autonomous Fund", "Fondo Autonomo"),
         ("Makes truly distinct choices, justifying active management.", "Compie scelte nettamente distinte, giustificando la gestione attiva.")),
    ],
    "active_fee": [
        (lambda v: v > 0.04, "🔴", ("Disproportionate Cost", "Costo Spropositato"),
         ("Requires unrealistic skill to recover this fee drag.", "Richiede un'abilità irrealistica per recuperare questa zavorra di costi.")),
        (lambda v: v > 0.025, "🟡", ("High Burden", "Fardello Elevato"),
         ("The active bet is expensive.", "La scommessa attiva è costosa.")),
        (lambda v: True, "🟢", ("Fair Pricing", "Prezzo Equo"),
         ("You are paying a fair price for the active risk taken.", "Stai pagando un prezzo giusto per il rischio attivo che il gestore si prende.")),
    ],
    "ir": [
        (lambda v: v < 0, "🔴", ("Value Destructor", "Distruttore di Valore"),
         ("Higher volatility than the index and lower returns.", "Volatilità più alta dell'indice e rendimenti inferiori.")),
        (lambda v: v < 0.5, "🟡", ("Inefficient", "Inefficiente"),
         ("Active risk is not adequately rewarded.", "Il rischio attivo preso non è adeguatamente ricompensato.")),
        (lambda v: True, "🟢", ("Efficient", "Efficiente"),
         ("The active bet pays off significantly.", "La scommessa attiva ripaga in modo significativo.")),
    ],
    "airbag": [
        (lambda v: v > 0.95, "🔴", ("No Airbag", "Nessun Airbag"),
         ("When it rains, you get just as wet as with a passive ETF.", "Quando piove, ti bagni esattamente come con un ETF passivo.")),
        (lambda v: v > 0.80, "🟡", ("Weak Protection", "Protezione Debole"),
         ("Only partially cushions market crashes.", "Attutisce solo in minima parte i crolli di mercato.")),
        (lambda v: True, "🟢", ("True Protection", "Vera Protezione"),
         ("Anti-crisis fund, sharply cuts market drawdowns.", "Fondo anti-crisi, taglia drasticamente i crolli del mercato.")),
    ],
    "p_value": [
        (lambda v: v > 0.10, "🔴", ("Pure Luck", "Pura Fortuna"),
         ("No mathematical proof that the manager can consistently beat the market.", "Nessuna prova matematica che il gestore sappia battere il mercato con costanza.")),
        (lambda v: v <= 0.05, "🟢", ("Solid Skill", "Abilità Solida"),
         ("Excellent statistical significance.", "Eccellente significatività statistica.")),
        (lambda v: True, "🟡", ("Gray Zone", "Zona Grigia"),
         ("Positive results, but the doubt of luck persists.", "Risultati positivi, ma resta il dubbio che sia stato un caso.")),
    ],
    "consistency": [
        (lambda v: v is None, "⚪", ("Not Enough Data", "Dati Insufficienti"),
         ("Requires at least 3 years of history.", "Richiede almeno 3 anni di storico.")),
        (lambda v: v > 0.75, "🟢", ("Consistent Winner", "Vincente Sistematico"),
         ("Reliably beats the benchmark across rolling 3-year periods.", "Batte sistematicamente il benchmark su orizzonti di 3 anni.")),
        (lambda v: v < 0.40, "🔴", ("Chronic Underperformer", "Perdente Cronico"),
         ("Holding this for 3 years historically results in losing to the index.", "Tenere questo fondo per 3 anni porta storicamente a perdere contro l'indice.")),
        (lambda v: True, "🟡", ("Coin Flip", "Lancio della Moneta"),
         ("Chances of beating the index over 3 years are near 50/50.", "Le probabilità di battere l'indice su 3 anni sono vicine al 50/50.")),
    ],
}


def get_retail_status(metric, value):
    for cond, icon, title, desc in STATUS_RULES[metric]:
        if cond(value):
            return icon, t(*title), t(*desc)


# Soglie punteggio: (minimo, lettera, (EN, IT))
GRADES = [
    (80, "S", ("🟢 OUTSTANDING: Rare excellence. Fully justifies the fees.", "🟢 ECCELLENTE: Rara eccellenza. Giustifica pienamente le commissioni.")),
    (65, "A", ("🟢 GREAT: Generates real, solid, and statistically valid value.", "🟢 OTTIMO: Genera valore reale, solido e statisticamente valido.")),
    (50, "B", ("🟡 FAIR: Does its job, but consider if a passive ETF is cheaper.", "🟡 DISCRETO: Fa il suo lavoro, ma valuta se un ETF passivo ti costerebbe meno.")),
    (35, "C", ("🟠 MEDIOCRE: Inefficient. You are paying for results mostly dictated by chance.", "🟠 MEDIOCRE: Inefficiente. Paghi caro per risultati spesso dettati dal caso.")),
    (20, "D", ("🔴 POOR: Destroys value, costs too much, and doesn't protect in crashes.", "🔴 SCARSO: Distrugge valore, costa troppo e non protegge nei crolli.")),
    (0, "F", ("☠️ TOXIC: Sell immediately. Capital bleed guaranteed.", "☠️ TOSSICO: Da vendere subito. Sanguinamento di capitale garantito.")),
]


def get_global_score(m):
    if m["r2"] > 0.80 and m["ter"] <= 0.02:
        return 85.0, "P", t("🔵 PASSIVE ETF: Honest and efficient tracker. Excellent as a core portfolio engine.",
                            "🔵 ETF PASSIVO: Tracker onesto ed efficiente. Eccellente come motore base del portafoglio.")

    score = 50.0 + m["ir"] * 25
    score += 10 if m["p_value"] <= 0.05 else -10 if m["p_value"] > 0.10 else 0
    score += 10 if m["down_cap"] < 0.80 else -10 if m["down_cap"] > 1.0 else 0
    score -= 15 if m["active_fee"] > 0.04 else 0
    if m["win_rate"] is not None:
        score += 15 if m["win_rate"] >= 0.70 else -15 if m["win_rate"] < 0.40 else 0
    score = max(0, min(100, score))

    for minimum, letter, verdict in GRADES:
        if score >= minimum:
            return score, letter, t(*verdict)


def weight_label(key):
    return {"equity": t("Global Equity", "Azionario Globale"),
            "bonds": t("Bonds", "Obbligazionario"),
            "cash": t("Cash / Liquidity", "Liquidità")}[key]


# =====================================================================
# DATA LOADING
# =====================================================================
def load_prices(tickers):
    prices, names = {}, {}
    for tk in tickers:
        raw = yf.download(tk, period="max", progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        raw.index = raw.index.normalize()
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        if col in raw.columns:
            prices[tk] = raw[col].dropna()
            names[tk] = yf.Ticker(tk).info.get("longName", tk)
    return pd.DataFrame(prices), names


def run_analysis(inputs, bench_inputs, rbsa):
    """Scarica i dati, analizza ogni strumento e salva tutto in session_state."""
    fund_map = {raw: resolve_ticker(raw) for raw in inputs["Ticker"]}
    bench_tickers = [resolve_ticker(b) for b in bench_inputs]

    data, names = load_prices(set(fund_map.values()) | set(bench_tickers))
    if data.empty:
        st.error(t("❌ No data found. Yahoo Finance is unresponsive or tickers are invalid.",
                   "❌ Nessun dato trovato. Yahoo Finance non risponde o i ticker non sono validi."))
        return

    missing = [b for b in bench_tickers if b not in data.columns]
    if missing:
        st.error(t(f"❌ Cannot analyze: Yahoo Finance missing benchmark data for {', '.join(missing)}",
                   f"❌ Impossibile analizzare: dati benchmark mancanti per {', '.join(missing)}"))
        return
    bench_df = data[bench_tickers].dropna()

    results, skipped = {}, []
    for raw, ter_pct in zip(inputs["Ticker"], inputs["TER (%)"]):
        resolved = fund_map[raw]
        if resolved not in data.columns:
            skipped.append(f"**{raw}**: " + t("ISIN/Ticker not found in Yahoo Finance database.",
                                              "ISIN/Ticker non trovato nel database di Yahoo Finance."))
            continue
        m = analyze_instrument(data[resolved], bench_df, ter_pct / 100 if pd.notna(ter_pct) else 0.0, rbsa)
        if m is None:
            skipped.append(f"**{raw}**: " + t("Time series too short or fragmented (min 1 year required).",
                                              "Serie storica troppo breve o frammentata (richiesto minimo 1 anno)."))
            continue
        results[raw] = m
        names[raw] = names.get(resolved, resolved)

    if skipped:
        st.warning(t("⚠️ **Some instruments were skipped:**\n", "⚠️ **Alcuni strumenti sono stati ignorati:**\n")
                   + "\n".join(f"- {s}" for s in skipped))
    if not results:
        st.error(t("No valid instruments for analysis.", "Nessuno strumento valido per l'analisi."))
        return

    st.session_state.results, st.session_state.names, st.session_state.rbsa = results, names, rbsa


# =====================================================================
# UI
# =====================================================================
st.title(t("🔬 Quant Efficiency Lab", "🔬 Laboratorio di Efficienza Quant"))
st.header(t("📦 Instruments to Evaluate", "📦 Strumenti da Valutare"))

with st.expander(t("💡 Ideas: Famous Funds to test", "💡 Spunti: Fondi famosi da testare")):
    st.markdown(t(
        """
Copy these famous Tickers into the table below to stress-test the algorithm:
* 🟢 **The Passive Pillars:** `VT` (Total World), `SPY` (S&P 500). *Some of the most capitalized global and US trackers in the world.*
* 🟠 **The Active Giants:** `FCNTX` (Fidelity), `AMCPX` (American Funds). *Historical actively managed funds with massive AUM.*
* 🔴 **The Thematic Bets:** `ARKK` (Ark Innovation), `ICLN` (Clean Energy). *High-volatility funds tied to specific sectors or mega-trends.*
* 🟡 **The Dividend Funds:** `JEPI` (JPMorgan Equity Premium). *Highly popular for their high-yield income distribution strategies.*
""",
        """
Copia questi Ticker nella tabella sottostante per mettere alla prova l'algoritmo:
* 🟢 **I Pilastri Passivi:** `VT` (Total World), `SPY` (S&P 500). *Tracker globali e statunitensi tra i più capitalizzati al mondo.*
* 🟠 **I Colossi Attivi:** `FCNTX` (Fidelity), `AMCPX` (American Funds). *Fondi a gestione attiva storici e con enormi masse in gestione.*
* 🔴 **Le Scommesse Tematiche:** `ARKK` (Ark Innovation), `ICLN` (Clean Energy). *Fondi ad alta volatilità legati a specifici settori o mega-trend.*
* 🟡 **I Fondi da Dividendo:** `JEPI` (JPMorgan Equity Premium). *Molto popolari per la strategia basata sulla distribuzione di alte cedole.*
"""))

edited_df = st.data_editor(
    pd.DataFrame({"Ticker": ["VT", "ARKK", "JEPI"], "TER (%)": [0.07, 0.75, 0.35]}),
    num_rows="dynamic", use_container_width=True)

run = st.button(t("Run X-Ray Analysis", "Avvia Analisi a Raggi X"), type="primary", use_container_width=True)

# --- SIDEBAR: impostazioni tecniche ---
with st.sidebar:
    st.header(t("⚙️ Analysis Engine", "⚙️ Motore di Analisi"))

    currency = st.selectbox(t("Currency / Region Setup", "Valuta / Area Geografica"),
                            ["🇺🇸 USD (Default - US/Global)", "🇪🇺 EUR (European UCITS)"])
    def_eq, def_bd, def_cs = ("VT", "AGG", "BIL") if "USD" in currency else ("EUNL.DE", "DBZB.DE", "XEON.DE")

    rbsa_label = t("🤖 Shadow ETF (RBSA)", "🤖 Shadow ETF (RBSA Sintetico)")
    direct_label = t("🎯 Single Benchmark (Direct)", "🎯 Benchmark Singolo (Diretto)")
    use_rbsa = st.radio(t("Comparison Method:", "Metodo di Confronto:"), [rbsa_label, direct_label]) == rbsa_label

    if use_rbsa:
        st.caption(t("The engine deconstructs the fund and builds a synthetic benchmark.",
                     "Il motore decostruisce il fondo e crea un benchmark sintetico su misura."))
        st.info(t("ℹ️ **RBSA Constraints:** The model uses constrained optimization to build a realistic, long-only shadow portfolio (weights must be ≥ 0 and sum exactly to 100%).",
                  "ℹ️ **Vincoli RBSA:** Il modello utilizza un'ottimizzazione vincolata per costruire un portafoglio ombra long-only realistico (i pesi devono essere ≥ 0 e sommare al 100%)."))
        bench_inputs = [st.text_input(t("Equity Proxy", "Proxy Azionario"), def_eq),
                        st.text_input(t("Bond Proxy", "Proxy Obbligazionario"), def_bd),
                        st.text_input(t("Cash Proxy", "Proxy Liquidità"), def_cs)]
    else:
        st.caption(t("1-vs-1 Match: Fund against your chosen Index.", "Scontro 1-vs-1: Il fondo contro l'indice che hai scelto."))
        bench_inputs = [st.text_input(t("Direct Benchmark Ticker", "Ticker del Benchmark"), def_eq)]
    bench_inputs = [b.strip().upper() for b in bench_inputs]

    st.caption(t("📉 *Risk-Free rate is dynamically calculated based on the selected Cash Proxy to reflect different market regimes.*",
                 "📉 *Il tasso Risk-Free è calcolato dinamicamente sulla base del Proxy Liquidità scelto per riflettere i diversi regimi di mercato.*"))

    with st.expander(t("📚 How the engine works (RBSA Methodology)", "📚 Come funziona l'algoritmo (Metodologia RBSA)")):
        st.markdown(t(
            """
**Returns-Based Style Analysis (RBSA)** is a quantitative technique introduced by Nobel laureate William Sharpe.
The software uses Sequential Least Squares Programming (SLSQP) to find the exact asset mix (constrained between 0% and 100%)
that minimizes the variance of the tracking error compared to the analyzed fund.
- **Active Fee Proxy:** Calculated by isolating the total expense ratio (TER) strictly on the active share of the portfolio ($1 - R^2$).
- **P-Value:** Derived from the T-Statistic of the OLS regression intercept, testing the null hypothesis that Alpha equals zero.
""",
            """
**Returns-Based Style Analysis (RBSA)** è una tecnica quantitativa introdotta dal premio Nobel William Sharpe.
Il software utilizza l'ottimizzazione quadratica sequenziale (SLSQP) per calcolare il mix esatto di indici passivi (vincolato tra 0% e 100%) che minimizza la varianza dell'errore di tracciamento rispetto al fondo.
- **Active Fee Proxy:** Stima il vero costo percentuale caricato solo sulla parte di portafoglio gestita attivamente ($1 - R^2$).
- **P-Value:** Derivato dalla Statistica-T dell'intercetta della regressione (OLS), valuta se l'Alpha prodotto è statisticamente diverso da zero o se è puro rumore.
"""))

# --- ESECUZIONE ---
if run:
    with st.spinner(t("Translating ISINs, aligning dates, and running quantitative analysis...",
                      "Traduzione ISIN, allineamento date e calcolo quantitativo in corso...")):
        try:
            inputs = edited_df.dropna(subset=["Ticker"]).copy()
            inputs["Ticker"] = inputs["Ticker"].str.strip()
            run_analysis(inputs, bench_inputs, use_rbsa)
        except Exception as e:
            st.error(f"Critical Error: {e}")

# --- RISULTATI ---
if "results" in st.session_state:
    res, names = st.session_state.results, st.session_state.names
    rbsa_used = st.session_state.rbsa

    st.divider()
    selected = st.selectbox(t("Select the instrument to inspect:", "Seleziona lo strumento da ispezionare:"),
                            list(res.keys()), format_func=lambda x: f"{names.get(x, x)} ({x})")
    m = res[selected]

    tab_retail, tab_quant = st.tabs([t("👨‍💼 Investor Dashboard (Intuitive)", "👨‍💼 Dashboard Investitore (Intuitiva)"),
                                     t("🔬 Quant Laboratory (Statistical)", "🔬 Laboratorio Quantitativo (Statistico)")])

    with tab_retail:
        score, letter, verdict = get_global_score(m)
        st.markdown(f"## {t('Global Score', 'Punteggio Globale')}: **{letter}** ({score:.0f}/100)")

        box_color, text_color = (("#1e3a8a", "#60a5fa") if letter == "P" else
                                 ("#052e16", "#34d399") if score >= 65 else
                                 ("#451a03", "#fbbf24") if score >= 35 else
                                 ("#450a0a", "#f87171"))
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; border: 1px solid {text_color}; margin-bottom: 25px;">
            <h4 style="color: {text_color}; margin: 0;">{verdict}</h4>
        </div>
        """, unsafe_allow_html=True)

        st.subheader(t("Real-World Performance Profile", "Profilo di Performance Reale"))
        bench_desc = (t("its real 'Shadow ETF'", "il suo 'Shadow ETF' reale") if rbsa_used
                      else t("your selected benchmark", "il benchmark selezionato"))
        st.markdown(t(f"Historical data validated since **{m['start']}**. The fund was analyzed against {bench_desc}.",
                      f"Dati storici validati dal **{m['start']}**. Il fondo è stato analizzato contro {bench_desc}."))

        st.markdown("---")
        st.markdown(t("### 📊 Absolute Performance (Fund vs Benchmark)", "### 📊 Performance Assoluta (Fondo vs Benchmark)"))

        pct = lambda x: f"{x * 100:.1f}%"
        abs_metrics = [
            (t("Annual Return", "Rendimento Annuo"), pct(m["ann_ret"]), pct(m["b_ann_ret"])),
            (t("Volatility (Ann)", "Volatilità (Annua)"), pct(m["vol"]), pct(m["b_vol"])),
            ("Sharpe Ratio", f"{m['sharpe']:.2f}", f"{m['b_sharpe']:.2f}"),
            (t("Max Drawdown", "Max Drawdown (Crollo Max)"), pct(m["max_dd"]), pct(m["b_max_dd"])),
        ]
        for col, (label, value, bench) in zip(st.columns(4), abs_metrics):
            col.metric(label, value, f"Bench: {bench}", delta_color="off")

        if m["n_days"] < TRADING_DAYS * 3:
            st.warning(t("⚠️ **Short Track Record:** The instrument has less than 3 years of historical data. Statistical metrics like P-Value and Rolling Win Rate may not be fully reliable and could be driven by market noise.",
                         "⚠️ **Track Record Breve:** Lo strumento ha meno di 3 anni di storico. Metriche statistiche come il P-Value e il Win Rate potrebbero non essere affidabili e derivare dal rumore di mercato."))

        if rbsa_used:
            st.markdown("---")
            st.markdown(t("### 🧬 Fund DNA (Under the Hood)", "### 🧬 DNA del Fondo (Sotto il Cofano)"))
            w_eq, w_bd, w_cs = (round(m["weights"][k], 4) for k in ("equity", "bonds", "cash"))

            col_text, col_chart = st.columns(2)
            with col_text:
                st.write("")
                st.metric(t("📈 Equity Weight", "📈 Peso Azionario"), pct(w_eq))
                st.metric(t("🛡️ Bond Weight", "🛡️ Peso Obbligazionario"), pct(w_bd))
                st.metric(t("💶 Cash / Liquidity", "💶 Peso Liquidità"), pct(w_cs))
                if w_eq >= 0.75:
                    st.warning(t(f"⚠ **WARNING:** This instrument is **{w_eq*100:.0f}%** driven by Global Equities. Do not be fooled by thematic narratives: if global stock markets crash, this fund will crash with them.",
                                 f"⚠ **ATTENZIONE:** Questo strumento è guidato per il **{w_eq*100:.0f}%** dall'Azionario Globale. Non farti ingannare dalle narrazioni: se le borse crollano, questo fondo crollerà con loro."))
                elif w_eq <= 0.30:
                    st.success(t("✅ **UNCORRELATED:** This instrument operates with an engine profoundly different from standard equities.",
                                 "✅ **DECORRELATO:** Questo strumento opera con un motore profondamente diverso dalle classiche azioni."))
            with col_chart:
                fig = go.Figure(go.Pie(
                    labels=[t("Equity", "Azioni"), t("Bonds", "Obbligazioni"), t("Cash", "Liquidità")],
                    values=[w_eq, w_bd, w_cs], hole=.6,
                    marker_colors=['#3B82F6', '#8B5CF6', '#10B981'],
                    hovertemplate="%{label}: %{value:.1%}<extra></extra>"))
                fig.update_layout(template="plotly_dark", height=250, margin=dict(t=10, b=10, l=10, r=10),
                                  showlegend=False,
                                  annotations=[dict(text='DNA', x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(color="white"))])
                st.plotly_chart(fig, use_container_width=True)

        # --- Le sei schede di verdetto, definite come dati ---
        fee_txt = pct(m["active_fee"]) if m["active_fee"] < 9 else "> 10%"
        win_txt = pct(m["win_rate"]) if m["win_rate"] is not None else "N/A"
        cards = [
            ("overlap", t("Index Overlap", "Sovrapposizione all'Indice"),
             t("How much is this fund copying the market instead of actively managing it?", "Quanto questo fondo copia il mercato invece di gestirlo attivamente?"),
             t("Correlation to Index", "Correlazione all'Indice"), pct(m["r2"]), m["r2"]),
            ("active_fee", t("Active Fee Proxy", "Stima Costo Attivo"),
             t("On the truly active portion, what hidden commission rate are you paying?", "Sulla porzione puramente attiva, che commissione nascosta stai pagando?"),
             t("Estimated Active Fee", "Costo Attivo Stimato"), fee_txt, m["active_fee"]),
            ("airbag", t("The Airbag Test", "Il Test dell'Airbag"),
             t("During market sell-offs, does the fund avoid the crash or fall with everyone else?", "Durante i crolli di mercato, il fondo li evita o cade insieme a tutti gli altri?"),
             "Drawdown Capture", pct(m["down_cap"]), m["down_cap"]),
            ("ir", t("Efficiency Index", "Indice di Efficienza"),
             t("For every sleepless night caused by active risk, how much extra return was generated?", "Per ogni notte insonne causata dal rischio attivo, quanto extra rendimento è stato generato?"),
             "Information Ratio", f"{m['ir']:.2f}", m["ir"]),
            ("p_value", t("Skill or Luck?", "Abilità o Fortuna?"),
             t("Probability that the positive performance is just a fortunate coincidence.", "Probabilità che le performance positive siano solo una fortunata coincidenza."),
             t("Luck Risk (P-Value)", "Rischio Fortuna (P-Value)"), pct(m["p_value"]), m["p_value"]),
            ("consistency", t("3Y Consistency", "Consistenza a 3 Anni"),
             t("How often does it beat the market over a 3-year holding period?", "Quante volte batte il mercato se lo tieni in portafoglio per 3 anni?"),
             t("Rolling Win Rate", "Win Rate Rolling"), win_txt, m["win_rate"]),
        ]
        for row in (cards[:3], cards[3:]):
            st.markdown("---")
            for col, (key, title, question, metric_label, shown, value) in zip(st.columns(3), row):
                icon, name, desc = get_retail_status(key, value)
                with col:
                    st.markdown(f"### {icon} {title}")
                    st.markdown(f"**{question}**")
                    st.metric(metric_label, shown)
                    st.markdown(f"> **{t('Verdict', 'Verdetto')}:** {name}. {desc}")

        st.markdown("---")
        st.markdown(t("### 📅 Year-by-Year Performance", "### 📅 Performance Anno per Anno"))
        st.markdown(t("<small>Shows the calendar year returns. A quick way to see if the fund manages to protect capital during crisis years (e.g., 2022) and keep up during bull markets.</small>",
                      "<small>Mostra i rendimenti per anno solare. Un modo rapido per vedere se il fondo protegge il capitale negli anni di crisi (es. 2022) e se tiene il passo nei mercati toro.</small>"),
                    unsafe_allow_html=True)

        hover_year = t("Year: %{x}<br>Return: %{y:.1%}<extra></extra>", "Anno: %{x}<br>Rendimento: %{y:.1%}<extra></extra>")
        years = m["f_yearly"].index.year
        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=m["f_yearly"], name=t('Fund', 'Fondo'),
                             marker_color=np.where(m["f_yearly"] >= 0, '#34d399', '#f87171'), hovertemplate=hover_year))
        fig.add_trace(go.Bar(x=years, y=m["b_yearly"], name='Shadow Benchmark',
                             marker_color='#9ca3af', opacity=0.6, hovertemplate=hover_year))
        fig.update_layout(template="plotly_dark", height=350, barmode='group', margin=dict(t=30, b=10, l=10, r=10),
                          yaxis_tickformat=".1%", xaxis=dict(type='category'),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)

    with tab_quant:
        st.subheader(t("Statistical Decomposition and Pure Alpha", "Scomposizione Statistica e Alpha Puro"))
        c_q1, c_q2 = st.columns([1, 2])
        with c_q1:
            st.markdown(t("#### Benchmark Composition", "#### Composizione Benchmark"))
            for k, v in m["weights"].items():
                if v > 0.01:
                    st.markdown(f"- **{weight_label(k)}**: `{v*100:.1f}%`")
        with c_q2:
            st.markdown(t("#### Econometric Metrics", "#### Metriche Econometriche"))
            st.dataframe(pd.DataFrame({
                t("Annualized Return", "Rendimento Annuo"): [f"{m['ann_ret']:.2%}"],
                "Pure Alpha (Ann)": [f"{m['alpha']:.2%}"],
                "Info Ratio": [f"{m['ir']:.2f}"],
                "R² Adjusted": [f"{m['r2']:.2f}"],
            }), hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown(t("#### 📉 Linear Regression (Monthly Returns)", "#### 📉 Regressione Lineare (Rendimenti Mensili)"))
        st.markdown(t("<small><b>How to read this:</b> This is the fund's fingerprint. If dots form a tight straight line, the fund is just copying the index. Scattered dots mean the manager is making active bets. The red line shows the trend: if it crosses the vertical axis above zero, the manager is creating pure 'Alpha'.</small>",
                      "<small><b>Come leggerlo:</b> Questa è l'impronta digitale del fondo. Se i punti formano una retta stretta, il fondo sta copiando l'indice. Punti sparsi indicano che il gestore fa scommesse attive. La linea rossa indica la tendenza: se incrocia l'asse verticale sopra lo zero, il gestore crea 'Alpha' puro.</small>"),
                    unsafe_allow_html=True)

        f_monthly, b_monthly = m["f_monthly"], m["b_monthly"]
        slope, intercept = np.polyfit(b_monthly, f_monthly, 1)
        line_x = np.array([b_monthly.min(), b_monthly.max()])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b_monthly, y=f_monthly, mode='markers', name=t('Historical Months', 'Mesi Storici'),
                                 marker=dict(color='#3B82F6', size=8, opacity=0.7),
                                 hovertemplate="Benchmark: %{x:.2%}<br>Fund: %{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode='lines', name=f'Regression (Beta: {slope:.2f})',
                                 line=dict(color='#F87171', width=3, dash='dot')))
        fig.update_layout(template="plotly_dark", height=500,
                          xaxis_title=t("Shadow Benchmark Return", "Rendimento Shadow Benchmark"),
                          yaxis_title=t("Fund Return", "Rendimento Fondo"),
                          xaxis_tickformat=".1%", yaxis_tickformat=".1%", margin=dict(t=30, b=10, l=10, r=10),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown(t("#### 🔄 3-Year Rolling Alpha (Consistency)", "#### 🔄 Alpha Rolling a 3 Anni (Consistenza)"))
        st.markdown(t("<small><b>How to read this:</b> This answers the question 'If I held the fund for exactly 3 years, would I have beaten the market?'. Green mountains show periods where the manager consistently won. Red valleys show periods where they lost. This removes the illusion of lucky start dates.</small>",
                      "<small><b>Come leggerlo:</b> Risponde alla domanda 'Se avessi tenuto il fondo per 3 anni esatti, avrei battuto il mercato?'. Le montagne verdi mostrano i cicli in cui il gestore ha vinto con costanza. I burroni rossi indicano le sconfitte. Questo elimina totalmente l'illusione ottica di una fortunata data di inizio.</small>"),
                    unsafe_allow_html=True)

        ra = m["rolling_alpha"]
        if ra.empty:
            st.info(t("Not enough historical data to compute 3-Year Rolling Alpha.",
                      "Storico insufficiente per calcolare l'Alpha Rolling a 3 anni."))
        else:
            fig = go.Figure()
            for series, color, fill, name in [(ra.clip(lower=0), '#34d399', 'rgba(52, 211, 153, 0.4)', 'Outperformance'),
                                              (ra.clip(upper=0), '#f87171', 'rgba(248, 113, 113, 0.4)', 'Underperformance')]:
                fig.add_trace(go.Scatter(x=ra.index, y=series, mode='lines', line=dict(color=color, width=0),
                                         fill='tozeroy', fillcolor=fill, name=name))
            fig.add_trace(go.Scatter(x=ra.index, y=ra, mode='lines', line=dict(color='#ffffff', width=1.5), showlegend=False,
                                     hovertemplate=t("Date: %{x}<br>Rolling Alpha: %{y:.2%}<extra></extra>",
                                                     "Data: %{x}<br>Alpha Rolling: %{y:.2%}<extra></extra>")))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(t=10, b=10, l=10, r=10),
                              yaxis_tickformat=".1%", hovermode="x unified")
            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.5)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown(t("#### 📊 Return Distribution (Tails & Skewness)", "#### 📊 Distribuzione dei Rendimenti (Asimmetria e Code)"))
        st.markdown(t("<small><b>How to read this:</b> Imagine putting all past months into 'buckets' based on their returns. A good fund should have fewer months in the extreme left buckets (crashes) and more months on the right side compared to the benchmark.</small>",
                      "<small><b>Come leggerlo:</b> Immagina di mettere tutti i mesi passati in dei 'secchi' in base a quanto hanno reso. Un buon fondo dovrebbe avere meno mesi nei secchi di estrema sinistra (i crolli) e più mesi nella parte destra rispetto al benchmark.</small>"),
                    unsafe_allow_html=True)

        hover_hist = t("Return: %{x:.1%}<br>Count: %{y}<extra></extra>", "Rendimento: %{x:.1%}<br>Mesi: %{y}<extra></extra>")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=f_monthly, name=t('Fund', 'Fondo'), marker_color='#34d399', opacity=0.7,
                                   nbinsx=50, hovertemplate=hover_hist))
        fig.add_trace(go.Histogram(x=b_monthly, name='Shadow Benchmark', marker_color='#9ca3af', opacity=0.5,
                                   nbinsx=50, hovertemplate=hover_hist))
        fig.update_layout(template="plotly_dark", height=350, barmode='overlay', margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_tickformat=".1%", legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig, use_container_width=True)
