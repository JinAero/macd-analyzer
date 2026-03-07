"""
MACD Multi-Timeframe Analyzer
교육용 AI 기술 지표 분석 도구 v0.2
------------------------------
- Jin의 DeepSeek API 키 서버에서 로드 (config_DEEPSEEK.json 또는 Streamlit Secrets)
- 세션당 3회 무료 AI 분석
- 3회 초과 시 구독 유도
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import json
import os
from pathlib import Path
from datetime import datetime, timezone

# ─────────────────────────────────────────
# DeepSeek API 키 로드
# 우선순위 1: Streamlit Cloud Secrets (배포 환경)
# 우선순위 2: config_DEEPSEEK.json (로컬 환경)
# ─────────────────────────────────────────
def verify_code(code: str) -> dict:
    """구독 코드 검증 — subscription_codes.json 대조"""
    code = code.strip()
    lang = st.session_state.get("lang", "en")
    if not code:
        return {"valid": False, "msg": "Please enter a code." if lang == "en" else "코드를 입력하세요."}

    # JSON 파일 위치 탐색
    paths = [
        Path(__file__).parent / "subscription_codes.json",
        Path("subscription_codes.json"),
    ]
    for p in paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                codes = data.get("codes", {})
                if code in codes:
                    entry = codes[code]
                    if not entry.get("active", False):
                        return {"valid": False, "msg": "Inactive code." if st.session_state.get("lang","en")=="en" else "비활성화된 코드입니다."}
                    # 만료일 체크
                    expires = entry.get("expires", "")
                    if expires:
                        from datetime import date
                        exp = date.fromisoformat(expires)
                        if date.today() > exp:
                            return {"valid": False, "msg": f"Code expired ({expires})." if st.session_state.get("lang","en")=="en" else f"만료된 코드입니다. (만료: {expires})"}
                    return {"valid": True, "plan": entry["plan"], "email": entry.get("email", "")}
                else:
                    return {"valid": False, "msg": "Code not found." if st.session_state.get("lang","en")=="en" else "존재하지 않는 코드입니다."}
            except Exception as e:
                return {"valid": False, "msg": f"Code file error: {e}" if st.session_state.get("lang","en")=="en" else f"코드 파일 오류: {e}"}

    # 파일 없으면 — 개발용 폴백 (배포 전 제거)
    if code.startswith("STD-"):
        return {"valid": True, "plan": "standard"}
    elif code.startswith("PRO-"):
        return {"valid": True, "plan": "pro"}

    return {"valid": False, "msg": T[st.session_state.get("lang","en")]["code_invalid"]}


def load_api_key() -> str:
    # 1) Streamlit Cloud Secrets
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass

    # 2) 환경변수
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key

    # 3) config_DEEPSEEK.json (llm_client.py와 동일한 방식)
    config_paths = [
        Path(__file__).parent / "config_DEEPSEEK.json",
        Path(os.path.expanduser("~")) / "config_DEEPSEEK.json",
        Path("config_DEEPSEEK.json"),
    ]
    for p in config_paths:
        if p.exists():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
                key = cfg.get("DEEPSEEK_API_KEY", "")
                if key:
                    return key
            except Exception:
                continue

    return ""


# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MACD Analyzer | AI-Powered Multi-TF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --accent-green: #00ff88;
    --accent-red: #ff4466;
    --accent-blue: #3b82f6;
    --accent-yellow: #fbbf24;
    --text-primary: #e2e8f0;
    --text-muted: #64748b;
    --border: #1e293b;
}

.stApp { background-color: var(--bg-primary); color: var(--text-primary); font-family: 'DM Sans', sans-serif; }

section[data-testid="stSidebar"] { background-color: #0d1321 !important; border-right: 1px solid var(--border); }

.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    letter-spacing: 0.05em; transition: all 0.2s; width: 100%;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(59,130,246,0.4); }

.metric-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
}
.badge-bull { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
.badge-bear { background: rgba(255,68,102,0.15); color: #ff4466; border: 1px solid rgba(255,68,102,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
.badge-neutral { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-family: 'Space Mono', monospace; }

.ai-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border: 1px solid #3730a3; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-top: 1rem;
    font-family: 'DM Sans', sans-serif; line-height: 1.7;
}
.ai-box h4 { color: #818cf8; font-family: 'Space Mono', monospace; font-size: 0.8rem; letter-spacing: 0.1em; margin-bottom: 0.8rem; }

.main-header {
    font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #3b82f6, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-header { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }

/* 카운터 뱃지 */
.counter-box {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 0.7rem 1rem;
    text-align: center; margin-bottom: 0.8rem;
}
.counter-num { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #00ff88; }
.counter-num.warn { color: #fbbf24; }
.counter-num.over { color: #ff4466; }

/* 업그레이드 배너 */
.upgrade-banner {
    background: linear-gradient(135deg, #1a0a2e 0%, #2d1b69 100%);
    border: 1px solid #7c3aed; border-radius: 12px;
    padding: 1.5rem; text-align: center; margin: 1rem 0;
}
.upgrade-banner h3 { color: #a78bfa; font-family: 'Space Mono', monospace; margin-bottom: 0.5rem; }

.plan-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem; text-align: center;
    transition: border-color 0.2s;
}
.plan-card:hover { border-color: #3b82f6; }

.disclaimer {
    background: rgba(255,68,102,0.05); border: 1px solid rgba(255,68,102,0.2);
    border-radius: 8px; padding: 0.8rem; font-size: 0.75rem; color: var(--text-muted); margin-top: 1rem;
}

hr { border-color: var(--border); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────
BINANCE_URL   = "https://api.binance.us/api/v3/klines"
INTERVALS     = ["1h", "30m", "15m", "5m", "3m", "1m"]
INTERVAL_LABELS = {
    "ko": {"1m": "1분봉", "3m": "3분봉", "5m": "5분봉",
           "15m": "15분봉", "30m": "30분봉", "1h": "1시간봉"},
    "en": {"1m": "1m", "3m": "3m", "5m": "5m",
           "15m": "15m", "30m": "30m", "1h": "1H"},
}

# ─────────────────────────────────────────
# i18n 텍스트
# ─────────────────────────────────────────
T = {
    "ko": {
        # 헤더
        "subtitle": "AI 기반 다중 타임프레임 기술 지표 분석 도구 · 교육 목적 전용",
        # 사이드바
        "settings": "⚙️ 설정",
        "symbol_label": "심볼",
        "plan_label": "🗝️ 플랜",
        "free_remaining": "무료 AI 분석 잔여",
        "session_reset": "세션 초기화 시 리셋",
        "free_plan_desc": "🆓 <b>무료 플랜</b><br>· 1H MACD 차트<br>· AI 분석 세션당 3회<br>· ETHUSDT 한정",
        "std_plan_desc":  "💎 <b>Standard</b><br>· MACD · 4개 TF (5m·15m·30m·1h)<br>· AI 분석: MACD 4TF 기반<br>· 심볼 5개 · 무제한 분석",
        "pro_plan_desc":  "👑 <b>Pro</b><br>· MACD + StochRSI + Volume · 6개 TF<br>· AI 분석: 전체 지표 종합 분석<br>· 심볼 무제한 · 우선 지원",
        "how_to_subscribe_title": "### 💳 구독 방법",
        "how_to_subscribe": "1. 아래 플랜 선택 후 Gumroad에서 결제<br>2. 결제 완료 메일의 구독 코드 확인<br>3. 위 입력란에 코드 입력 후 적용",
        "code_label": "### 🔑 구독 코드 입력",
        "code_placeholder": "MACD-XXXX-XXXX",
        "code_apply": "코드 적용",
        "code_invalid": "유효하지 않은 코드입니다.",
        "code_std_ok": "💎 Standard 플랜 활성화!",
        "code_pro_ok": "👑 Pro 플랜 활성화!",
        "footer": "📊 데이터: Binance REST API<br>🤖 AI: DeepSeek<br>⚠️ 교육 목적 전용",
        # 메인
        "analyze_btn": "🔍 지금 분석하기",
        "idle_msg": '"분석하기" 버튼을 눌러 현재 차트 상태를 확인하세요',
        "idle_sub": "6개 타임프레임 MACD · AI 교육 해설 · 실시간 바이낸스 데이터",
        "fetching": "Binance 실시간 데이터 수집 중...",
        "fetch_error": "데이터를 가져올 수 없습니다.",
        "tf_status": "#### 📋 타임프레임별 MACD 상태",
        "trend_bull": "▲ 상승", "trend_bear": "▼ 하락", "trend_neu": "— 중립",
        "golden": "🔔 골든크로스", "dead": "🔔 데드크로스",
        "chart_title": "#### 📉 MACD 차트",
        "ai_title": "#### 🤖 AI 교육 해설",
        "ai_box_title": "🧠 AI 기술 지표 분석",
        "ai_analyzing": "AI가 현재 차트를 분석 중...",
        "no_api_key": "⚠️ 서버에서 DeepSeek API 키를 찾을 수 없습니다. config_DEEPSEEK.json을 확인하세요.",
        "free_used_all": "⚡ 무료 분석 횟수를 모두 사용했습니다. 구독하면 무제한으로 사용할 수 있습니다.",
        "free_remaining_msg": "💡 무료 분석 {}회 남았습니다.",
        "guide_title": "📖 지표 해석 가이드 — 클릭해서 펼치기",
        "upgrade_title": "🔒 무료 AI 분석 {}회 소진",
        "upgrade_desc": "구독하면 더 많은 타임프레임과 무제한 AI 분석을 이용할 수 있습니다.",
        "upgrade_code_hint": "사이드바에서 구독 코드를 입력하거나 아래에서 구독을 시작하세요.",
        "per_month": "/월",
        "std_features": "MACD · 4 TFs (5m / 15m / 30m / 1H)<br>AI analysis: MACD 4TF 기반<br>심볼 5개 · 무제한 분석",
        "pro_features": "MACD + StochRSI + Volume · 6 TFs<br>AI analysis: 전체 지표 종합<br>심볼 무제한 · 우선 지원",
        "subscribe_btn": "구독 시작하기 →",
        "disclaimer": "⚠️ <b>면책 조항</b>: 본 도구는 기술 지표 교육 목적으로만 제공됩니다. AI 분석 결과는 투자 조언이 아니며, 실제 매매에 대한 책임은 사용자 본인에게 있습니다. 암호화폐 투자는 원금 손실의 위험이 있습니다.",
        "lang_toggle": "🇺🇸 English",
        "api_err": "⚠️ AI 분석 오류",
    },
    "en": {
        # 헤더
        "subtitle": "AI-Powered Multi-Timeframe Technical Education Tool · Educational Use Only",
        # 사이드바
        "settings": "⚙️ Settings",
        "symbol_label": "Symbol",
        "plan_label": "🗝️ Plan",
        "free_remaining": "Free AI Analysis Left",
        "session_reset": "Resets on session refresh",
        "free_plan_desc": "🆓 <b>Free Plan</b><br>· 1H MACD chart<br>· 3 AI analyses per session<br>· ETHUSDT only",
        "std_plan_desc":  "💎 <b>Standard</b><br>· MACD · 4 TFs (5m·15m·30m·1h)<br>· AI analysis: MACD across 4 TFs<br>· 5 symbols · Unlimited analyses",
        "pro_plan_desc":  "👑 <b>Pro</b><br>· MACD + StochRSI + Volume · 6 TFs<br>· AI analysis: All indicators combined<br>· All symbols · Priority support",
        "how_to_subscribe_title": "### 💳 How to Subscribe",
        "how_to_subscribe": "1. Choose a plan below and pay via Gumroad<br>2. Check your email for the subscription code<br>3. Enter the code above and click Apply",
        "code_label": "### 🔑 Enter Subscription Code",
        "code_placeholder": "MACD-XXXX-XXXX",
        "code_apply": "Apply Code",
        "code_invalid": "Invalid code.",
        "code_std_ok": "💎 Standard plan activated!",
        "code_pro_ok": "👑 Pro plan activated!",
        "footer": "📊 Data: Binance REST API<br>🤖 AI: DeepSeek<br>⚠️ For educational use only",
        # 메인
        "analyze_btn": "🔍 Analyze Now",
        "idle_msg": 'Press "Analyze Now" to load current chart data',
        "idle_sub": "6 Timeframes · AI Education Analysis · Live Binance Data",
        "fetching": "Fetching live data from Binance...",
        "fetch_error": "Failed to fetch data.",
        "tf_status": "#### 📋 MACD Status by Timeframe",
        "trend_bull": "▲ Bullish", "trend_bear": "▼ Bearish", "trend_neu": "— Neutral",
        "golden": "🔔 Golden Cross", "dead": "🔔 Death Cross",
        "chart_title": "#### 📉 MACD Charts",
        "ai_title": "#### 🤖 AI Analysis",
        "ai_box_title": "🧠 AI Technical Analysis",
        "ai_analyzing": "AI is analyzing current charts...",
        "no_api_key": "⚠️ DeepSeek API key not found on server. Check config_DEEPSEEK.json.",
        "free_used_all": "⚡ Free analyses used up. Subscribe for unlimited access.",
        "free_remaining_msg": "💡 {} free analyses remaining.",
        "guide_title": "📖 Indicator Guide — Click to expand",
        "upgrade_title": "🔒 {} Free Analyses Used",
        "upgrade_desc": "Subscribe to unlock more timeframes and unlimited AI analysis.",
        "upgrade_code_hint": "Enter your subscription code in the sidebar or subscribe below.",
        "per_month": "/mo",
        "std_features": "MACD · 4 TFs (5m / 15m / 30m / 1H)<br>AI analysis: MACD across 4 TFs<br>5 symbols · Unlimited analyses",
        "pro_features": "MACD + StochRSI + Volume · 6 TFs<br>AI analysis: All indicators combined<br>All symbols · Priority support",
        "subscribe_btn": "Subscribe →",
        "disclaimer": "⚠️ <b>Disclaimer</b>: This tool is for educational purposes only. AI analysis results are not investment advice. You are solely responsible for any trading decisions. Cryptocurrency trading carries significant risk of loss.",
        "lang_toggle": "🇰🇷 한국어",
        "api_err": "⚠️ AI analysis error",
    },
}
FREE_LIMIT    = 3    # 무료 세션당 AI 분석 횟수
FREE_TF       = ["1h"]
STANDARD_TF   = ["1h", "30m", "15m", "5m"]
PRO_TF        = INTERVALS

# Gumroad 구독 링크 (나중에 실제 URL로 교체)
GUMROAD_STANDARD = "https://gumroad.com/l/your-standard-link"
GUMROAD_PRO      = "https://gumroad.com/l/your-pro-link"


# ─────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────
if "ai_count" not in st.session_state:
    st.session_state.ai_count = 0
if "plan" not in st.session_state:
    st.session_state.plan = "free"
if "lang" not in st.session_state:
    st.session_state.lang = "en"    # 기본 영어


# ─────────────────────────────────────────
# 데이터 수집 & 지표 계산 (rawDataarranger_V1.py 동일 로직)
# ─────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_URL, params=params, timeout=10)
    resp.raise_for_status()
    records = [
        {"t": int(r[0]//1000), "o": float(r[1]), "h": float(r[2]),
         "l": float(r[3]), "c": float(r[4]), "v": float(r[5])}
        for r in resp.json()
    ]
    return pd.DataFrame(records)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["c"]
    df["e9"]  = close.ewm(span=9,  adjust=False).mean()
    df["e12"] = close.ewm(span=12, adjust=False).mean()
    df["e26"] = close.ewm(span=26, adjust=False).mean()
    df["m"]   = df["e12"] - df["e26"]
    df["ms"]  = df["m"].ewm(span=9, adjust=False).mean()
    df["mh"]  = df["m"] - df["ms"]

    delta = close.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + roll_up / (roll_dn + 1e-9)))

    rsi = df["rsi"]
    sr_min = rsi.rolling(14).min()
    sr_max = rsi.rolling(14).max()
    df["sr"]  = (rsi - sr_min) / (sr_max - sr_min + 1e-9)
    df["srk"] = df["sr"].rolling(3).mean()
    df["srd"] = df["srk"].rolling(3).mean()

    ha_c = (df["o"] + df["h"] + df["l"] + df["c"]) / 4
    ha_o = ha_c.copy()
    ha_o.iloc[0] = (df["o"].iloc[0] + df["c"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_o.iloc[i] = (ha_o.iloc[i-1] + ha_c.iloc[i-1]) / 2
    df["ha_o"] = ha_o
    df["ha_c"] = ha_c

    prev = (df["e9"] - df["e26"]).shift(1)
    curr = (df["e9"] - df["e26"])
    df["bull_cross"] = ((prev <= 0) & (curr > 0)).astype(int)
    df["bear_cross"] = ((prev >= 0) & (curr < 0)).astype(int)

    return df.dropna().tail(60)


def macd_status(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    m, ms, mh = last["m"], last["ms"], last["mh"]
    if mh > 0 and m > ms:
        trend = "BULLISH"
    elif mh < 0 and m < ms:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    return {
        "macd": round(m, 5), "signal": round(ms, 5),
        "hist": round(mh, 5), "srk": round(last["srk"], 3),
        "srd": round(last.get("srd", 0), 3),
        "vol": round(last.get("v", 0), 0),
        "ha_bull": bool(last.get("ha_c", last["c"]) >= last.get("ha_o", last["o"])),
        "trend": trend,
        "bull_cross": bool(last.get("bull_cross", 0)),
        "bear_cross": bool(last.get("bear_cross", 0)),
        "close": round(last["c"], 2),
    }


def plot_macd(df: pd.DataFrame, interval: str, symbol: str = "ETHUSDT", labels: dict = None, plan: str = "free") -> go.Figure:
    """
    독립 차트 — 각 TF별 개별 figure
    - 캔들 위에 투명 Scatter 덮어씌워 spike line이 캔들+MACD 패널 모두 관통
    - hovermode="x unified" 로 두 패널 동시 수직 점선
    """
    if plan == "pro":
        n_rows = 4
        row_heights = [0.42, 0.22, 0.18, 0.18]
        subplot_titles_list = ["", "", "StochRSI", "Volume"]
    else:
        n_rows = 2
        row_heights = [0.62, 0.38]
        subplot_titles_list = ["", ""]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.0,
        subplot_titles=subplot_titles_list,
    )
    ts = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("Asia/Seoul")

    # ── 캔들
    fig.add_trace(go.Candlestick(
        x=ts, open=df["o"], high=df["h"], low=df["l"], close=df["c"],
        name="Price",
        increasing_fillcolor="#00ff88", increasing_line_color="#00ff88",
        decreasing_fillcolor="#ff4466", decreasing_line_color="#ff4466",
        hoverinfo="skip",               # 캔들 자체 hover 끄기
    ), row=1, col=1)

    # ── 캔들 위에 투명 Scatter 덮기 → spike line 트리거용
    fig.add_trace(go.Scatter(
        x=ts, y=df["c"],
        mode="markers",
        marker=dict(size=8, opacity=0),  # 완전 투명
        name="Close",
        showlegend=False,
        hovertemplate=(
            "<b>%{x|%m/%d %H:%M}</b><br>"
            "O: %{customdata[0]:.2f}  H: %{customdata[1]:.2f}<br>"
            "L: %{customdata[2]:.2f}  C: %{y:.2f}<extra></extra>"
        ),
        customdata=list(zip(df["o"], df["h"], df["l"])),
    ), row=1, col=1)

    # ── EMA
    fig.add_trace(go.Scatter(x=ts, y=df["e9"],  name="EMA9",
        line=dict(color="#fbbf24", width=1.2), opacity=0.8,
        hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df["e26"], name="EMA26",
        line=dict(color="#818cf8", width=1.2), opacity=0.8,
        hoverinfo="skip"), row=1, col=1)

    # ── MACD Histogram
    colors = ["#00ff88" if v >= 0 else "#ff4466" for v in df["mh"]]
    fig.add_trace(go.Bar(
        x=ts, y=df["mh"], name="Hist",
        marker_color=colors, opacity=0.7,
        hovertemplate="Hist: %{y:.5f}<extra></extra>",
    ), row=2, col=1)

    # ── MACD & Signal
    fig.add_trace(go.Scatter(
        x=ts, y=df["m"], name="MACD",
        line=dict(color="#3b82f6", width=1.5),
        hovertemplate="MACD: %{y:.5f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ts, y=df["ms"], name="Signal",
        line=dict(color="#f97316", width=1.2, dash="dot"),
        hovertemplate="Signal: %{y:.5f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#334155", line_width=1, row=2, col=1)

    # ── Pro 전용: StochRSI + Volume 패널 (row 3, 4)
    if plan == "pro":
        # StochRSI
        fig.add_trace(go.Scatter(
            x=ts, y=df["srk"], name="StochRSI K",
            line=dict(color="#a78bfa", width=1.4),
            hovertemplate="StochRSI K: %{y:.3f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ts, y=df["srd"], name="StochRSI D",
            line=dict(color="#f97316", width=1.1, dash="dot"),
            hovertemplate="StochRSI D: %{y:.3f}<extra></extra>",
        ), row=3, col=1)
        fig.add_hline(y=0.8, line_dash="dot", line_color="#ff4466", line_width=0.8, row=3, col=1)
        fig.add_hline(y=0.2, line_dash="dot", line_color="#00ff88", line_width=0.8, row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="#334155", line_width=0.8, row=3, col=1)

        # Volume
        vol_colors = ["#00ff88" if c >= o else "#ff4466" for c, o in zip(df["c"], df["o"])]
        fig.add_trace(go.Bar(
            x=ts, y=df["v"], name="Volume",
            marker_color=vol_colors, opacity=0.6,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=4, col=1)

    # ── spike line 설정 (캔들+MACD 두 패널 모두 관통)
    spike_cfg = dict(
        showgrid=False,
        rangeslider=dict(visible=False),
        color="#475569",
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        spikecolor="#64748b",
        spikethickness=1,
        spikedash="dot",
    )

    fig.update_layout(
        title=dict(
            text=f"{symbol} — {(labels or INTERVAL_LABELS['en'])[interval]}",
            font=dict(family="Space Mono", size=12, color="#64748b"), x=0.01,
        ),
        paper_bgcolor="#0a0e17", plot_bgcolor="#111827", height=400,
        margin=dict(l=10, r=10, t=35, b=0),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1e293b", bordercolor="#334155",
            font=dict(color="#e2e8f0", size=11),
        ),
        legend=dict(
            orientation="h", y=1.04, x=0,
            font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=spike_cfg,
        xaxis2={**spike_cfg},
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#475569"),
        yaxis2=dict(showgrid=True, gridcolor="#1e293b", color="#475569"),
        **({"xaxis3": spike_cfg, "xaxis4": spike_cfg,
            "yaxis3": dict(showgrid=True, gridcolor="#1e293b", color="#475569", range=[0, 1]),
            "yaxis4": dict(showgrid=True, gridcolor="#1e293b", color="#475569")}
           if plan == "pro" else {}),
    )

    # Pro subplot 타이틀 스타일
    if plan == "pro":
        for ann in fig.layout.annotations:
            ann.font.color  = "#64748b"
            ann.font.size   = 10
            ann.font.family = "Space Mono"

    return fig


# ─────────────────────────────────────────
# AI 교육 분석 (Jin의 DeepSeek API 키 사용)
# ─────────────────────────────────────────
SYSTEM_PROMPT_EDU = """너는 10년 이상 경력의 퀀트 트레이더이자 기술적 분석 전문가다.
다중 타임프레임 MACD 스냅샷 데이터를 보고 현재 시장 구조를 분석하라.

출력 형식 (반드시 이 구조를 따를 것):

**[현재 구조 요약]**
— 2~3줄로 현재 TF 전반의 MACD 배열 상태를 압축 서술. 숫자 근거 포함.

**[TF별 핵심 포인트]**
— 각 TF를 1줄씩. 불필요한 설명 없이 수치와 의미만.
— 형식: `[1H] MACD 15.82 / Hist -1.33 → 하락 모멘텀 지속, Signal 이하 유지`

**[TF 간 정합성]**
— 단기(1m~5m)와 중기(15m~1h) 방향이 일치하는지 불일치하는지.
— 불일치 시 어느 쪽이 우세한지 판단 근거 제시.

**[주목할 시그널]**
— 골든/데드크로스, 히스토그램 반전, StochRSI 극값 등 유의미한 신호만 기술.
— 없으면 "특이 신호 없음" 한 줄.

규칙:
- 직접적 매매 지시 ("매수하라" 등) 절대 금지
- 쉬운 해설, 일반 설명, 인사말 불필요 — 분석만
- 한국어, 간결하고 밀도 있게
- 마지막 줄: ⚠️ 본 분석은 교육 목적이며 투자 판단의 근거로 사용할 수 없습니다."""

SYSTEM_PROMPT_EDU_EN = """You are a quantitative trader and technical analysis expert with 10+ years of experience.
Analyze the provided multi-timeframe MACD snapshot and describe the current market structure.

Required output format:

**[Structure Summary]**
— 2-3 lines summarizing the overall MACD alignment across timeframes. Include specific numbers.

**[Key Points by TF]**
— One line per TF. Numbers and meaning only, no generic explanation.
— Format: `[1H] MACD 15.82 / Hist -1.33 → downward momentum sustained, below Signal`

**[TF Alignment]**
— Are short-term (1m-5m) and mid-term (15m-1h) aligned or diverging?
— If diverging, state which side is dominant and why.

**[Notable Signals]**
— Golden/death crosses, histogram reversals, StochRSI extremes only.
— If none: "No notable signals."

Rules:
- No direct trading instructions ("buy", "sell", etc.)
- No introductions, general explanations, or greetings — analysis only
- English, concise and dense
- Last line: ⚠️ This analysis is for educational purposes only and should not be used as investment advice."""

SYSTEM_PROMPT_EDU_EN_PRO = """You are a quantitative trader and technical analysis expert with 10+ years of experience.
Analyze ALL provided indicators across all timeframes: MACD, StochRSI (K & D), Volume, and Heikin-Ashi candle direction.

Required output format:

**[Structure Summary]**
— 2-3 lines. Synthesize MACD alignment, StochRSI levels, Volume trend, and HA candle direction together.
— Include specific numbers.

**[Key Points by TF]**
— One line per TF combining all available indicators.
— Format: `[1H] MACD Hist -1.33 (bearish) | StochRSI 0.245 (near oversold) | Vol declining | HA bearish`

**[TF Alignment]**
— Cross-indicator alignment: Do MACD, StochRSI, Volume, and HA all point the same direction?
— Note any divergence between indicators (e.g. MACD bearish but StochRSI oversold bounce possible).

**[Notable Signals]**
— MACD crosses, StochRSI K/D cross, Volume spike vs contraction, HA color change.
— If none: "No notable signals."

Rules:
- No direct trading instructions ("buy", "sell", etc.)
- No introductions or greetings — analysis only
- English, concise and dense
- Last line: ⚠️ This analysis is for educational purposes only and should not be used as investment advice."""

SYSTEM_PROMPT_EDU_KO_PRO = """너는 10년 이상 경력의 퀀트 트레이더이자 기술적 분석 전문가다.
제공된 모든 지표를 종합 분석하라: MACD, StochRSI (K & D), Volume, Heikin-Ashi 캔들 방향.

출력 형식:

**[현재 구조 요약]**
— 2~3줄. MACD 배열, StochRSI 레벨, Volume 추세, HA 캔들 방향을 종합 서술. 수치 포함.

**[TF별 핵심 포인트]**
— 각 TF 1줄, 전체 지표 포함.
— 형식: `[1H] MACD Hist -1.33 (하락) | StochRSI 0.245 (과매도 근접) | Vol 감소 | HA 음봉`

**[지표 간 정합성]**
— MACD, StochRSI, Volume, HA가 같은 방향을 가리키는지.
— 지표 간 다이버전스 명시 (예: MACD 하락인데 StochRSI 과매도 반등 가능).

**[주목할 시그널]**
— MACD 크로스, StochRSI K/D 크로스, Volume 급등/급감, HA 색상 전환.
— 없으면 "특이 신호 없음".

규칙:
- 직접적 매매 지시 금지
- 인사말, 일반 설명 불필요 — 분석만
- 한국어, 간결하고 밀도 있게
- 마지막 줄: ⚠️ 본 분석은 교육 목적이며 투자 판단의 근거로 사용할 수 없습니다."""


def build_context(status_dict: dict, plan: str = "free") -> str:
    """plan에 따라 AI에 전달하는 지표 범위 결정"""
    lines = ["[Multi-Timeframe Snapshot]"]
    for itv, s in status_dict.items():
        line = (
            f"[{itv}] MACD={s['macd']:.5f} | Signal={s['signal']:.5f} | "
            f"Hist={s['hist']:.5f} | StochRSI_K={s['srk']:.3f} | "
            f"Close={s['close']} | Trend={s['trend']}"
            + (" ★BullCross" if s["bull_cross"] else "")
            + (" ★BearCross" if s["bear_cross"] else "")
        )
        if plan == "pro":
            line += (
                f" | StochRSI_D={s.get('srd', 0):.3f}"
                f" | Vol={s.get('vol', 0):,.0f}"
                f" | HA={'Bullish' if s.get('ha_bull') else 'Bearish'}"
            )
        lines.append(line)
    return "\n".join(lines)


def md_to_html(text: str) -> str:
    """마크다운 → HTML 변환 (AI 분석 결과 렌더링용)"""
    import re
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<code style="background:#1e293b;padding:2px 6px;border-radius:4px;font-family:Space Mono,monospace;">\1</code>', text)
    text = text.replace("\n", "<br>")
    return text


def ask_ai(context: str, api_key: str, lang: str = "en", plan: str = "free") -> str:
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        if plan == "pro":
            prompt = SYSTEM_PROMPT_EDU_KO_PRO if lang == "ko" else SYSTEM_PROMPT_EDU_EN_PRO
        else:
            prompt = SYSTEM_PROMPT_EDU if lang == "ko" else SYSTEM_PROMPT_EDU_EN
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": context},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"{T[lang]['api_err']}: {e}"


# ─────────────────────────────────────────
# 메인 UI
# ─────────────────────────────────────────
def main():
    api_key = load_api_key()
    lang = st.session_state.lang
    t = T[lang]
    labels = INTERVAL_LABELS[lang]

    # ── 헤더
    st.markdown('<div class="main-header">📊 MACD Multi-TF Analyzer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{t["subtitle"]}</div>', unsafe_allow_html=True)

    # ── 사이드바
    with st.sidebar:
        # 언어 토글 (최상단)
        if st.button(t["lang_toggle"], use_container_width=False):
            st.session_state.lang = "ko" if lang == "en" else "en"
            st.rerun()

        st.markdown("---")
        st.markdown(f"### {t['settings']}")
        plan_now = st.session_state.plan
        if plan_now == "free":
            symbol = "ETHUSDT"
            st.markdown(
                '<div style="font-size:0.78rem;color:#475569;">Symbol: ETHUSDT<br>(Free 플랜 한정 / Free plan only)</div>',
                unsafe_allow_html=True,
            )
        elif plan_now == "standard":
            symbol = st.selectbox(
                t["symbol_label"],
                ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
            )
        else:  # pro
            symbol = st.selectbox(
                t["symbol_label"],
                ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                 "ADAUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"],
            )

        st.markdown("---")
        st.markdown(f"### {t['plan_label']}")

        plan = st.session_state.plan

        # 플랜별 안내
        if plan == "free":
            remaining = max(0, FREE_LIMIT - st.session_state.ai_count)
            num_color = "counter-num" if remaining > 1 else ("counter-num warn" if remaining == 1 else "counter-num over")
            st.markdown(f"""
            <div class="counter-box">
              <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.2rem;">{t["free_remaining"]}</div>
              <div class="{num_color}">{remaining} / {FREE_LIMIT}</div>
              <div style="font-size:0.72rem;color:#475569;margin-top:0.3rem;">{t["session_reset"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.78rem;color:#475569;line-height:1.6;">{t["free_plan_desc"]}</div>',
                        unsafe_allow_html=True)

        elif plan == "standard":
            st.markdown(f'<div style="font-size:0.78rem;color:#3b82f6;line-height:1.6;">{t["std_plan_desc"]}</div>',
                        unsafe_allow_html=True)

        elif plan == "pro":
            st.markdown(f'<div style="font-size:0.78rem;color:#fbbf24;line-height:1.6;">{t["pro_plan_desc"]}</div>',
                        unsafe_allow_html=True)

        # ── 구독 플랜 카드
        st.markdown("---")
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:10px;padding:0.9rem;margin-bottom:0.6rem;">
          <div style="color:#64748b;font-size:0.72rem;font-family:Space Mono;margin-bottom:0.5rem;">🆓 FREE</div>
          <div style="font-size:0.76rem;color:#475569;line-height:1.7;">{t["free_plan_desc"]}</div>
        </div>
        <div style="background:#0f172a;border:1px solid #3b82f6;border-radius:10px;padding:0.9rem;margin-bottom:0.6rem;">
          <div style="color:#3b82f6;font-size:0.72rem;font-family:Space Mono;margin-bottom:0.5rem;">💎 STANDARD — $9{t["per_month"]}</div>
          <div style="font-size:0.76rem;color:#94a3b8;line-height:1.7;">{t["std_plan_desc"]}</div>
        </div>
        <div style="background:#0f172a;border:1px solid #fbbf24;border-radius:10px;padding:0.9rem;margin-bottom:0.6rem;">
          <div style="color:#fbbf24;font-size:0.72rem;font-family:Space Mono;margin-bottom:0.5rem;">👑 PRO — $19{t["per_month"]}</div>
          <div style="font-size:0.76rem;color:#94a3b8;line-height:1.7;">{t["pro_plan_desc"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 구독 방법
        st.markdown("---")
        st.markdown(t["how_to_subscribe_title"])
        st.markdown(
            f'<div style="font-size:0.78rem;color:#64748b;line-height:1.9;">{t["how_to_subscribe"]}</div>',
            unsafe_allow_html=True,
        )
        col_std, col_pro = st.columns(2)
        with col_std:
            st.markdown(
                f'<a href="{GUMROAD_STANDARD}" target="_blank">'
                f'<div style="background:#1d4ed8;color:white;text-align:center;padding:0.45rem;'
                f'border-radius:8px;font-size:0.8rem;cursor:pointer;">💎 Standard</div></a>',
                unsafe_allow_html=True,
            )
        with col_pro:
            st.markdown(
                f'<a href="{GUMROAD_PRO}" target="_blank">'
                f'<div style="background:#d97706;color:white;text-align:center;padding:0.45rem;'
                f'border-radius:8px;font-size:0.8rem;cursor:pointer;">👑 Pro</div></a>',
                unsafe_allow_html=True,
            )

        # ── 구독 코드 입력
        st.markdown("---")
        st.markdown(t["code_label"])
        code = st.text_input("Code", placeholder=t["code_placeholder"], type="password", label_visibility="collapsed")
        if st.button(t["code_apply"]):
            result = verify_code(code)
            if result["valid"]:
                st.session_state.plan = result["plan"]
                st.success(t["code_std_ok"] if result["plan"] == "standard" else t["code_pro_ok"])
                st.rerun()
            else:
                st.error(result["msg"])

        st.markdown("---")
        st.markdown(f'<div style="font-size:0.72rem;color:#334155;line-height:1.7;">{t["footer"]}</div>',
                    unsafe_allow_html=True)

    # ── 플랜별 타임프레임
    if plan == "free":
        available_tf = FREE_TF
    elif plan == "standard":
        available_tf = STANDARD_TF
    else:
        available_tf = PRO_TF

    # ── 분석 버튼
    col_btn, col_time = st.columns([1, 3])
    with col_btn:
        run = st.button(t["analyze_btn"], use_container_width=True)
    with col_time:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.markdown(f"<div style='padding-top:0.6rem;color:#475569;font-size:0.85rem;font-family:Space Mono'>⏱ {now_utc}</div>",
                    unsafe_allow_html=True)

    if not run:
        st.markdown(f"""
        <div style="margin-top:3rem;text-align:center;color:#1e293b;">
          <div style="font-size:3rem;margin-bottom:1rem;">📈</div>
          <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#3b82f6;">
            {t["idle_msg"]}
          </div>
          <div style="font-size:0.8rem;margin-top:0.5rem;color:#334155;">
            {t["idle_sub"]}
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── 데이터 수집
    with st.spinner(t["fetching"]):
        data_dict = {}
        for itv in available_tf:
            try:
                df = fetch_ohlcv(symbol, itv)
                df = add_indicators(df)
                data_dict[itv] = {"df": df, "status": macd_status(df)}
            except Exception as e:
                st.error(f"{itv} 데이터 오류: {e}")

    if not data_dict:
        st.error(t["fetch_error"])
        return

    # ── TF 상태 카드
    st.markdown(t["tf_status"])
    cols = st.columns(len(data_dict))
    for i, (itv, d) in enumerate(data_dict.items()):
        s = d["status"]
        badge = "bull" if s["trend"] == "BULLISH" else ("bear" if s["trend"] == "BEARISH" else "neutral")
        label = t["trend_bull"] if s["trend"] == "BULLISH" else (t["trend_bear"] if s["trend"] == "BEARISH" else t["trend_neu"])
        cross = t["golden"] if s["bull_cross"] else (t["dead"] if s["bear_cross"] else "")
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#64748b;">{labels[itv]}</div>
              <div style="margin:0.3rem 0;"><span class="badge-{badge}">{label}</span></div>
              <div style="font-size:0.78rem;color:#94a3b8;">MACD: {s['macd']:.5f}</div>
              <div style="font-size:0.78rem;color:#94a3b8;">Hist: {s['hist']:.5f}</div>
              <div style="font-size:0.78rem;color:#94a3b8;">StoRSI: {s['srk']:.3f}</div>
              {f'<div style="font-size:0.75rem;color:#fbbf24;margin-top:0.3rem;">{cross}</div>' if cross else ''}
            </div>
            """, unsafe_allow_html=True)

    # ── 차트 (TF별 독립 figure — 각 차트 내 spike line)
    st.markdown("---")
    st.markdown(t["chart_title"])
    for itv, d in data_dict.items():
        st.plotly_chart(plot_macd(d["df"], itv, symbol, labels, plan), use_container_width=True)

    # ── AI 분석 섹션
    st.markdown("---")
    st.markdown(t["ai_title"])

    can_use_ai = (plan != "free") or (st.session_state.ai_count < FREE_LIMIT)

    if not api_key:
        st.warning(t["no_api_key"])

    elif can_use_ai:
        with st.spinner(t["ai_analyzing"]):
            context = build_context({itv: d["status"] for itv, d in data_dict.items()}, plan)
            reply = ask_ai(context, api_key, lang, plan)

        # 무료 플랜이면 카운터 증가
        if plan == "free":
            st.session_state.ai_count += 1
            remaining = FREE_LIMIT - st.session_state.ai_count

        st.markdown(f"""
        <div class="ai-box">
          <h4>{t["ai_box_title"]}</h4>
          <div style="color:#cbd5e1;">{md_to_html(reply)}</div>
        </div>
        """, unsafe_allow_html=True)

        # 무료 플랜 — 잔여 횟수 경고
        if plan == "free":
            if remaining == 0:
                st.warning(t["free_used_all"])
            elif remaining == 1:
                st.info(t["free_remaining_msg"].format(remaining))

        # ── 지표 상세 설명 (접이식)
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        with st.expander(t["guide_title"]):
            if lang == "ko":
                st.markdown("""
<div style="color:#94a3b8; line-height:1.9; font-size:0.88rem;">

### MACD (Moving Average Convergence Divergence)

**구성**: EMA12 − EMA26 = MACD 라인 / MACD의 EMA9 = Signal 라인 / 두 값의 차 = Histogram

| 상태 | 의미 |
|------|------|
| MACD > Signal, Hist 양수 증가 | 상승 모멘텀 강화 |
| MACD < Signal, Hist 음수 확대 | 하락 모멘텀 강화 |
| Hist가 음수 → 양수 전환 | 골든크로스 (강세 전환 신호) |
| Hist가 양수 → 음수 전환 | 데드크로스 (약세 전환 신호) |
| Hist 절대값 감소 | 현재 추세의 모멘텀 약화 |

**주의**: MACD는 후행 지표입니다. 단독 사용보다 다른 지표와 조합이 효과적입니다.

---

### StochRSI (Stochastic RSI)

RSI 값에 Stochastic 공식을 적용한 지표. 0~1 범위로 표시됩니다.

| 범위 | 해석 |
|------|------|
| 0.8 이상 | 과매수 구간 — 조정 가능성 |
| 0.2 이하 | 과매도 구간 — 반등 가능성 |
| K > D (상향) | 단기 상승 압력 |
| K < D (하향) | 단기 하락 압력 |

**주의**: 추세가 강할 경우 과매수/과매도 상태가 오래 유지될 수 있습니다.

---

### EMA Cross (EMA9 / EMA26)

| 상태 | 의미 |
|------|------|
| EMA9 > EMA26 (골든크로스) | 단기 상승 추세 전환 |
| EMA9 < EMA26 (데드크로스) | 단기 하락 추세 전환 |
| 두 EMA 간격 확대 | 추세 강화 |
| 두 EMA 수렴 | 추세 약화 또는 횡보 |

---

### 다중 타임프레임 분석 원칙

```
1H / 30M  → 중기 방향 판단 (큰 그림)
15M / 5M  → 추세 확인 및 타이밍
3M / 1M   → 세부 진입 시점 (노이즈 많음)
```

**상위 TF와 하위 TF가 일치**할 때 신호의 신뢰도가 높습니다.

</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="color:#94a3b8; line-height:1.9; font-size:0.88rem;">

### MACD (Moving Average Convergence Divergence)

**Components**: EMA12 − EMA26 = MACD line / EMA9 of MACD = Signal line / Difference = Histogram

| State | Meaning |
|-------|---------|
| MACD > Signal, Hist positive & growing | Bullish momentum strengthening |
| MACD < Signal, Hist negative & expanding | Bearish momentum strengthening |
| Hist negative → positive | Golden Cross (bullish reversal signal) |
| Hist positive → negative | Death Cross (bearish reversal signal) |
| Hist absolute value shrinking | Current trend momentum weakening |

**Note**: MACD is a lagging indicator. More effective when combined with other indicators.

---

### StochRSI (Stochastic RSI)

Stochastic formula applied to RSI values. Range: 0–1.

| Range | Interpretation |
|-------|---------------|
| Above 0.8 | Overbought — potential correction |
| Below 0.2 | Oversold — potential bounce |
| K > D (rising) | Short-term upward pressure |
| K < D (falling) | Short-term downward pressure |

**Note**: In strong trends, overbought/oversold conditions can persist for extended periods.

---

### EMA Cross (EMA9 / EMA26)

| State | Meaning |
|-------|---------|
| EMA9 > EMA26 (Golden Cross) | Short-term uptrend shift |
| EMA9 < EMA26 (Death Cross) | Short-term downtrend shift |
| EMA gap widening | Trend strengthening |
| EMA convergence | Trend weakening or consolidation |

---

### Multi-Timeframe Analysis Principles

```
1H / 30M  → Mid-term direction (big picture)
15M / 5M  → Trend confirmation & timing
3M / 1M   → Granular entry points (more noise)
```

Signals are more reliable when **higher and lower TFs align**.  
When **diverging**, prioritize the higher timeframe direction.

</div>
                """, unsafe_allow_html=True)

    else:
        # 3회 초과 — 업그레이드 유도
        st.markdown(f"""
        <div class="upgrade-banner">
          <h3>{t["upgrade_title"].format(FREE_LIMIT)}</h3>
          <p style="color:#a78bfa;margin-bottom:1rem;">{t["upgrade_desc"]}</p>
          <p style="color:#64748b;font-size:0.8rem;">{t["upgrade_code_hint"]}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="plan-card" style="border-color:#3b82f6;">
              <div style="color:#3b82f6;font-family:'Space Mono',monospace;font-size:0.9rem;">💎 Standard</div>
              <div style="font-size:1.8rem;font-weight:700;margin:0.5rem 0;">$9<span style="font-size:0.9rem;color:#64748b;">{t["per_month"]}</span></div>
              <div style="font-size:0.8rem;color:#94a3b8;line-height:1.7;">{t["std_features"]}</div>
              <a href="{GUMROAD_STANDARD}" target="_blank">
                <div style="margin-top:1rem;background:#3b82f6;color:white;padding:0.5rem;border-radius:8px;font-size:0.85rem;cursor:pointer;">
                  {t["subscribe_btn"]}
                </div>
              </a>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="plan-card" style="border-color:#fbbf24;">
              <div style="color:#fbbf24;font-family:'Space Mono',monospace;font-size:0.9rem;">👑 Pro</div>
              <div style="font-size:1.8rem;font-weight:700;margin:0.5rem 0;">$19<span style="font-size:0.9rem;color:#64748b;">{t["per_month"]}</span></div>
              <div style="font-size:0.8rem;color:#94a3b8;line-height:1.7;">{t["pro_features"]}</div>
              <a href="{GUMROAD_PRO}" target="_blank">
                <div style="margin-top:1rem;background:#d97706;color:white;padding:0.5rem;border-radius:8px;font-size:0.85rem;cursor:pointer;">
                  {t["subscribe_btn"]}
                </div>
              </a>
            </div>
            """, unsafe_allow_html=True)

    # ── 면책 조항
    st.markdown(f'<div class="disclaimer">{t["disclaimer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
