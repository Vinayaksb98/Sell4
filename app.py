import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime

st.set_page_config(page_title='NIFTY 500 Sell Scanner', layout='wide')
st.title('🔻 NIFTY 500 — Core + Extraordinary Sell Scanner')
st.caption('Selling side • Breakdown Confirmed / Near Breakdown • Live CMP')

# ---------- Universe ----------
@st.cache_data(ttl=86400, show_spinner=False)
def get_nifty500():
    url='https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
    headers={'User-Agent':'Mozilla/5.0','Accept':'text/csv,*/*','Referer':'https://www.niftyindices.com/'}
    try:
        r=requests.get(url,headers=headers,timeout=25); r.raise_for_status()
        d=pd.read_csv(StringIO(r.text)); col='Symbol' if 'Symbol' in d.columns else d.columns[0]
        s=d[col].astype(str).str.strip().str.upper().replace('',np.nan).dropna().unique().tolist()
        return [x+'.NS' for x in s]
    except Exception as e:
        st.warning('NIFTY 500 list could not be refreshed; using the built-in fallback list.')
        return ['RELIANCE.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','TCS.NS','BHARTIARTL.NS','ITC.NS','LT.NS','SBIN.NS','AXISBANK.NS']

# ---------- Indicators ----------
def indicators(df):
    d=df.copy().dropna(subset=['Open','High','Low','Close'])
    ll=d['Low'].rolling(14,min_periods=14).min(); hh=d['High'].rolling(14,min_periods=14).max()
    rawk=100*(d['Close']-ll)/(hh-ll).replace(0,np.nan)
    d['STOCH']=rawk.rolling(3,min_periods=3).mean()
    ema8=d['Close'].ewm(span=8,adjust=False,min_periods=8).mean(); ema17=d['Close'].ewm(span=17,adjust=False,min_periods=17).mean()
    d['MACD']=ema8-ema17; d['MACD_SIGNAL']=d['MACD'].ewm(span=9,adjust=False,min_periods=9).mean()
    d['EMA20']=d['Close'].ewm(span=20,adjust=False).mean(); d['EMA50']=d['Close'].ewm(span=50,adjust=False).mean(); d['EMA150']=d['Close'].ewm(span=150,adjust=False).mean(); d['EMA200']=d['Close'].ewm(span=200,adjust=False).mean()
    tr=pd.concat([(d['High']-d['Low']), (d['High']-d['Close'].shift()).abs(), (d['Low']-d['Close'].shift()).abs()],axis=1).max(axis=1)
    d['ATR']=tr.rolling(14,min_periods=14).mean()
    plus=(d['High'].diff()).clip(lower=0); minus=(-d['Low'].diff()).clip(lower=0)
    atr14=d['ATR'].replace(0,np.nan)
    d['PDI']=100*plus.ewm(alpha=1/14,adjust=False).mean()/atr14; d['MDI']=100*minus.ewm(alpha=1/14,adjust=False).mean()/atr14
    dx=100*(d['PDI']-d['MDI']).abs()/(d['PDI']+d['MDI']).replace(0,np.nan); d['ADX']=dx.ewm(alpha=1/14,adjust=False).mean()
    d['VOL_AVG20']=d['Volume'].rolling(20,min_periods=5).mean() if 'Volume' in d else np.nan
    # OBV
    if 'Volume' in d:
        direction=np.sign(d['Close'].diff()).fillna(0); d['OBV']=(direction*d['Volume']).cumsum(); d['OBV_MA20']=d['OBV'].rolling(20,min_periods=5).mean()
    else: d['OBV']=d['OBV_MA20']=np.nan
    # CMF
    if 'Volume' in d:
        mfm=((d['Close']-d['Low'])-(d['High']-d['Close']))/(d['High']-d['Low']).replace(0,np.nan); mfv=mfm*d['Volume']; d['CMF']=mfv.rolling(20,min_periods=10).sum()/d['Volume'].rolling(20,min_periods=10).sum()
    else: d['CMF']=np.nan
    return d.dropna(subset=['Close'])

# ---------- Historical data ----------
@st.cache_data(ttl=300, show_spinner=False)
def download_history(symbol, interval):
    period={'1d':'2y','1wk':'5y','1h':'59d'}[interval]
    try:
        d=yf.download(symbol,period=period,interval=interval,auto_adjust=False,progress=False,threads=False)
        if d is None or d.empty: return pd.DataFrame()
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        cols=[c for c in ['Open','High','Low','Close','Volume'] if c in d.columns]
        return d[cols].dropna(subset=['Open','High','Low','Close'])
    except Exception:
        return pd.DataFrame()

# Live CMP is intentionally NOT cached.
def live_cmp(symbol, fallback):
    try:
        t=yf.Ticker(symbol)
        fi=t.fast_info
        for k in ('last_price','regularMarketPrice'):
            try:
                v=fi.get(k)
                if v is not None and np.isfinite(float(v)): return float(v)
            except Exception: pass
    except Exception: pass
    try:
        d=yf.download(symbol,period='2d',interval='1m',auto_adjust=False,progress=False,threads=False)
        if d is not None and not d.empty:
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            return float(d['Close'].dropna().iloc[-1])
    except Exception: pass
    return fallback

# ---------- Sell-side core ----------
def analyse(d):
    if len(d)<40: return None
    x=indicators(d)
    if len(x)<40: return None
    # A bearish reference/support candle: stochastic below 20 AND MACD below zero.
    refmask=(x['STOCH']<20)&(x['MACD']<0)
    refs=np.flatnonzero(refmask.fillna(False).to_numpy())
    if len(refs)==0: return None
    last=float(x['Close'].iloc[-1])
    # Search recent bearish reference levels; choose the latest one that has a later candle.
    for rp in refs[::-1]:
        if rp>=len(x)-1: continue
        support=float(x['Low'].iloc[rp])
        later=x.iloc[rp+1:]
        # A fresh confirmed breakdown: close below support + bearish stochastic/MACD.
        bm=(later['Close']<support)&(later['STOCH']<20)&(later['MACD']<0)
        bpos=np.flatnonzero(bm.fillna(False).to_numpy())
        if len(bpos):
            bp=bpos[-1]; b=later.iloc[bp]
            # If price subsequently recovered above support, this is not a live confirmed breakdown.
            if last < support and float(x['MACD'].iloc[-1])<0:
                break_date=later.index[bp]
                return build_result(x,support,rp,last,'🔴 BREAKDOWN CONFIRMED',break_date,b)
            # Otherwise continue looking for a newer valid support.
        # Near breakdown is not a filter: it is a classification when price is above but close to support.
        if last>=support:
            dist=(last/support-1)*100 if support else 999
            if dist<=3.0 and float(x['MACD'].iloc[-1])<0 and float(x['STOCH'].iloc[-1])<35:
                return build_result(x,support,rp,last,'🟠 NEAR BREAKDOWN',None,None)
    return None

def build_result(x,support,rp,cmp,status,break_date,b):
    last=x.iloc[-1]
    # Extraordinary markers: informational, not rejection conditions.
    score=0; marks=[]
    if float(last['EMA50'])<float(last['EMA150'])<float(last['EMA200']): score+=1; marks.append('50/150/200 DMA bearish')
    if len(x)>=252 and float(last['Close'])<=float(x['Low'].rolling(252,min_periods=50).min().iloc[-1])*1.10: score+=1; marks.append('Near 52W low')
    if float(last['Close'])<float(last['EMA20']): score+=1; marks.append('Relative/price weakness')
    if pd.notna(last.get('ADX')) and float(last['ADX'])>20 and float(last['MDI'])>float(last['PDI']): score+=1; marks.append('ADX/-DI bearish')
    if pd.notna(last.get('OBV')) and pd.notna(last.get('OBV_MA20')) and float(last['OBV'])<float(last['OBV_MA20']): score+=1; marks.append('OBV falling')
    if pd.notna(last.get('CMF')) and float(last['CMF'])<0: score+=1; marks.append('CMF negative')
    if pd.notna(last.get('VOL_AVG20')) and float(last.get('Volume',np.nan))>float(last['VOL_AVG20'])*1.5: score+=1; marks.append('Volume expansion')
    if len(x)>=20:
        mid=x['Close'].rolling(20).mean(); sd=x['Close'].rolling(20).std(); bw=(4*sd/mid.replace(0,np.nan)); recent=float(bw.iloc[-1]); avg_bw=float(bw.rolling(60,min_periods=20).mean().iloc[-1]) if len(bw)>=20 else np.nan
        if pd.notna(recent) and pd.notna(avg_bw) and recent<avg_bw*0.75: score+=1; marks.append('Bollinger squeeze')
    if pd.notna(last.get('ATR')) and len(x)>=28 and float(last['ATR'])>float(x['ATR'].rolling(20,min_periods=5).mean().iloc[-1])*1.2: score+=1; marks.append('ATR expansion')
    if len(x)>=252 and float(last['Close'])<=float(x['High'].rolling(252,min_periods=50).max().iloc[-1])*0.80: score+=1; marks.append('52W weakness')
    if len(x)>=20:
        ranges=(x['High']-x['Low'])/x['Close'].replace(0,np.nan); rr=float(ranges.iloc[-1]); rravg=float(ranges.rolling(20,min_periods=10).mean().iloc[-1])
        if pd.notna(rr) and pd.notna(rravg) and rr<rravg*0.75: score+=1; marks.append('Range contraction')
    return {'Support Price':support,'CMP':cmp,'Status':status,'Breakdown Date':break_date,'Stoch':float(last['STOCH']),'MACD':float(last['MACD']),'Extraordinary Score':score,'Extraordinary Markers':', '.join(marks) if marks else '—'}

# ---------- UI ----------
symbols=get_nifty500()
col1,col2,col3=st.columns(3)
with col1: interval=st.selectbox('Timeframe',['1d','1wk'],format_func=lambda z:{'1d':'1 Day','1wk':'1 Week'}[z])
with col2: limit=st.slider('Stocks to scan',20,min(500,max(20,len(symbols))),min(500,len(symbols)))
with col3: topn=st.selectbox('Show results',[20,50,100,500],index=2)

st.info('Sell Core: support = prior LOW formed with Stochastic <20 and MACD <0. Breakdown Confirmed = later CLOSE below that support with Stochastic <20 and MACD <0, while current CMP remains below support and MACD <0. Near Breakdown = CMP is above support but within 3%, with bearish momentum. Extraordinary markers are informational and never reject a Core signal.')
run=st.button('🔎 Scan NIFTY 500',type='primary',use_container_width=True)

if run:
    rows=[]; progress=st.progress(0); status=st.empty(); started=datetime.now()
    scan=symbols[:limit]
    for i,sym in enumerate(scan,1):
        status.write(f'Scanning {i}/{len(scan)} — {sym.replace(".NS","")}')
        hist=download_history(sym,interval)
        a=analyse(hist) if not hist.empty else None
        if a:
            # Replace historical close with an uncached live/latest CMP on every scan.
            a['CMP']=live_cmp(sym,a['CMP'])
            # Reclassify using live CMP so stale historical close cannot create a false status.
            if a['CMP']<a['Support Price'] and a['MACD']<0:
                if a['Status']=='🟠 NEAR BREAKDOWN' or a['Breakdown Date'] is not None: a['Status']='🔴 BREAKDOWN CONFIRMED'
            elif a['CMP']>=a['Support Price'] and (a['CMP']/a['Support Price']-1)*100<=3 and a['MACD']<0 and a['Stoch']<35:
                a['Status']='🟠 NEAR BREAKDOWN'; a['Breakdown Date']=None
            else:
                continue
            a['Stock']=sym.replace('.NS',''); a['Timeframe']={'1d':'1 Day','1wk':'1 Week'}[interval]
            rows.append(a)
        progress.progress(i/len(scan))
    progress.empty(); status.empty()
    result=pd.DataFrame(rows)
    if result.empty:
        st.warning('No live Core sell candidates were found in this scan. This is a real result, not a hidden filter; try the other timeframe.')
    else:
        result['Distance to Support %']=((result['CMP']/result['Support Price'])-1)*100
        order={'🔴 BREAKDOWN CONFIRMED':0,'🟠 NEAR BREAKDOWN':1}
        result['_o']=result['Status'].map(order).fillna(9)
        result=result.sort_values(['_o','Extraordinary Score','Distance to Support %'],ascending=[True,False,True]).head(topn).reset_index(drop=True)
        # Requested adjacent layout: CMP | Support Price | Breakdown Status
        cols=['Stock','CMP','Support Price','Status','Distance to Support %','Stoch','MACD','Extraordinary Score','Extraordinary Markers','Breakdown Date','Timeframe']
        result=result[cols]
        result['CMP']=result['CMP'].round(2); result['Support Price']=result['Support Price'].round(2); result['Distance to Support %']=result['Distance to Support %'].round(2); result['Stoch']=result['Stoch'].round(2); result['MACD']=result['MACD'].round(3)
        st.success(f'{len(result)} sell candidate(s) shown • {len(scan)} stocks scanned • refreshed {datetime.now().strftime("%H:%M:%S")}')
        st.dataframe(result,use_container_width=True,hide_index=True)
        st.download_button('⬇️ Download CSV',result.to_csv(index=False).encode('utf-8'),file_name=f'nifty500_sell_{interval}.csv',mime='text/csv')

st.caption('Data: Yahoo Finance via yfinance. CMP is requested live/latest on each scan and is not Streamlit-cached. Technical screening only; not investment advice.')
