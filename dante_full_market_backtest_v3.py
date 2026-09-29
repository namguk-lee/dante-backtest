#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd

def read(p): return pd.read_parquet(p) if p.suffix.lower() in [".parquet",".pq"] else pd.read_csv(p,low_memory=False)

def prep(df,ex):
    low={str(c).lower():c for c in df.columns}; ren={}
    for dst,names in {"code":["code","symbol","ticker"],"date":["date"],"open":["open"],"high":["high"],"low":["low"],"close":["close"],"volume":["volume","vol"],"series_id":["series_id"],"exchange":["exchange"],"name":["name"],"amount":["amount"]}.items():
        for n in names:
            if n in low: ren[low[n]]=dst; break
    x=df.rename(columns=ren).copy()
    for c in ["code","date","open","high","low","close","volume"]:
        if c not in x: raise ValueError(f"missing {c}")
    if "exchange" not in x: x["exchange"]=ex
    if "series_id" not in x: x["series_id"]=x.exchange.astype(str)+"_"+x.code.astype(str)
    if "name" not in x: x["name"]=""
    x["code"]=x.code.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    for c in ["open","high","low","close","volume"]+(['amount'] if 'amount' in x else []): x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["date","open","high","low","close","volume"])
    x=x[(x.open>0)&(x.high>0)&(x.low>0)&(x.close>0)]
    return x.sort_values(["exchange","series_id","date"]).drop_duplicates(["exchange","series_id","date"])

def indicators(g):
    g=g.sort_values("date").copy()
    for n in [112,224,448]: g[f"ema{n}"]=g.close.ewm(span=n,adjust=False,min_periods=n).mean()
    g["vol20"]=g.volume.rolling(20,min_periods=20).mean()
    turn=g.amount.where(g.amount>0,g.close*g.volume) if "amount" in g else g.close*g.volume
    g["turn20"]=turn.rolling(20,min_periods=20).mean()
    pc=g.close.shift(1)
    tr=pd.concat([g.high-g.low,(g.high-pc).abs(),(g.low-pc).abs()],axis=1).max(axis=1)
    g["atr14"]=tr.rolling(14,min_periods=14).mean()
    return g.reset_index(drop=True)

def fixed(g,entry_i,horizon,stop):
    entry=float(g.iloc[entry_i].open); end=min(entry_i+horizon,len(g)-1)
    if not(math.isfinite(stop) and 0<stop<entry): return float(g.iloc[end].close),end,False,"TIME"
    for j in range(entry_i,end+1):
        r=g.iloc[j]
        if r.open<=stop: return float(r.open),j,True,"STOP_GAP"
        if r.low<=stop: return float(stop),j,True,"STOP"
    return float(g.iloc[end].close),end,False,"TIME"

def ema_close(g,entry_i,horizon):
    end=min(entry_i+horizon,len(g)-1)
    for j in range(entry_i,end):
        if g.iloc[j].close<g.iloc[j].ema224: return float(g.iloc[j+1].open),j+1,True,"EMA224_CLOSE"
    return float(g.iloc[end].close),end,False,"TIME"

def split(dt,train,valid): return "TRAIN" if dt<=train else ("VALID" if dt<=valid else "TEST")

def scan(g,a):
    g=indicators(g); sigs=[]; trades=[]; n=len(g)
    if n<a.min_bars: return sigs,trades
    last=-99999; train=pd.Timestamp(a.train_end); valid=pd.Timestamp(a.valid_end); s0=pd.Timestamp(a.signal_start)
    for i in range(max(448,a.below_window+1),n-a.horizon-1):
        if i-last<a.cooldown: continue
        r=g.iloc[i]; p=g.iloc[i-1]
        if r.date<s0 or not np.isfinite([r.ema112,r.ema224,r.ema448,r.vol20]).all(): continue
        if not(r.close>r.ema224 and p.close<=p.ema224): continue
        h=g.iloc[i-a.below_window:i]
        below=int((h.close<h.ema224).sum())
        if below<a.below_min or not(r.ema112<r.ema224<r.ema448): continue
        if not(r.volume>=a.volume_multiple*r.vol20 and r.close>r.open): continue
        if a.min_turnover>0 and r.turn20<a.min_turnover: continue
        ei=i+1; entry=float(g.iloc[ei].open)
        if entry<a.min_entry_price: continue
        last=i; sp=split(r.date,train,valid)
        sigs.append({"exchange":r.exchange,"series_id":r.series_id,"code":r.code,"name":r.get("name",""),"signal_date":r.date,"entry_date":g.iloc[ei].date,"entry":entry,"ema112":r.ema112,"ema224":r.ema224,"ema448":r.ema448,"below80":below,"volume_ratio":r.volume/r.vol20,"turnover20":r.turn20,"fwd20":g.iloc[ei+min(20,a.horizon)].close/entry-1,"fwd60":g.iloc[ei+a.horizon].close/entry-1,"split":sp})
        swing=float(g.iloc[max(0,i-19):i+1].low.min())
        methods={"S1_EMA224_INTRADAY_2PCT":fixed(g,ei,a.horizon,float(r.ema224*.98)),
                 "S2_EMA224_CLOSE":ema_close(g,ei,a.horizon),
                 "S3_SIGNAL_LOW":fixed(g,ei,a.horizon,float(r.low*.995)),
                 "S4_SWING20_LOW":fixed(g,ei,a.horizon,float(swing*.995)),
                 "S5_ATR1_5":fixed(g,ei,a.horizon,float(entry-1.5*r.atr14) if math.isfinite(r.atr14) else np.nan)}
        for m,(px,xi,stopped,reason) in methods.items():
            gross=px/entry-1
            trades.append({"exchange":r.exchange,"series_id":r.series_id,"code":r.code,"name":r.get("name",""),"signal_date":r.date,"entry_date":g.iloc[ei].date,"exit_date":g.iloc[xi].date,"method":m,"entry":entry,"exit":px,"gross_return":gross,"net_return":gross-a.cost_bps/10000,"hold_days":xi-ei,"stopped":stopped,"exit_reason":reason,"split":sp})
    return sigs,trades

def summary(t,cols):
    rows=[]
    for k,q in t.groupby(cols,dropna=False):
        if not isinstance(k,tuple): k=(k,)
        r=dict(zip(cols,k)); x=q.net_return.sort_values().to_numpy(); trim=x[int(len(x)*.1):len(x)-int(len(x)*.1)] if len(x)>=10 else x
        r.update(n=len(q),mean_return=q.net_return.mean(),median_return=q.net_return.median(),trim10_mean_return=float(np.mean(trim)) if len(trim) else np.nan,win_rate=(q.net_return>0).mean(),stop_rate=q.stopped.mean(),avg_hold_days=q.hold_days.mean(),p10=q.net_return.quantile(.1),p90=q.net_return.quantile(.9))
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ko",type=Path); ap.add_argument("--kq",type=Path); ap.add_argument("--out",type=Path,default=Path("dante_results"))
    ap.add_argument("--signal-start",default="2018-01-01"); ap.add_argument("--horizon",type=int,default=60); ap.add_argument("--cooldown",type=int,default=40)
    ap.add_argument("--below-window",type=int,default=80); ap.add_argument("--below-min",type=int,default=60); ap.add_argument("--volume-multiple",type=float,default=1.5)
    ap.add_argument("--min-bars",type=int,default=600); ap.add_argument("--min-entry-price",type=float,default=0); ap.add_argument("--min-turnover",type=float,default=0)
    ap.add_argument("--cost-bps",type=float,default=0); ap.add_argument("--train-end",default="2021-12-31"); ap.add_argument("--valid-end",default="2024-12-31")
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for p,ex in [(a.ko,"KO"),(a.kq,"KQ")]:
        if p: frames.append(prep(read(p),ex))
    df=pd.concat(frames,ignore_index=True)
    sigs=[]; trs=[]; groups=list(df.groupby(["exchange","series_id"],sort=False))
    for j,(_,g) in enumerate(groups,1):
        s,t=scan(g,a); sigs+=s; trs+=t
        if j%100==0 or j==len(groups): print(f"[{j}/{len(groups)}] signals={len(sigs)} trades={len(trs)}",flush=True)
    s=pd.DataFrame(sigs); t=pd.DataFrame(trs)
    s.to_csv(a.out/"signals.csv",index=False,encoding="utf-8-sig"); t.to_csv(a.out/"stop_trades.csv",index=False,encoding="utf-8-sig")
    if not t.empty:
        t["year"]=pd.to_datetime(t.signal_date).dt.year
        summary(t,["method"]).to_csv(a.out/"stop_comparison.csv",index=False,encoding="utf-8-sig")
        summary(t,["split","method"]).to_csv(a.out/"split_comparison.csv",index=False,encoding="utf-8-sig")
        summary(t,["year","method"]).to_csv(a.out/"yearly_comparison.csv",index=False,encoding="utf-8-sig")
        summary(t,["exchange","method"]).to_csv(a.out/"exchange_comparison.csv",index=False,encoding="utf-8-sig")
        if not s.empty:
            z=t.merge(s[["series_id","signal_date","turnover20"]],on=["series_id","signal_date"],how="left"); out=[]
            for th in [0,500_000_000,1_000_000_000,5_000_000_000]:
                q=z[z.turnover20>=th]
                if len(q): 
                    x=summary(q,["split","method"]); x["min_turnover20"]=th; out.append(x)
            if out: pd.concat(out).to_csv(a.out/"liquidity_comparison.csv",index=False,encoding="utf-8-sig")
        out=[]
        for bps in [0,20,40]:
            q=t.copy(); q["net_return"]=q.gross_return-bps/10000; x=summary(q,["split","method"]); x["cost_bps"]=bps; out.append(x)
        pd.concat(out).to_csv(a.out/"cost_sensitivity.csv",index=False,encoding="utf-8-sig")
    (a.out/"run_manifest.txt").write_text(f"rows={len(df)}\nseries={df.series_id.nunique()}\nsignal_start={a.signal_start}\n",encoding="utf-8")

if __name__=="__main__": main()
