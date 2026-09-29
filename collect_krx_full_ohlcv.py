#!/usr/bin/env python3
import argparse, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import FinanceDataReader as fdr

def first(df, names):
    m={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in m: return m[n.lower()]
    return None

def listing(df, hint, active):
    x=df.copy()
    sc=first(x,["Symbol","Code"])
    if sc is None: raise ValueError("listing has no Symbol/Code")
    x["code"]=x[sc].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    mc=first(x,["Market","Exchange"])
    x["exchange"]=(x[mc].astype(str).str.upper() if mc else hint)
    x["exchange"]=x["exchange"].replace({"KOSPI":"KO","KOSDAQ":"KQ"})
    x=x[x["exchange"].isin(["KO","KQ"])].copy()
    nc=first(x,["Name"]); lc=first(x,["ListingDate","Listing Date"]); dc=first(x,["DelistingDate","Delisting Date"])
    x["name"]=x[nc].astype(str) if nc else ""
    x["listing_date"]=pd.to_datetime(x[lc],errors="coerce") if lc else pd.NaT
    x["delisting_date"]=pd.to_datetime(x[dc],errors="coerce") if dc else pd.NaT
    x["active"]=active
    x["source"]="ACTIVE" if active else "DELISTED"
    x["series_id"]=x.apply(lambda r:f"{r.exchange}_{r.code}_{r.listing_date.strftime('%Y%m%d') if pd.notna(r.listing_date) else 'UNKNOWN'}_{r.source[0]}",axis=1)
    return x[["series_id","code","exchange","name","active","source","listing_date","delisting_date"]]

def universe(start,end):
    ko=listing(fdr.StockListing("KOSPI"),"KO",True)
    kq=listing(fdr.StockListing("KOSDAQ"),"KQ",True)
    try: raw=fdr.StockListing("KRX-DELISTING",start.strftime("%Y-%m-%d"),end.strftime("%Y-%m-%d"))
    except TypeError: raw=fdr.StockListing("KRX-DELISTING")
    de=listing(raw,None,False)
    de=de[de.delisting_date.isna() | (de.delisting_date>=start)]
    return pd.concat([ko,kq,de],ignore_index=True).drop_duplicates("series_id")

def fetch(rec,start,end,cache,retries,delay):
    qs=max(start,rec["listing_date"]) if pd.notna(rec["listing_date"]) else start
    qe=min(end,rec["delisting_date"]) if pd.notna(rec["delisting_date"]) else end
    if qs>qe: return {**rec,"rows":0,"error":"outside_window"}
    p=cache/rec["exchange"]/f"{rec['series_id']}.parquet"; p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        try:
            q=pd.read_parquet(p,columns=["date"])
            d=pd.to_datetime(q.date,errors="coerce").dropna()
            if len(d) and d.min()<=qs+pd.Timedelta(days=10) and d.max()>=qe-pd.Timedelta(days=10):
                return {**rec,"rows":len(q),"error":None,"cached":True}
        except Exception: pass
    err=None
    for attempt in range(retries):
        try:
            sym=rec["code"] if rec["active"] else f"KRX-DELISTING:{rec['code']}"
            q=fdr.DataReader(sym,qs.strftime("%Y-%m-%d"),qe.strftime("%Y-%m-%d"))
            if q is None or q.empty: return {**rec,"rows":0,"error":"empty","cached":False}
            q=q.reset_index()
            ren={}
            for dst,names in {"date":["Date","index"],"open":["Open"],"high":["High"],"low":["Low"],"close":["Close"],"volume":["Volume"],"amount":["Amount"]}.items():
                src=first(q,names)
                if src is not None: ren[src]=dst
            q=q.rename(columns=ren)
            need=["date","open","high","low","close","volume"]
            if any(c not in q for c in need): raise ValueError(f"missing columns {list(q.columns)}")
            keep=need+(["amount"] if "amount" in q else [])
            q=q[keep].copy()
            q["date"]=pd.to_datetime(q.date,errors="coerce")
            for c in keep[1:]: q[c]=pd.to_numeric(q[c],errors="coerce")
            q=q.dropna(subset=need)
            q=q[(q.date>=qs)&(q.date<=qe)].sort_values("date").drop_duplicates("date")
            if q.empty: return {**rec,"rows":0,"error":"empty_after_clip","cached":False}
            for k in ["series_id","code","exchange","name","active","listing_date","delisting_date"]: q[k]=rec[k]
            q.to_parquet(p,index=False)
            time.sleep(delay+random.random()*delay)
            return {**rec,"rows":len(q),"error":None,"cached":False}
        except Exception as e:
            err=repr(e); time.sleep((attempt+1)*1.2+random.random())
    return {**rec,"rows":0,"error":err,"cached":False}

def merge(cache,ex,out):
    fs=[]
    for p in sorted((cache/ex).glob("*.parquet")):
        try:
            q=pd.read_parquet(p)
            if not q.empty: fs.append(q)
        except Exception as e: print("skip",p,e,flush=True)
    if not fs: return 0,0
    x=pd.concat(fs,ignore_index=True).sort_values(["series_id","date"]).drop_duplicates(["series_id","date"])
    x.to_parquet(out,index=False)
    return len(x),x.series_id.nunique()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2016-01-01"); ap.add_argument("--end",default=None)
    ap.add_argument("--out",type=Path,default=Path("krx_data")); ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--retries",type=int,default=4); ap.add_argument("--delay",type=float,default=.2)
    ap.add_argument("--max-failure-rate",type=float,default=.20)
    a=ap.parse_args()
    start=pd.Timestamp(a.start); end=pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize()
    a.out.mkdir(parents=True,exist_ok=True); cache=a.out/"symbols"; cache.mkdir(exist_ok=True)
    u=universe(start,end).sort_values(["exchange","code","listing_date"])
    u.to_csv(a.out/"universe.csv",index=False,encoding="utf-8-sig")
    print(f"episodes={len(u):,} active={int(u.active.sum()):,}",flush=True)
    logs=[]
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as pool:
        futs=[pool.submit(fetch,r,start,end,cache,a.retries,a.delay) for r in u.to_dict("records")]
        for i,f in enumerate(as_completed(futs),1):
            logs.append(f.result())
            if i%50==0 or i==len(futs): print(f"[{i}/{len(futs)}] ok={sum(z['rows']>0 for z in logs)}",flush=True)
    log=pd.DataFrame(logs); log.to_csv(a.out/"collection_log.csv",index=False,encoding="utf-8-sig")
    for ex,name in [("KO","ko_eod.parquet"),("KQ","kq_eod.parquet")]:
        rows,series=merge(cache,ex,a.out/name); print(name,rows,series,flush=True)
    attempted=int((log.error!="outside_window").sum()); failed=int(((log.rows<=0)&(log.error!="outside_window")).sum())
    rate=failed/attempted if attempted else 1
    print(f"failure_rate={rate:.2%} ({failed}/{attempted})",flush=True)
    if rate>a.max_failure_rate: raise SystemExit("collection failure rate too high")

if __name__=="__main__": main()
