#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent, transparent public-Dante research indicators (NOT proprietary).

Officially described ideas: 5-day MA turn ("Odori"), reverse MA +
accumulation candle + multi-month base + breakout, EMA112/224/448 progression,
EMA5/112/224 early 256, and post-breakout MA3 hill/pullback.

Unpublished paid formulas for Blue Dot, Watermelon, Rainbow/Bridge remain
UNKNOWN. BB/ATR/PSAR overlays are standard comparison candidates, explicitly
not original Dante signals. Absolutely no trading orders.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EMA_PERIODS=(3,5,15,20,33,56,60,112,224,448)
HORIZON=20
FEE=.4
START=pd.Timestamp("2018-01-01")
WEEK_FROM=pd.Timestamp("2026-11-09")
WEEK_END=pd.Timestamp("2026-11-13")
SECTIONS={
    "ODORI_EMA5_PROXY":"B_early",
    "ODORI_SMA5_PROXY":"B_early",
    "BOTTOM_ODORI_PROXY":"B_early",
    "REVERSE_ODORI_PROXY":"B_early",
    "PUBLIC_256_LONG_PROXY":"B_early",
    "MA_HIT_112_PROXY":"B_early",
    "MA_HIT_224_PROXY":"A_224",
    "REVERSE_ACCUM_CONCRETE_PROXY":"A_base",
    "EMA3_HILL_REBREAK_PROXY":"C_follow",
}
OFFICIAL_STRUCTURE={"PUBLIC_256_LONG_PROXY","MA_HIT_112_PROXY","MA_HIT_224_PROXY"}
COOLDOWN=20


def parabolic_sar(high, low, step=.02, max_step=.2):
    """Wilder-style SAR with bounded acceleration; for research overlay only."""
    h=np.asarray(high,dtype=float);l=np.asarray(low,dtype=float)
    n=len(h);sar=np.full(n,np.nan);bull=np.zeros(n,dtype=bool)
    if n<3 or not (np.isfinite(h).all() and np.isfinite(l).all()):
        return sar,bull
    up=(h[1]+l[1])>=(h[0]+l[0])
    extreme=h[0] if up else l[0]
    value=l[0] if up else h[0]
    af=step
    sar[0]=value; bull[0]=up
    for i in range(1,n):
        value=value+af*(extreme-value)
        if up:
            value=min(value,l[i-1],l[max(0,i-2)])
            if l[i]<value:
                up=False;value=extreme;extreme=l[i];af=step
            elif h[i]>extreme:
                extreme=h[i];af=min(max_step,af+step)
        else:
            value=max(value,h[i-1],h[max(0,i-2)])
            if h[i]>value:
                up=True;value=extreme;extreme=h[i];af=step
            elif l[i]<extreme:
                extreme=l[i];af=min(max_step,af+step)
        sar[i]=value;bull[i]=up
    return sar,bull


def prepare(group):
    z=group.sort_values("date").reset_index(drop=True).copy()
    z["date"]=pd.to_datetime(z.date)
    raw=z.close.astype(float)
    a=z.adjusted_close.astype(float)
    fac=(a/raw).replace([np.inf,-np.inf],np.nan).ffill().bfill()
    for field,alias in (("open","ao"),("high","ah"),("low","al")):
        z[alias]=z[field].astype(float)*fac
    z["ac"]=a
    for span in EMA_PERIODS:
        z[f"ema{span}"]=a.ewm(span=span,adjust=False,min_periods=span).mean()
    z["sma5"]=a.rolling(5,min_periods=5).mean()
    z["prevolume20"]=z.volume.shift(1).rolling(20,min_periods=20).mean()
    z["volratio"]=z.volume/z.prevolume20
    z["preamount20"]=z.amount.shift(1).rolling(20,min_periods=20).mean()
    z["prev20hi"]=z.ah.shift(1).rolling(20,min_periods=20).max()
    z["prev5hi"]=z.ah.shift(1).rolling(5,min_periods=5).max()
    z["ema224_20d_pct"]=(z.ema224/z.ema224.shift(20)-1)*100
    z["prior80_below224"]=(z.ac<z.ema224).shift(1).rolling(80,min_periods=80).sum()
    spread=(z.ah-z.al).replace(0,np.nan)
    z["closepos"]=(z.ac-z.al)/spread
    z["wickup"]=(z.ah-np.maximum(z.ao,z.ac))/spread
    z["body"]=z.ac/z.ao-1
    z["atr14"]=(pd.concat([z.ah-z.al,(z.ah-z.ac.shift(1)).abs(),
                 (z.al-z.ac.shift(1)).abs()],axis=1).max(axis=1)
                 .ewm(alpha=1/14,adjust=False,min_periods=14).mean())
    for n in (20,35,40):
        mid=z.ac.rolling(n,min_periods=n).mean()
        sd=z.ac.rolling(n,min_periods=n).std(ddof=0)
        z[f"bb{n}top"]=mid+2*sd
    # Standard ATR channel proxy, NOT 'blue dotted line'.
    mid=z.ac.ewm(span=35,adjust=False,min_periods=35).mean()
    z["atr_channel_top_proxy"]=mid+2*z.atr14
    sar,bull=parabolic_sar(z.ah.to_numpy(float),z.al.to_numpy(float))
    z["psar_standard"]=sar;z["psar_bull"]=bull
    z["ema5_upturn"]=(z.ema5>z.ema5.shift(1))&(z.ema5.shift(1)<=z.ema5.shift(2))
    z["sma5_upturn"]=(z.sma5>z.sma5.shift(1))&(z.sma5.shift(1)<=z.sma5.shift(2))
    return z


def base_quality(z, start, end, min_days=60):
    """No forward information; candidate true base ending immediately before event."""
    if end-start<min_days or start<0:return None
    p=z.ac.iloc[start:end].to_numpy(float)
    if not np.isfinite(p).all() or np.min(p)<=0:return None
    lo,hi=np.quantile(p,[.1,.9]);width=hi/lo-1
    net=p[-1]/p[0]-1
    delayed=np.min(p[max(10,len(p)//3):]) < .92*np.min(p[:max(10,len(p)//3)])
    peak=float(z.ah.iloc[max(0,start-260):start].max())
    drop=p[:10].mean()/peak-1 if peak>0 else np.nan
    ok=bool(width<=.25 and -.10<=net<=.15 and not delayed and drop<=-.20)
    return dict(ok=ok,days=end-start,width_pct=100*width,net_pct=100*net,
                decline_pct=100*drop,anchor_i=start)


def event_tests(z,i):
    """Return named hypotheses firing AS OF bar i and their diagnostics."""
    r=z.iloc[i];p=z.iloc[i-1]
    if any(not np.isfinite(float(r[k])) for k in
           ("ac","ao","ah","al","ema112","ema224","ema448","ema5",
            "volratio","preamount20","body","closepos","atr14")):return []
    oldlow=float(z.ac.iloc[max(0,i-80):i].min())
    oldhigh=float(z.ac.iloc[max(0,i-260):i].max())
    decline=oldlow/oldhigh-1 if oldhigh>0 else np.nan
    reverse=bool(r.ema112<r.ema224<r.ema448)
    below224=float(r.prior80_below224)
    v=float(r.volratio)
    out=[]
    def put(name,reason,base=None):
        out.append((name,reason,base))
    if r.ema5_upturn and r.ac>r.ema5 and v>=1.0:
        put("ODORI_EMA5_PROXY","EMA5 upturn, price above EMA5")
    if r.sma5_upturn and r.ac>r.sma5 and v>=1.0:
        put("ODORI_SMA5_PROXY","SMA5 upturn, price above SMA5")
    if r.ema5_upturn and r.ac>r.ema5 and decline<=-.25 and r.ac<r.ema224 and v>=1.2:
        put("BOTTOM_ODORI_PROXY","EMA5 upturn after >=25% 80/260-day decline")
    if reverse and r.ema5_upturn and r.ac>=r.ema5 and r.ac<r.ema224 and v>=1.15:
        put("REVERSE_ODORI_PROXY","EMA5 upturn in reversed 112<224<448")
    if (reverse and below224>=48 and r.ema5>r.ema112 and
        p.ema5<=p.ema112 and r.ema112<=r.ac<r.ema224):
        put("PUBLIC_256_LONG_PROXY","EMA5 crosses 112, EMA224 overhead")
    if (reverse and below224>=48 and r.ac>=r.ema112 and p.ac<p.ema112 and
        r.ac<r.ema224 and v>=1.3):
        put("MA_HIT_112_PROXY","new EMA112 recovery under EMA224")
    if (below224>=48 and r.ac>r.ema224 and p.ac<=p.ema224 and
        r.ac<r.ema448 and v>=1.3):
        put("MA_HIT_224_PROXY","EMA224 upward cross and EMA448 overhead")
    # A high-volume 20-day closing breakout after a stable, long bottom.
    if (r.ac>r.prev20hi and v>=1.6 and r.body>=.025 and r.closepos>=.65):
        for days in (90,60):
            b=base_quality(z,i-days,i,min_days=days)
            if b and b["ok"]:
                # Prior accumulation day inside the base: positive body, >=2x
                # volume at any earlier bar, followed by lower-volume base.
                during=z.iloc[i-days:i]
                accum=during[(during.volratio>=2)&(during.body>=.035)]
                if not accum.empty:
                    after=during.iloc[-min(20,len(during)):]
                    if after.volume.median() <= accum.volume.max()*.80:
                        put("REVERSE_ACCUM_CONCRETE_PROXY",
                            "stable base, past volume anchor, fresh resistance breakout",b)
                        break
    # Rebreakout through a real EMA3 'hill' (local 3EMA peak), with pullback.
    if (r.ac>r.ema224 and r.ac>r.prev5hi and v>=1.1 and i>=50):
        for j in range(i-17,i-4):
            if not (z.ema3.iloc[j]>z.ema3.iloc[j-1] and
                    z.ema3.iloc[j]>=z.ema3.iloc[j+1]):continue
            if z.ac.iloc[j]<=z.ema224.iloc[j]:continue
            after=z.iloc[j+1:i]
            if len(after)<3:continue
            pull=float(after.al.min()/z.ah.iloc[j]-1)
            if not -.20<=pull<=-.03:continue
            if after.volume.median()>z.volume.iloc[j]*.85:continue
            if float(after.ac.min())<float(z.ema224.iloc[j])*.97:continue
            if r.ac>z.ah.iloc[j]:
                put("EMA3_HILL_REBREAK_PROXY","confirmed EMA3 hill, quiet pullback, rebreak")
                break
    return out


def scan_group(g,min_amount):
    z=prepare(g)
    if len(z)<560:return [],[]
    events=[]
    # Week coverage means actual eligible price histories, not signal dates.
    weekmask=z.date.between(WEEK_FROM,WEEK_END)
    cov=[(str(t.date()),str(code).zfill(6))
         for t,code in zip(z.loc[weekmask,"date"],z.loc[weekmask,"code"])]
    # All public proxy event predicates require at least one of these precursors.
    # Vector prefilter skips impossible bars without skipping any signal family.
    active=(
        z.ema5_upturn | z.sma5_upturn |
        ((z.ema5>z.ema112)&(z.ema5.shift(1)<=z.ema112.shift(1))) |
        ((z.ac>=z.ema112)&(z.ac.shift(1)<z.ema112.shift(1))) |
        ((z.ac>z.ema224)&(z.ac.shift(1)<=z.ema224.shift(1))) |
        ((z.ac>z.prev20hi)&(z.volratio>=1.6)) |
        ((z.ac>z.prev5hi)&(z.volratio>=1.1))
    )
    active=active.fillna(False)&z.date.ge(START)&z.preamount20.ge(min_amount)
    candidate_idx=np.flatnonzero(active.to_numpy())
    last={name:-10_000 for name in SECTIONS}
    for i in candidate_idx:
        if i<560:continue
        date=pd.Timestamp(z.date.iloc[i])
        r=z.iloc[i]
        for name,why,b in event_tests(z,i):
            if i-last[name]<COOLDOWN:continue
            last[name]=i
            factor=float(r.ac/r.close) if r.close>0 else 1
            x=dict(date=str(date.date()),code=str(r.code).zfill(6),name=str(r["name"]),
                   exchange=str(r.exchange),series_id=str(r.series_id),
                   model=name,page=SECTIONS[name],official_formula="NO",
                   evidence="PUBLIC_CONCEPT_NUMERIC_HYPOTHESIS",
                   reason=why,signal_price=float(r.close),
                   volume_ratio=round(float(r.volratio),3),
                   ema224_dist_pct=round(float(100*(r.ac/r.ema224-1)),3),
                   ema224_slope20_pct=round(float(r.ema224_20d_pct),3),
                   atr14_pct=round(float(100*r.atr14/r.ac),3),
                   upper_wick_pct=round(float(100*r.wickup),2),
                   close_position_pct=round(float(100*r.closepos),2),
                   bb20_upper_proxy=float(r.bb20top/factor) if np.isfinite(r.bb20top) else np.nan,
                   bb35_upper_proxy=float(r.bb35top/factor) if np.isfinite(r.bb35top) else np.nan,
                   bb40_upper_proxy=float(r.bb40top/factor) if np.isfinite(r.bb40top) else np.nan,
                   psar_standard=float(r.psar_standard/factor),
                   base_days=b["days"] if b else np.nan,
                   base_width_pct=b["width_pct"] if b else np.nan,
                   anchor_support_price=float(r.ema224/factor))
            h=i+HORIZON
            x["mature_20d"]=h<len(z)
            if h<len(z):
                entry=float(z.ao.iloc[i+1])
                ret=float(z.ac.iloc[h]/entry-1)*100-FEE if entry>0 else np.nan
                mae=float(z.al.iloc[i+1:h+1].min()/entry-1)*100 if entry>0 else np.nan
                x["ret_20d_next_open_hold_cost_pct"]=ret
                x["mae_20d_pct"]=mae
            else:
                x["ret_20d_next_open_hold_cost_pct"]=np.nan
                x["mae_20d_pct"]=np.nan
            events.append(x)
    return events,cov


def build_chart(g,ev,out):
    z=prepare(g)
    dates=z.date.dt.strftime("%Y-%m-%d")
    hits=z.index[dates.eq(ev["date"])].tolist()
    if not hits:return
    i=int(hits[-1])
    left=max(0,i-160); t=z.iloc[left:i+1].copy().reset_index(drop=True)
    f=float(z.ac.iloc[i]/z.close.iloc[i])
    fig,(ax,av)=plt.subplots(2,1,figsize=(16.5,9.3),sharex=True,
                            gridspec_kw={"height_ratios":[4,.85]},facecolor="white")
    for j,r in t.iterrows():
        up=r.ac>=r.ao;col="#d94a5b" if up else "#3679ba"
        ax.vlines(j,r.al/f,r.ah/f,color=col,lw=.8)
        ax.add_patch(Rectangle((j-.32,min(r.ao,r.ac)/f),.64,max(abs(r.ac-r.ao)/f,.5),
                       linewidth=.4,edgecolor=col,facecolor=col))
        av.bar(j,r.volume/1e6,width=.75,color=col,alpha=.85)
    for n,col in ((5,"#b1b864"),(112,"#2a9b86"),(224,"#9c65d5"),(448,"#728092")):
        ax.plot(t[f"ema{n}"]/f,color=col,lw=1.4 if n<224 else 1.8,label=f"EMA{n}")
    # Real, daily HLC3*volume anchored at observed window start; chart-proxy
    # AVWAP, not intraday VWAP and not official proprietary Blue Dot.
    price=(t.ah+t.al+t.ac)/3
    v=t.volume.astype(float).clip(lower=0)
    avwap=(price*v).cumsum()/v.cumsum().replace(0,np.nan)
    ax.plot(avwap/f,lw=3.1,color="#f59b24",label="DAILY AVWAP (visible start)")
    ax.plot(t.bb35top/f,lw=1.25,color="#8d7be7",ls=":",label="BB35 upper PROXY")
    ax.scatter([len(t)-1],[z.ac.iloc[i]/f],marker="^",s=190,color="#e5aa31",
               edgecolors="#72470c",linewidth=1.3,zorder=12)
    ax.annotate(f"{ev['model']}\n{ev['date']}",(len(t)-1,z.ac.iloc[i]/f),
                xytext=(-140,36),textcoords="offset points",fontsize=10,
                color="#835e11",weight="bold",arrowprops=dict(arrowstyle="->",color="#c99028"))
    ax.axhline(float(ev["anchor_support_price"]),color="#bb4c55",ls="--",
               label="EMA224 reference",alpha=.48)
    av.set_ylabel("volume (million shares)");ax.set_ylabel("nominal KRW")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda a,b:f"{a:,.0f}"))
    ticks=list(range(0,len(t),max(1,len(t)//8)))
    if ticks[-1]!=len(t)-1:ticks.append(len(t)-1)
    av.set_xticks(ticks)
    av.set_xticklabels([str(t.date.iloc[j].date()) for j in ticks],rotation=25,ha="right")
    ax.grid(alpha=.15);av.grid(alpha=.1)
    ax.legend(loc="upper left",fontsize=8,ncol=3)
    fig.suptitle(f"{ev['name']} {ev['code']} — independent Dante-inspired study",
                 fontsize=15,weight="bold")
    fig.text(.025,.018,"Unverified research proxy, not original paid Dante indicator. "
             "No orders. OHLCV source: marcap and/or Naver; daily AVWAP uses displayed window start.",
             fontsize=9,color="#455669")
    fig.tight_layout(rect=[0,.035,1,.96])
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=135);plt.close(fig)


def self_test():
    c=pd.Series([10,9,8,7,6,6,6.2,6.5,7,8],dtype=float)
    ema=c.ewm(span=5,adjust=False,min_periods=5).mean()
    assert pd.isna(ema.iloc[3]) and ema.iloc[-1]>ema.iloc[-2]
    ups=ema.gt(ema.shift(1))&ema.shift(1).le(ema.shift(2))
    assert ups.any()
    s,b=parabolic_sar(np.arange(1,35,dtype=float)+3,np.arange(1,35,dtype=float))
    assert np.isfinite(s).all() and b[-1]
    # Same-day horizons must not contain future data or label immature signals.
    assert len(c)==10 and HORIZON>len(c)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--min-amount",type=float,default=5e9)
    p.add_argument("--max-charts",type=int,default=12)
    p.add_argument("--test-only",action="store_true")
    args=p.parse_args()
    self_test()
    if args.test_only:
        print("SELF_TEST_PASSED",flush=True);return
    from dante_classic_scanner import load
    market=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    args.out.mkdir(exist_ok=True,parents=True)
    events=[];cov=[]
    groups={}
    for n,(sid,g) in enumerate(market.groupby("series_id",sort=False),1):
        e,c=scan_group(g,args.min_amount)
        events.extend(e);cov.extend(c)
        if n%300==0:print("episodes",n,"events",len(events),flush=True)
        if e and any(WEEK_FROM<=pd.Timestamp(x["date"])<=WEEK_END for x in e):
            groups[str(sid)]=g
    df=pd.DataFrame(events)
    if df.empty:raise RuntimeError("No research events generated; investigate panel")
    df["split"]=np.select([df.date.str[:4].astype(int)<=2021,
                          df.date.str[:4].astype(int)<=2024],
                          ["TRAIN","VALID"],default="TEST")
    df.to_csv(args.out/"all_research_events.csv",index=False,encoding="utf-8-sig")
    week=df[df.date.between(str(WEEK_FROM.date()),str(WEEK_END.date()))].copy()
    week.to_csv(args.out/"week_research_events.csv",index=False,encoding="utf-8-sig")
    recent=week.drop_duplicates(["date","code","model"])
    recent.groupby(["date","page","model"]).size().to_csv(args.out/"week_counts.csv",header=["signals"],encoding="utf-8-sig")
    pd.DataFrame(cov,columns=["date","code"]).groupby("date").code.nunique().to_csv(
        args.out/"daily_coverage.csv",header=["codes"],encoding="utf-8-sig")
    summaries=[]
    for (typ,phase),a in df.groupby(["model","split"]):
        good=a.dropna(subset=["ret_20d_next_open_hold_cost_pct"])
        v=good.ret_20d_next_open_hold_cost_pct
        summaries.append(dict(model=typ,phase=phase,events=len(a),mature20=len(good),
                              mean20_pct=v.mean(),median20_pct=v.median(),
                              win20_pct=(v>0).mean()*100,mean_mae20_pct=good.mae_20d_pct.mean(),
                              year2025=int((a.date.str[:4]=="2025").sum()),
                              year2026=int((a.date.str[:4]=="2026").sum())))
    pd.DataFrame(summaries).to_csv(args.out/"historical_summary.csv",index=False,encoding="utf-8-sig")
    # Keep weekly charts modest; only actual new signal dates, per stock once.
    seen=set()
    for r in week.sort_values(["date","volume_ratio"],ascending=[False,False]).to_dict("records"):
        if r["code"] in seen:continue
        seen.add(r["code"])
        if len(seen)>args.max_charts:break
        try:
            build_chart(groups[r["series_id"]],r,args.out/"charts"/f"{r['date']}_{r['code']}_{r['model']}.png")
        except Exception as exc:
            print("CHART_FAILED",r["code"],repr(exc),flush=True)
    print("PANEL_LAST",market.date.max(),"NEW_EVENTS",len(week),"NUNIQUE",week.code.nunique(),flush=True)
    print("WEEK\n",week.groupby(["date","model"]).size().to_string(),flush=True)
    print("HISTORICAL\n",pd.DataFrame(summaries).to_string(index=False),flush=True)
    (args.out/"README.txt").write_text(
       "Public-concept numeric research proxies only. No original paid Dante formulas. "
       "All events as of each signal close; relative volume uses preceding 20 sessions. "
       "Backtest: next adjusted open -> +20 trading sessions adjusted close, costs .4 percentage points, no stop. "
       "Data adjustment is research-grade and may differ from broker-grade adjusted prices. "
       "No actual orders and NO proof of predictive edge.\n",
       encoding="utf-8")

if __name__=="__main__":
    main()
