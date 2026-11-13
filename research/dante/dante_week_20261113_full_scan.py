#!/usr/bin/env python3
"""One-off KOSPI/KOSDAQ full-market daily retrospective scan, no orders.

Research implementation of public chart rules; NOT proprietary Dante formula.
Every event uses only OHLCV known by its signal-day close. Last market date
and per-day coverage exported so missing 2026-11-13 bars are never concealed.
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

START=pd.Timestamp("2026-11-09")
END=pd.Timestamp("2026-11-13")
LENS=(40,60,90)
MIN_AMOUNT=5_000_000_000

def base_metrics(c, hi, i, duration):
    start=i-duration
    if start < 260: return None
    v=np.asarray(c[start:i],dtype=float)
    if len(v)!=duration or not np.isfinite(v).all() or (v<=0).any():return None
    qlo,qhi=np.quantile(v,[.1,.9])
    spread=qhi/qlo-1 if qlo>0 else np.inf
    change=v[-1]/v[0]-1
    m=max(10,duration//3)
    newlow=np.min(v[m:])<np.min(v[:m])*.92
    drawdown=np.min(v/np.maximum.accumulate(v)-1)
    prior_peak=np.max(hi[max(0,start-260):start])
    decline=np.median(v[:min(10,len(v))])/prior_peak-1 if prior_peak>0 else np.nan
    ok=(spread<=.30 and -.18<=change<=.20 and
        not newlow and drawdown>=-.26 and
        np.isfinite(decline) and decline<=-.20)
    return dict(days=duration,base_range_pct=100*spread,
                base_net_pct=100*change,
                previous_peak_decline_pct=100*decline,
                base_start_i=start,passes=ok)

def scan_series(g):
    g=g.sort_values("date").reset_index(drop=True)
    if len(g)<520:return [],[]
    g["date"]=pd.to_datetime(g.date)
    dates=g.date.to_numpy()
    raw=g.close.to_numpy(float)
    c=g.adjusted_close.to_numpy(float)
    ratio=np.divide(c,raw,out=np.ones(len(raw)),where=raw>0)
    o=g.open.to_numpy(float)*ratio
    hi=g.high.to_numpy(float)*ratio
    lo=g.low.to_numpy(float)*ratio
    vol=g.volume.to_numpy(float)
    amount=g.amount.to_numpy(float)
    ma112=pd.Series(c).ewm(span=112,adjust=False,min_periods=112).mean().to_numpy()
    ma224=pd.Series(c).ewm(span=224,adjust=False,min_periods=224).mean().to_numpy()
    ma448=pd.Series(c).ewm(span=448,adjust=False,min_periods=448).mean().to_numpy()
    v20=pd.Series(vol).rolling(20).mean().to_numpy()
    amt20=pd.Series(amount).rolling(20).mean().to_numpy()
    prev20hi=pd.Series(hi).shift(1).rolling(20).max().to_numpy()
    prior80=((pd.Series(c)<pd.Series(ma224)).shift(1).rolling(80).sum()).to_numpy()
    events=[]
    cov=[]
    for i in range(520,len(g)):
        dt=g.at[i,"date"]
        if dt<START or dt>END:continue
        cov.append(dict(date=str(dt.date()),code=str(g.at[i,"code"]).zfill(6),
                        valid_history=bool(len(g.iloc[:i])>=520)))
        if not all(np.isfinite(q) and q>0 for q in (o[i],c[i],hi[i],lo[i],v20[i],ma112[i],ma224[i],ma448[i],amt20[i],prev20hi[i])):continue
        if amt20[i]<MIN_AMOUNT:continue
        ret=(c[i]/c[i-1]-1)
        body=(c[i]/o[i]-1)
        volratio=(vol[i]/v20[i])
        cross=bool(c[i]>ma224[i] and c[i-1]<=ma224[i-1] and volratio>=1.25 and c[i]/ma224[i]-1<=.11)
        event=bool(ret>=.05 and body>=.035 and volratio>=2 and c[i]>prev20hi[i])
        if not (cross or event):continue
        base_list=[base_metrics(c,hi,i,d) for d in LENS]
        valid=[b for b in base_list if b and b["passes"]]
        longest=max(valid,key=lambda b:b["days"]) if valid else None
        if longest and longest["days"]>=90 and np.isfinite(prior80[i]) and prior80[i]>=48:
            tier="LONG_BASE_90"
        elif longest and longest["days"]>=60:
            tier="BASE_60"
        elif longest:
            tier="EARLY_BASE_40"
        else:
            tier="UNCONFIRMED_BASE"
        kinds=[]
        if cross:kinds.append("EMA224_FIRST_CROSS")
        if event:kinds.append("VOLUME_20D_HIGH_BREAK")
        event_type="+".join(kinds)
        if tier=="UNCONFIRMED_BASE" and not event and volratio<1.8:
            continue
        base=longest or {}
        factor=ratio[i]
        low14=np.nanmin(lo[max(0,i-14):i+1])
        stop=np.nanmin([low14,ma224[i]])/factor if factor>0 else np.nan
        resistance=prev20hi[i]/factor if factor>0 else np.nan
        events.append(dict(date=str(dt.date()),code=str(g.at[i,"code"]).zfill(6),
                           name=str(g.at[i,"name"]),exchange=str(g.at[i,"exchange"]),
                           series_id=str(g.at[i,"series_id"]),event_type=event_type,tier=tier,
                           close=float(raw[i]),event_ret_pct=100*ret,body_pct=100*body,
                           volume_ratio=float(volratio),amount20_krw=float(amt20[i]),
                           dist224_pct=100*(c[i]/ma224[i]-1),
                           prior80_below224=int(prior80[i]) if np.isfinite(prior80[i]) else None,
                           base_days=int(base["days"]) if base else None,
                           base_start=str(g.at[base["base_start_i"],"date"].date()) if base else None,
                           base_range_pct=base.get("base_range_pct",np.nan),
                           base_net_pct=base.get("base_net_pct",np.nan),
                           prior_peak_decline_pct=base.get("previous_peak_decline_pct",np.nan),
                           ema112=float(ma112[i]/factor),
                           ema224=float(ma224[i]/factor),ema448=float(ma448[i]/factor),
                           reference_low14=float(low14/factor),
                           reference_resistance=float(resistance),
                           refreshed_from="full-market panel; may include Naver overlay",
                           action_status="RESEARCH_ONLY"))
    return events,cov

def plot_one(g, rec, folder):
    g=g.sort_values("date").reset_index(drop=True)
    idxs=g.index[g.date.eq(pd.Timestamp(rec["date"]))].tolist()
    if not idxs:return
    i=int(idxs[0])
    sub=g.iloc[max(0,i-180):i+1].copy().reset_index(drop=True)
    if len(sub)<100:return
    ratio=(sub.adjusted_close/sub.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.)
    fac=float(ratio.iloc[-1])
    for key in ("open","high","low","close"):
        sub["p_"+key]=sub[key]*ratio/fac
    # Compute EMA using the FULL episode, then align to selected bars.
    fullfac=(g.adjusted_close/g.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.)
    for n in (112,224,448):
        vals=g.adjusted_close.ewm(span=n,adjust=False,min_periods=n).mean()/fac
        sub[f"ema{n}"]=vals.iloc[max(0,i-180):i+1].to_numpy()
    fig,(ax,av)=plt.subplots(2,1,figsize=(16,9),gridspec_kw={"height_ratios":[4,1]},sharex=True)
    x=np.arange(len(sub))
    for k,r in sub.iterrows():
        up=r.p_close>=r.p_open
        color="#C9414D" if up else "#326AA5"
        ax.vlines(k,r.p_low,r.p_high,color=color,lw=.85)
        ax.add_patch(Rectangle((k-.31,min(r.p_close,r.p_open)),.62,max(abs(r.p_close-r.p_open),.1),
                               color=color,lw=.4))
        av.bar(k,r.volume/1e6,color=color,width=.75)
    ax.plot(x,sub.ema112,label="EMA112",lw=1.25,color="#229A88")
    ax.plot(x,sub.ema224,label="EMA224",lw=1.55,color="#895BC3")
    ax.plot(x,sub.ema448,label="EMA448",lw=1.25,color="#657485")
    if rec.get("base_start"):
        ix=sub.index[sub.date.eq(pd.Timestamp(rec["base_start"]))]
        if len(ix):ax.axvspan(int(ix[0]),len(sub)-1,alpha=.06,color="#559F75",label="Base study span")
    ax.scatter([len(sub)-1],[float(sub.p_close.iloc[-1])],s=180,marker="^",
               color="#E5AA27",edgecolor="#704A14",zorder=8,label="New breakout")
    for key,color,label in (("ema224","#895BC3","EMA224"),("reference_low14","#CC4A4B","Recent low"),
                            ("reference_resistance","#D99D24","Prior 20D high")):
        v=rec.get(key)
        if v is None or not np.isfinite(float(v)):continue
        ax.axhline(float(v),color=color,ls="--",lw=.8,alpha=.56)
    ax.grid(alpha=.15);av.grid(alpha=.1)
    ax.set_ylabel("Price (KRW)");av.set_ylabel("Volume (mil)")
    ax.legend(loc="upper left",fontsize=9,ncol=5)
    n=len(sub);ticks=np.arange(0,n,max(1,n//9))
    av.set_xticks(ticks)
    av.set_xticklabels([str(sub.date.iloc[t].date()) for t in ticks],rotation=22,ha="right")
    fig.suptitle(f'{rec["code"]} / {rec["name"]} | {rec["date"]} | {rec["tier"]} | {rec["event_type"]}',
                 fontsize=15,weight="bold")
    fig.text(.01,.01,
             f'Base {rec.get("base_days")} trading days | volume x{rec["volume_ratio"]:.2f} | '
             f'EMA224 gap {rec["dist224_pct"]:+.1f}% | Study only, not executed trades',
             fontsize=9)
    fig.tight_layout(rect=[0,.03,1,.95])
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{rec["date"]}_{rec["code"]}.png',dpi=130)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ko",type=Path,required=True)
    parser.add_argument("--kq",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    from dante_classic_scanner import load
    market=pd.concat([load(args.ko,"KO"),load(args.kq,"KQ")],ignore_index=True)
    args.out.mkdir(parents=True,exist_ok=True)
    latest=pd.Timestamp(market.date.max())
    print("PANEL_LATEST",latest.date(),flush=True)
    episodes=list(market.groupby("series_id",sort=False))
    all_rows=[];coverage=[]
    for i,(sid,g) in enumerate(episodes,1):
        rows,cv=scan_series(g)
        all_rows.extend(rows);coverage.extend(cv)
        if i%400==0:print("EPISODES",i,"NEW_SIGNALS",len(all_rows),flush=True)
    s=pd.DataFrame(all_rows)
    if s.empty:
        s=pd.DataFrame(columns=["date","code","name","tier","event_type"])
    else:
        s=s.sort_values(["date","tier","volume_ratio"],ascending=[True,True,False])
    s.to_csv(args.out/"daily_new_breakouts.csv",index=False,encoding="utf-8-sig")
    cov=pd.DataFrame(coverage)
    sums=cov.groupby("date").agg(n_series=("code","size"),unique_codes=("code","nunique")).reset_index() if not cov.empty else pd.DataFrame(columns=["date","n_series","unique_codes"])
    sums.to_csv(args.out/"daily_coverage.csv",index=False)
    c=s.groupby(["date","tier"]).size().unstack(fill_value=0).reset_index() if len(s) else pd.DataFrame()
    c.to_csv(args.out/"signal_counts.csv",index=False)
    (args.out/"META.txt").write_text(
        f"MAX_MARKET_BAR={latest.date()}\nEXPECTED_END={END.date()}\n"
        f"VALID_2026_11_13={'YES' if latest>=END else 'NO'}\n"
        "Screened: KOSPI/KOSDAQ listed and adequately seasoned episodes.\n"
        "These are research-only price-pattern signals, NOT proprietary Dante alerts.\n"
        "A candidate can have multiple independent signal dates in the week.\n"
        "No orders are sent; price provider coverage may still be incomplete.\n")
    if len(s):
        # Highest priority first and representative breadth of dates.
        chart_rows=s.copy()
        chart_rows["sortscore"]=chart_rows.tier.map({
            "LONG_BASE_90":0,"BASE_60":1,"EARLY_BASE_40":2,
            "UNCONFIRMED_BASE":3}).fillna(9)
        chart_rows=chart_rows.sort_values(["sortscore","volume_ratio"],ascending=[True,False])
        seen=set();selected=[]
        for row in chart_rows.to_dict("records"):
            if row["code"] in seen:continue
            selected.append(row);seen.add(row["code"])
            if len(selected)>=12:break
        groups={str(key):g for key,g in episodes}
        for rec in selected:
            try:plot_one(groups[str(rec["series_id"])],rec,args.out/"charts")
            except Exception as e:print("CHART_ERROR",rec["code"],repr(e),flush=True)
    print("=== DAILY COVERAGE ===\n",sums.to_string(index=False),flush=True)
    print("=== DAILY CANDIDATE COUNTS ===\n",c.to_string(index=False),flush=True)
    print("=== STRICT 60/90 CANDIDATES ===\n",
          s[s.tier.isin(["LONG_BASE_90","BASE_60"])].to_string(index=False),flush=True)

if __name__=="__main__":
    main()
