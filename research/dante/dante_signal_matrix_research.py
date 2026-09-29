#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research-only Dante signal matrix.

Purpose:
- keep publicly described structures separate from research proxies
- scan the current KOSPI/KOSDAQ market
- show signal confluence instead of forcing one serial A/B grade

This file DOES NOT reproduce proprietary indicators and DOES NOT emit buy calls.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dante_share_1to1_study import structural_up_crosses, contiguous_runs, prior_down_cross, safe_ratio

EMAS=(5,15,20,33,56,60,112,224,448)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("dante_signal_matrix"))
    p.add_argument("--min-turnover",type=float,default=5_000_000_000)
    p.add_argument("--max-charts",type=int,default=12)
    return p.parse_args()

def load(path,exchange):
    x=pd.read_parquet(path).copy()
    x["date"]=pd.to_datetime(x.date,errors="coerce")
    x["code"]=x.code.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
    x["name"]=x.get("name","").astype(str)
    if "exchange" not in x:x["exchange"]=exchange
    if "adjusted_close" not in x:x["adjusted_close"]=x.close
    if "amount" not in x:x["amount"]=x.close*x.volume
    return x.sort_values(["series_id","date"])

def features(g):
    z=g.sort_values("date").copy().reset_index(drop=True)
    f=(z.adjusted_close/z.close).replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(1.0)
    z["ao"]=z.open*f; z["ah"]=z.high*f; z["al"]=z.low*f; z["ac"]=z.adjusted_close
    for n in EMAS:
        z[f"ema{n}"]=z.ac.ewm(span=n,adjust=False,min_periods=n).mean()
    z["v20"]=z.volume.rolling(20,min_periods=20).mean()
    z["vr20"]=z.volume/z.v20
    z["amount20"]=z.amount.rolling(20,min_periods=20).mean()
    z["ret1"]=z.ac.pct_change(fill_method=None)
    z["body"]=(z.ac-z.ao)/z.ao
    for n in (112,224,448):
        z[f"ema{n}_slope20"]=(z[f"ema{n}"]/z[f"ema{n}"].shift(20)-1)*100
    mid=z.ac.rolling(35,min_periods=35).mean()
    sd=z.ac.rolling(35,min_periods=35).std(ddof=0)
    z["bb35_upper2"]=mid+2*sd
    return z

def last_cross_days(z,a,b,max_lookback=120):
    """Days since series a crossed above b."""
    cross=(z[a]>z[b])&(z[a].shift(1)<=z[b].shift(1))
    ids=np.flatnonzero(cross.tail(max_lookback).fillna(False).to_numpy())
    if not len(ids):return np.nan
    tail_start=max(0,len(z)-max_lookback)
    idx=tail_start+int(ids[-1])
    return len(z)-1-idx

def last_price_cross_days(z,ema_n,max_lookback=180):
    e=z[f"ema{ema_n}"]
    cross=(z.ac>=e)&(z.ac.shift(1)<e.shift(1))
    ids=np.flatnonzero(cross.tail(max_lookback).fillna(False).to_numpy())
    if not len(ids):return np.nan
    tail_start=max(0,len(z)-max_lookback)
    return len(z)-1-(tail_start+int(ids[-1]))

def share_proxy_for_ma(z,ma_n):
    s=z[["date","ac","ah","al"]].copy()
    s["ma"]=z[f"ema{ma_n}"]
    s["dist"]=(s.ac-s.ma)/s.ma
    s["above"]=s.ac>=s.ma
    crosses=structural_up_crosses(s,ma_n)
    candidates=[]
    for cross_i,_,_ in crosses:
        ps,pe,_,_=contiguous_runs(s,cross_i)
        pre_days=pe-ps+1
        if pre_days<30:continue
        pre=s.iloc[ps:pe+1]
        ma0=float(s.at[cross_i,"ma"])
        if not math.isfinite(ma0) or ma0<=0:continue
        down=(ma0-float(pre.al.min()))/ma0
        up=(float(s.iloc[-1].ac)-ma0)/ma0
        long_ratio=safe_ratio(up,down)
        dc=prior_down_cross(s,cross_i,ma_n)
        start=max(0,cross_i-max(60,ma_n)) if dc is None else dc
        seg=s.iloc[start:].copy(); seg=seg[pd.notna(seg.ma)]
        up_mean=float(seg.loc[seg.dist>0,"dist"].mean()) if (seg.dist>0).any() else np.nan
        dn_mean=float((-seg.loc[seg.dist<0,"dist"]).mean()) if (seg.dist<0).any() else np.nan
        cycle_mean=safe_ratio(up_mean,dn_mean)
        candidates.append({
            "ma":ma_n,"cross_i":cross_i,"cross_date":s.at[cross_i,"date"],
            "pre_days":pre_days,"long_ratio":long_ratio,"cycle_mean_ratio":cycle_mean
        })
    if not candidates:return None
    # Latest long-below structural recovery.
    c=sorted(candidates,key=lambda x:pd.Timestamp(x["cross_date"]))[-1]
    lr=c["long_ratio"]; cm=c["cycle_mean_ratio"]
    c["match"]=bool(
        s.iloc[-1].above and pd.notna(lr) and pd.notna(cm) and
        .65<=lr<=1.75 and .65<=cm<=1.35
    )
    if pd.notna(lr) and lr>0 and pd.notna(cm) and cm>0:
        c["balance_error"]=float(np.sqrt(np.log(lr)**2+np.log(cm)**2))
    else:c["balance_error"]=np.nan
    return c

def below224_run_before_recovery(z):
    """Consecutive sessions below EMA224 before the latest current recovery.

    Public Bowl-3 search material commonly refers to a long stay below EMA224.
    This is exposed as evidence/context only; it is not a universal entry rule.
    """
    if z.empty or pd.isna(z.iloc[-1].ema224):return np.nan
    i=len(z)-1
    if z.at[i,"ac"]<z.at[i,"ema224"]:
        j=i; n=0
        while j>=0 and pd.notna(z.at[j,"ema224"]) and z.at[j,"ac"]<z.at[j,"ema224"]:
            n+=1; j-=1
        return n
    cross=(z.ac>=z.ema224)&(z.ac.shift(1)<z.ema224.shift(1))
    ids=np.flatnonzero(cross.fillna(False).to_numpy())
    if not len(ids):return np.nan
    ci=int(ids[-1]); j=ci-1; n=0
    while j>=0 and pd.notna(z.at[j,"ema224"]) and z.at[j,"ac"]<z.at[j,"ema224"]:
        n+=1; j-=1
    return n

def bowl3_structure_proxy(z):
    """Transparent approximation of the publicly described Bowl 1->2->3 structure.

    Public concept used:
    - 1: meaningful decline
    - 2: longer base/accumulation than the decline
    - 3: recovery toward/through EMA224

    Exact segmentation is not public, so this remains a structure proxy.
    """
    if len(z)<520:return {
        "match":False,"stage":"","decline_days":np.nan,"base_days":np.nan,
        "decline_pct":np.nan,"trough_date":pd.NaT,"peak_date":pd.NaT
    }
    cur=z.iloc[-1]
    # Search the latest plausible trough in the last ~220 sessions, then the
    # preceding swing high that started the decline.
    start=max(0,len(z)-220)
    tail=z.iloc[start:].copy()
    trough_i=int(tail.ac.idxmin())
    if trough_i<=0:return {"match":False,"stage":"","decline_days":np.nan,"base_days":np.nan,"decline_pct":np.nan,"trough_date":pd.NaT,"peak_date":pd.NaT}
    peak_start=max(0,trough_i-160)
    pre=z.iloc[peak_start:trough_i+1]
    if len(pre)<15:return {"match":False,"stage":"","decline_days":np.nan,"base_days":np.nan,"decline_pct":np.nan,"trough_date":pd.NaT,"peak_date":pd.NaT}
    peak_i=int(pre.ac.idxmax())
    decline_days=trough_i-peak_i
    base_days=len(z)-1-trough_i
    peak=float(z.at[peak_i,"ac"]); trough=float(z.at[trough_i,"ac"])
    decline_pct=(trough/peak-1)*100 if peak>0 else np.nan
    if pd.isna(cur.ema224):return {"match":False,"stage":"","decline_days":decline_days,"base_days":base_days,"decline_pct":decline_pct,"trough_date":z.at[trough_i,"date"],"peak_date":z.at[peak_i,"date"]}
    dist224=(float(cur.ac)/float(cur.ema224)-1)*100
    # Bowl-3 is the 224 approach / *early* recovery zone, not a mature
    # stage-4 expansion far above the long MA.
    stage=("RECOVERED_224" if 0<=dist224<=15 else
           ("APPROACH_224" if -8<=dist224<0 else ""))
    match=bool(
        decline_days>=10 and pd.notna(decline_pct) and decline_pct<=-20 and
        base_days>=40 and base_days>=decline_days and
        float(cur.ac)>=float(cur.ema112) and
        pd.notna(cur.ema112_slope20) and float(cur.ema112_slope20)>0 and
        bool(stage)
    )
    return {
        "match":match,"stage":stage if match else "",
        "decline_days":decline_days,"base_days":base_days,"decline_pct":decline_pct,
        "trough_date":z.at[trough_i,"date"],"peak_date":z.at[peak_i,"date"]
    }

def candle_line_distance_pct(line,low,high):
    if pd.isna(line) or line<=0:return np.nan
    if low<=line<=high:return 0.0
    edge=low if line<low else high
    return abs(edge/line-1)*100

def concrete_proxy(z):
    """Transparent approximation: prior swing high breaks, then holds on retest."""
    if len(z)<90:return (False,np.nan,pd.NaT,pd.NaT)
    w=z.tail(90).copy()
    # Search resistance before the most recent 15 sessions.
    pre=w.iloc[:-15]
    if len(pre)<30:return (False,np.nan,pd.NaT,pd.NaT)
    piv=(pre.ah==pre.ah.rolling(7,center=True,min_periods=7).max())
    ph=pre[piv]
    if ph.empty:return (False,np.nan,pd.NaT,pd.NaT)
    # Prefer the latest meaningful pivot that was later broken.
    for idx,row in ph.sort_index(ascending=False).iterrows():
        level=float(row.ah)
        after=w.loc[idx+1:]
        br=after[after.ac>level*1.005]
        if br.empty:continue
        bi=int(br.index[0])
        post=w.loc[bi:]
        rt=post[(post.al<=level*1.03)&(post.ac>=level*.98)]
        if rt.empty:continue
        ri=int(rt.index[0])
        alive=bool(float(w.iloc[-1].ac)>=level*.98)
        if alive:return (True,level,w.loc[bi,"date"],w.loc[ri,"date"])
    return (False,np.nan,pd.NaT,pd.NaT)

def scan(z,min_turnover):
    if len(z)<520:return None
    cur=z.iloc[-1]
    if pd.isna(cur.ema448) or pd.isna(cur.amount20) or cur.amount20<min_turnover:return None

    # ---------- Publicly described structures ----------
    cross5_112=last_cross_days(z,"ema5","ema112",120)
    sig_256_long=bool(
        pd.notna(cross5_112) and cross5_112<=20 and
        cur.ema5>cur.ema112 and cur.ac>=cur.ema112 and cur.ac<cur.ema224
    )

    cross112=last_price_cross_days(z,112,180)
    cross224=last_price_cross_days(z,224,180)
    reverse_long=bool(cur.ema112<cur.ema224<cur.ema448)
    sig_ma_hit_112_224=bool(
        reverse_long and pd.notna(cross112) and cross112<=60 and
        cur.ac>=cur.ema112 and cur.ac<cur.ema224
    )
    sig_ma_hit_224_448=bool(
        pd.notna(cross224) and cross224<=60 and cur.ac>=cur.ema224 and cur.ac<cur.ema448
    )
    sig_reverse112=bool(reverse_long and cur.ac>=cur.ema112 and cur.ac<cur.ema224)
    bowl=bowl3_structure_proxy(z)
    sig_bowl3=bool(bowl.get("match"))
    below224_run=below224_run_before_recovery(z)
    bowl_near224_10=bool(pd.notna(cur.ema224) and abs((cur.ac/cur.ema224-1)*100)<=10)
    bowl_below224_80=bool(pd.notna(below224_run) and below224_run>=80)
    public_224_4month_near10=bool(bowl_near224_10 and bowl_below224_80)
    bowl_public_conditions=bool(sig_bowl3 and public_224_4month_near10)

    # ---------- Research proxies ----------
    share=[]
    for n in (60,112,224):
        x=share_proxy_for_ma(z,n)
        if x and x.get("match"):share.append(x)
    share_best=min(share,key=lambda x:x.get("balance_error",np.inf)) if share else None
    sig_share=bool(share_best)

    blue_dist=candle_line_distance_pct(float(cur.bb35_upper2),float(cur.al),float(cur.ah))
    recent_blue=[]
    for i in range(max(0,len(z)-5),len(z)):
        recent_blue.append(candle_line_distance_pct(float(z.at[i,"bb35_upper2"]) if pd.notna(z.at[i,"bb35_upper2"]) else np.nan,float(z.at[i,"al"]),float(z.at[i,"ah"])))
    blue_min5=float(np.nanmin(recent_blue)) if np.isfinite(recent_blue).any() else np.nan
    sig_blue=bool(pd.notna(blue_dist) and blue_dist<=1.0)
    sig_blue_recent5=bool(pd.notna(blue_min5) and blue_min5<=1.0)

    prior20=z.tail(20)
    max_vr=float(prior20.vr20.max()) if prior20.vr20.notna().any() else np.nan
    # Research proxy inspired by labeled public examples: accumulation burst first, quieter current session.
    sig_accum_cool=bool(pd.notna(max_vr) and max_vr>=2.0 and pd.notna(cur.vr20) and cur.vr20<=1.2)

    # Descriptive state for the direct public EMA224 search context.
    # This is intentionally a watchlist state, not an entry signal.
    if public_224_4month_near10:
        if pd.notna(cur.vr20) and float(cur.vr20)>=2.0:
            public224_watch_state="EVENT_SPIKE"
            public224_watch_score=0
        elif pd.notna(cur.ema112_slope20) and float(cur.ema112_slope20)>0 and sig_accum_cool:
            public224_watch_state="TURNING_COOLED"
            public224_watch_score=4
        elif pd.notna(cur.ema112_slope20) and float(cur.ema112_slope20)>0:
            public224_watch_state="TURNING"
            public224_watch_score=3
        elif sig_accum_cool:
            public224_watch_state="RAW_COOLED_CONTEXT"
            public224_watch_score=2
        else:
            public224_watch_state="RAW_CONTEXT"
            public224_watch_score=1
    else:
        public224_watch_state=""
        public224_watch_score=np.nan

    conc,conc_level,conc_break,conc_retest=concrete_proxy(z)

    technique_flags={
        "public_256_long":sig_256_long,
        "public_ma_hit_112_224":sig_ma_hit_112_224,
        "public_ma_hit_224_448":sig_ma_hit_224_448,
        "public_bowl3_structure_proxy":sig_bowl3,
    }
    context_flags={
        "public_reverse112_context":sig_reverse112,
        "public_224_4month_near10_context":public_224_4month_near10,
    }
    proxy_flags={
        "research_share_1to1_proxy":sig_share,
        "research_blue_dot_bb35_proxy":sig_blue,
        "research_blue_dot_bb35_recent5":sig_blue_recent5,
    }
    support_flags={
        "research_accumulation_cool_proxy":sig_accum_cool,
        "research_concrete_proxy":conc,
    }
    # Do not count correlated context/support proxies as independent public techniques.
    technique_count=sum(technique_flags.values())
    context_count=sum(context_flags.values())
    # Share proxy remains a ranking proxy. Blue-dot BB35 stays visible only
    # as broad chart context because historical validation was not robust.
    proxy_count=int(sig_share)
    # Concrete proxy is currently very broad; keep it visible but do not let it
    # inflate ranking/confluence until its selectivity is improved.
    support_count=int(sig_accum_cool)
    broad_context_count=int(conc)+int(sig_blue)
    total=technique_count+proxy_count+support_count
    # Keep direct public search-context matches even when no other detector fires.
    if total<1 and context_count<1:return None

    all_flags={**technique_flags,**context_flags,**proxy_flags,**support_flags}
    labels=[k.replace("public_","").replace("research_","").replace("_proxy","") for k,v in all_flags.items() if v and k!="research_blue_dot_bb35_recent5"]

    factor=float(cur.ac/cur.close) if float(cur.close)>0 else 1.0
    return {
        "date":cur.date,"code":str(cur.code).zfill(6),"name":cur.get("name",""),"exchange":cur.exchange,
        "current_close":float(cur.close),"avg_turnover20":float(cur.amount20),
        **technique_flags,**context_flags,**proxy_flags,**support_flags,
        "public_technique_count":technique_count,
        "public_context_count":context_count,
        "public_signal_count":technique_count,
        "research_proxy_count":proxy_count,
        "support_context_count":support_count,
        "broad_context_count":broad_context_count,
        "confluence_count":total,
        "signals":";".join(labels),
        "action_status":"WATCHLIST_RESEARCH_ONLY",
        "entry_validation":"NOT_VALIDATED",
        "reverse_112_224_448":reverse_long,
        "ema112_slope20_pct":float(cur.ema112_slope20) if pd.notna(cur.ema112_slope20) else np.nan,
        "ema224_slope20_pct":float(cur.ema224_slope20) if pd.notna(cur.ema224_slope20) else np.nan,
        "ema448_slope20_pct":float(cur.ema448_slope20) if pd.notna(cur.ema448_slope20) else np.nan,
        "long_ma_state":(
            ("112UP" if pd.notna(cur.ema112_slope20) and cur.ema112_slope20>0 else "112DOWN")+";"+
            ("224UP" if pd.notna(cur.ema224_slope20) and cur.ema224_slope20>0 else "224DOWN")+";"+
            ("448UP" if pd.notna(cur.ema448_slope20) and cur.ema448_slope20>0 else "448DOWN")
        ),
        "ma_turn_quality":(
            "TURNED_UP" if pd.notna(cur.ema112_slope20) and cur.ema112_slope20>0 else
            ("FLAT_TO_DOWN" if pd.notna(cur.ema112_slope20) and cur.ema112_slope20>-1 else "FALLING")
        ),
        "ma_turn_score":(
            2 if pd.notna(cur.ema112_slope20) and cur.ema112_slope20>0 else
            (1 if pd.notna(cur.ema112_slope20) and cur.ema112_slope20>-1 else 0)
        ),
        "days_since_5x112":cross5_112,"days_since_price_x112":cross112,"days_since_price_x224":cross224,
        "bowl3_stage":bowl.get("stage",""),"bowl1_decline_days":bowl.get("decline_days",np.nan),
        "bowl2_base_days":bowl.get("base_days",np.nan),"bowl1_decline_pct":bowl.get("decline_pct",np.nan),
        "bowl_peak_date":bowl.get("peak_date",pd.NaT),"bowl_trough_date":bowl.get("trough_date",pd.NaT),
        "below224_run_before_recovery":below224_run,
        "bowl_near224_10":bowl_near224_10,
        "bowl_below224_80":bowl_below224_80,
        "public_224_4month_near10":public_224_4month_near10,
        "public224_watch_state":public224_watch_state,
        "public224_watch_score":public224_watch_score,
        "public224_entry_evidence":(
            "SEARCH_DAY_ENTRY_BACKTEST_NEGATIVE" if public_224_4month_near10 else ""
        ),
        "bowl_public_conditions":bowl_public_conditions,
        "ema5":float(cur.ema5/factor),"ema112":float(cur.ema112/factor),"ema224":float(cur.ema224/factor),"ema448":float(cur.ema448/factor),
        "dist112_pct":float((cur.ac/cur.ema112-1)*100),"dist224_pct":float((cur.ac/cur.ema224-1)*100),"dist448_pct":float((cur.ac/cur.ema448-1)*100),
        "current_volume_ratio20":float(cur.vr20) if pd.notna(cur.vr20) else np.nan,
        "max_volume_ratio20_last20":max_vr,
        "blue_dot_bb35_dist_pct":blue_dist,"blue_dot_bb35_min5_dist_pct":blue_min5,
        "concrete_level":float(conc_level/factor) if pd.notna(conc_level) else np.nan,
        "concrete_break_date":conc_break,"concrete_retest_date":conc_retest,
        "share_ma":share_best["ma"] if share_best else np.nan,
        "share_cross_date":share_best["cross_date"] if share_best else pd.NaT,
        "share_pre_below_days":share_best["pre_days"] if share_best else np.nan,
        "share_recovery_ratio":share_best["long_ratio"] if share_best else np.nan,
        "share_cycle_mean_ratio":share_best["cycle_mean_ratio"] if share_best else np.nan,
        "share_balance_error":share_best["balance_error"] if share_best else np.nan,
    }

def render(z,r,out_dir):
    t=z.tail(280).copy().reset_index(drop=True)
    factor=float(t.iloc[-1].ac/t.iloc[-1].close) if float(t.iloc[-1].close)>0 else 1.0
    x=np.arange(len(t))
    fig,(ax,av)=plt.subplots(2,1,figsize=(15,9),sharex=True,gridspec_kw={"height_ratios":[4,1]})
    ax.plot(x,t.ac/factor,label="Close",linewidth=1.0)
    for n in (5,15,33,56,112,224,448):
        lw=1.25 if n in (112,224,448) else .75
        ax.plot(x,t[f"ema{n}"]/factor,label=f"EMA{n}",linewidth=lw,alpha=.9)
    ax.plot(x,t.bb35_upper2/factor,label="BB35x2 proxy",linewidth=.9,linestyle="--")
    if pd.notna(r.get("concrete_level")):
        ax.axhline(float(r["concrete_level"]),linestyle=":",linewidth=.9,label="Concrete proxy")
    av.bar(x,t.volume)
    av.plot(x,t.v20,linewidth=.8,label="V20")
    title=(f'{r["code"]} | techniques {int(r["public_technique_count"])} | '
           f'context {int(r["public_context_count"])} | proxy {int(r["research_proxy_count"])} | '
           f'support {int(r["support_context_count"])} | {r.get("ma_turn_quality","")}')
    ax.set_title(title)
    ax.text(.01,.98,r["signals"],transform=ax.transAxes,va="top",fontsize=8)
    ax.legend(ncol=5,fontsize=8); ax.grid(alpha=.15); av.grid(alpha=.15)
    step=max(1,len(t)//10); ticks=list(range(0,len(t),step))
    av.set_xticks(ticks); av.set_xticklabels([pd.Timestamp(t.at[i,"date"]).strftime("%y-%m-%d") for i in ticks],rotation=30)
    out_dir.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    p=out_dir/f'{r["code"]}_signal_matrix.png'
    fig.savefig(p,dpi=150); plt.close(fig)

def main():
    a=args(); a.out.mkdir(parents=True,exist_ok=True)
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    latest=panel.date.max(); rows=[]; series={}
    for sid,g in panel.groupby("series_id",sort=False):
        if g.date.max()!=latest:continue
        z=features(g)
        r=scan(z,a.min_turnover)
        if r:
            rows.append(r); series[r["code"]]=z
    out=pd.DataFrame(rows)
    if not out.empty:
        out["rank_group"]=np.select(
            [
                (out.public_technique_count>=2)&(out.ma_turn_score>=2),
                out.public_technique_count>=2,
                (out.public_technique_count==1)&(out.ma_turn_score>=2),
                out.public_technique_count==1
            ],
            ["PUBLIC_MULTI_TURNED","PUBLIC_MULTI_EARLY","PUBLIC_SINGLE_TURNED","PUBLIC_SINGLE_EARLY"],
            default="PROXY_ONLY"
        )
        out=out.sort_values(
            ["public_technique_count","ma_turn_score","research_proxy_count","support_context_count","public_context_count","share_balance_error","avg_turnover20"],
            ascending=[False,False,False,False,False,True,False],na_position="last"
        ).reset_index(drop=True)
        out["rank"]=np.arange(1,len(out)+1)
    out.to_csv(a.out/"signal_matrix_all.csv",index=False,encoding="utf-8-sig")
    focus=out[out.confluence_count>=2].copy() if not out.empty else out.copy()
    focus.to_csv(a.out/"signal_matrix_confluence2plus.csv",index=False,encoding="utf-8-sig")
    public_focus=out[out.public_technique_count>=1].copy() if not out.empty else out.copy()
    public_focus.to_csv(a.out/"signal_matrix_public_focus.csv",index=False,encoding="utf-8-sig")
    public_turning=out[(out.public_technique_count>=1)&(out.ma_turn_score>=2)].copy() if not out.empty else out.copy()
    public_turning.to_csv(a.out/"signal_matrix_public_turning.csv",index=False,encoding="utf-8-sig")
    proxy_only=out[out.public_technique_count.eq(0)].copy() if not out.empty else out.copy()
    proxy_only.to_csv(a.out/"signal_matrix_proxy_only.csv",index=False,encoding="utf-8-sig")
    if not out.empty:
        technique_boards={
            "256_long":out[out.public_256_long].copy(),
            "ma_hit_112_224":out[out.public_ma_hit_112_224].copy(),
            "ma_hit_224_448":out[out.public_ma_hit_224_448].copy(),
            "bowl3":out[out.public_bowl3_structure_proxy].copy(),
            "bowl3_public_conditions":out[out.bowl_public_conditions].copy(),
            "public_224_4month_near10":out[out.public_224_4month_near10].copy(),
        }
        for board_name,board in technique_boards.items():
            board=board.copy()
            if board_name=="public_224_4month_near10" and not board.empty:
                board["abs_dist224_pct"]=pd.to_numeric(board.dist224_pct,errors="coerce").abs()
                board=board.sort_values(
                    ["public224_watch_score","public_technique_count","research_proxy_count","support_context_count","abs_dist224_pct","avg_turnover20"],
                    ascending=[False,False,False,False,True,False],na_position="last"
                ).reset_index(drop=True)
            else:
                board=board.reset_index(drop=True)
            board["board_rank"]=np.arange(1,len(board)+1)
            board.to_csv(a.out/f"board_{board_name}.csv",index=False,encoding="utf-8-sig")
            if board_name=="public_224_4month_near10":
                for _,br in board.head(a.max_charts).iterrows():
                    if br.code in series:
                        render(series[br.code],br,a.out/"charts_public_224_4month")
    if not out.empty:
        summary=[]
        signal_cols=[
            "public_256_long","public_ma_hit_112_224","public_ma_hit_224_448","public_bowl3_structure_proxy","public_reverse112_context",
            "research_share_1to1_proxy","research_blue_dot_bb35_proxy",
            "research_accumulation_cool_proxy","research_concrete_proxy"
        ]
        for col in signal_cols:summary.append({"signal":col,"count":int(out[col].sum())})
        for n,cnt in out.confluence_count.value_counts().sort_index().items():
            summary.append({"signal":f"confluence_{int(n)}","count":int(cnt)})
        pd.DataFrame(summary).to_csv(a.out/"signal_matrix_summary.csv",index=False,encoding="utf-8-sig")
        chart_source=public_focus if not public_focus.empty else out
        for _,r in chart_source.head(a.max_charts).iterrows():
            render(series[r.code],r,a.out/"charts")
    print(f"latest={latest.date()} matrix_any={len(out)} confluence2plus={len(focus)} public_focus={len(public_focus) if not out.empty else 0}")
    if not out.empty:
        cols=["rank","rank_group","code","name","exchange","current_close","public_technique_count","public_context_count","research_proxy_count","support_context_count","broad_context_count","confluence_count","ma_turn_quality","signals",
              "share_ma","share_recovery_ratio","share_cycle_mean_ratio","blue_dot_bb35_dist_pct",
              "max_volume_ratio20_last20","current_volume_ratio20","dist112_pct","dist224_pct","dist448_pct",
              "below224_run_before_recovery","public_224_4month_near10","public224_watch_state",
              "public224_entry_evidence","bowl_public_conditions","long_ma_state","entry_validation"]
        print(out[cols].head(40).to_string(index=False))

if __name__=="__main__":
    main()
