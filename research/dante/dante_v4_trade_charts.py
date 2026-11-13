#!/usr/bin/env python3
"""Annotated candlestick charts for a small, audited V4 research signal sample.

No orders are placed or implied. Markers are hypothetical backtest actions,
not discretionary buy/sell advice. Marcap-adjusted OHLC is translated into
the latest unadjusted price denomination for visual comparability.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager

TARGETS = {
    "328130": "루닛",
    "282720": "금양그린파워",
    "411080": "샌즈랩",
    "030960": "양지사",
    "326030": "SK바이오팜",
}
COLOR_UP = "#D6575B"   # red up candle, conventional KR market
COLOR_DOWN = "#3974B9" # blue down candle, conventional KR market
COLOR_BUY = "#007F57"
COLOR_SELL = "#AF3137"
COLOR_WATCH = "#C47F14"
EMA_COLORS = {"ema112":"#43A08C", "ema224":"#9668C6", "ema448":"#677487"}


def set_korean_font():
    font_paths = font_manager.findSystemFonts(fontext="ttf") + font_manager.findSystemFonts(fontext="otf")
    font_name = None
    for p in font_paths:
        if any(tag in p.lower() for tag in ("nanumgothic", "nanumsquare", "notosanscjk")):
            try:
                font_name = font_manager.FontProperties(fname=p).get_name()
                break
            except Exception:
                continue
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


def prepare(g):
    x = g.sort_values("date").drop_duplicates("date", keep="last").copy().reset_index(drop=True)
    for c in ("open","high","low","close","adjusted_close","volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    factor = (x.adjusted_close / x.close).replace([np.inf,-np.inf],np.nan).ffill().bfill()
    last_factor = float(factor.iloc[-1])
    for c in ("open","high","low","close"):
        x[c+"_vis"] = x[c]*factor/last_factor
    for n in (112,224,448):
        x[f"ema{n}"] = x.adjusted_close.ewm(span=n, adjust=False, min_periods=n).mean()/last_factor
    return x, last_factor


def simulate_record(x, record, last_factor):
    dt = pd.Timestamp(record.date)
    ids = x.index[x.date.eq(dt)].tolist()
    if not ids:
        return dict(error=f"signal date missing from panel: {record.date}")
    si = int(ids[0])
    ei = si + 1
    if ei >= len(x):
        return dict(error="entry day not available")
    recorded_entry = float(record.entry_open_adjusted)/last_factor
    recorded_stop = float(record.stop_adjusted)/last_factor
    chart_entry = float(x.at[ei,"open_vis"])
    entry_gap_percent = 100*(chart_entry/recorded_entry-1) if recorded_entry>0 else np.nan
    # Favor the original V4 recorded adjusted next-open and stop, rather than
    # recomputing entries from an unrelated current-day price.
    entry=recorded_entry
    stop=recorded_stop
    risk=entry-stop
    if not np.isfinite(risk) or risk<=0:
        return dict(error="invalid entry/stop")
    target=entry+2*risk
    limit = min(len(x)-1,si+20)
    exit_i = None
    outcome = "OPEN_ASOF"
    price = np.nan
    for j in range(ei, limit+1):
        o,h,l,c = [float(x.at[j,col]) for col in ("open_vis","high_vis","low_vis","close_vis")]
        if not all(np.isfinite(z) and z>0 for z in (o,h,l,c)):
            return dict(error=f"OHLC missing on {x.at[j,'date']}")
        if o<=stop:
            exit_i=j;price=o;outcome="STOP_GAP";break
        if o>=target:
            exit_i=j;price=o;outcome="TARGET_GAP";break
        if l<=stop:
            exit_i=j;price=stop;outcome="STOP";break
        if h>=target:
            exit_i=j;price=target;outcome="TARGET";break
    if exit_i is None and len(x)-1>=si+20:
        exit_i=si+20;price=float(x.at[exit_i,"close_vis"]);outcome="TIMEOUT_20D"
    net_return=(100*(price/entry-1)-.4) if exit_i is not None else np.nan
    # A guard against changed data vintages. The downloaded sample's original
    # completed outcomes are retained verbatim for the user to cross-check.
    orig_outcome = str(record.outcome_20d) if pd.notna(record.outcome_20d) else ""
    orig_held = float(record.held_20d) if pd.notna(record.held_20d) else np.nan
    mismatch = bool(orig_outcome and orig_outcome != outcome)
    if orig_outcome and np.isfinite(orig_held) and exit_i is not None:
        mismatch |= (int(orig_held) != int(exit_i-si))
    return dict(signal_i=si,entry_i=ei,exit_i=exit_i,
                entry_date=str(x.at[ei,"date"].date()),
                exit_date=str(x.at[exit_i,"date"].date()) if exit_i is not None else None,
                entry=entry, stop=stop, target=target, exit_price=price,
                outcome=outcome, net_return=net_return,
                original_outcome=orig_outcome, original_return=float(record.net_20d) if pd.notna(record.net_20d) else np.nan,
                original_held=orig_held, source_price_deviation_pct=entry_gap_percent,
                mismatch=mismatch)


def plot_chart(x, events, ticker, name, output, recent=False):
    limit = x.date.max()
    # Full 2026 history supports older completed signals. Recent view supports
    # verification of latest accumulation / pullback structure.
    lowdt = pd.Timestamp("2026-07-01" if recent else "2026-01-01")
    ix=x.index[x.date.ge(lowdt)]
    if len(ix)<20:
        ix=x.index[-min(len(x),240):]
    start=int(ix.min())
    t=x.iloc[start:].copy().reset_index(drop=True)
    n=len(t)
    p=np.arange(n)
    fig=plt.figure(figsize=(19.2,11.5))
    gs=fig.add_gridspec(2,1,height_ratios=[4.15,0.92],hspace=.04,
                        left=.055,right=.96,top=.87,bottom=.20)
    ax=fig.add_subplot(gs[0])
    av=fig.add_subplot(gs[1],sharex=ax)
    ax.set_facecolor("#FBFCFE")
    av.set_facecolor("#FBFCFE")
    for i,row in t.iterrows():
        o=float(row.open_vis);cl=float(row.close_vis)
        hi=float(row.high_vis);lo=float(row.low_vis)
        if not all(np.isfinite(a) and a>0 for a in (o,cl,hi,lo)):
            continue
        color=COLOR_UP if cl>=o else COLOR_DOWN
        ax.vlines(i,lo,hi,color=color,linewidth=.85,zorder=2)
        lo_b=min(o,cl);height=max(abs(o-cl),max(hi,lo)*.0005)
        ax.add_patch(Rectangle((i-.32,lo_b),.64,height,
                               facecolor=color,edgecolor=color,linewidth=.5,zorder=3))
        av.bar(i,float(row.volume)/1e6,color=color,width=.7,alpha=.68)
    for col in ("ema112","ema224","ema448"):
        if t[col].notna().any():
            ax.plot(p,t[col].values,color=EMA_COLORS[col],label=col.upper(),
                    lw=1.65 if col=="ema224" else 1.15,alpha=.92,zorder=4)
    actual_x=len(t)-1
    actual_last=float(t.close_vis.iloc[-1])
    ax.scatter([actual_x],[actual_last],s=150,marker="*",color=COLOR_WATCH,
               edgecolor="white",lw=1.3,zorder=12,label="11/12 관찰 위치")
    ax.annotate(f"관찰 {actual_last:,.0f}",(actual_x,actual_last),
                xytext=(-10,-25),textcoords="offset points",fontsize=9,
                color=COLOR_WATCH,ha="right",fontweight="bold")
    trade_notes=[]
    count=0
    marker_margin=[]
    for record,ev in events:
        if "error" in ev:
            trade_notes.append(f"{record.date}: 데이터 누락 ({ev['error']})")
            continue
        count+=1
        sig_i=ev["signal_i"]-start
        buy_i=ev["entry_i"]-start
        sell_i=ev["exit_i"]-start if ev["exit_i"] is not None else None
        if buy_i < 0:
            continue
        tag=str(count)
        if 0<=sig_i<n:
            ax.scatter([sig_i],[t.at[sig_i,"close_vis"]],marker="o",s=80,
                       color="#E8AB3D",edgecolors="#714500",linewidth=.8,zorder=8)
        if 0<=buy_i<n:
            ax.scatter([buy_i],[ev["entry"]],marker="^",s=170,
                       color=COLOR_BUY,edgecolors="white",linewidth=1,zorder=14)
            ax.annotate(f"매수{tag}\n{ev['entry']:,.0f}",(buy_i,ev["entry"]),
                        xytext=(-10,-42 if count%2 else -56),textcoords="offset points",
                        fontsize=9,color=COLOR_BUY,ha="right",fontweight="bold",
                        arrowprops=dict(arrowstyle="-",color=COLOR_BUY,lw=.75))
            ax.hlines(ev["stop"],max(0,buy_i-3),min(n-1,buy_i+17),
                      color=COLOR_SELL,alpha=.4,linestyle=":",linewidth=1.2)
        if sell_i is not None and 0<=sell_i<n:
            ax.scatter([sell_i],[ev["exit_price"]],marker="v",s=170,
                       color=COLOR_SELL,edgecolors="white",linewidth=1,zorder=14)
            ax.annotate(f"매도{tag}\n{ev['exit_price']:,.0f}",(sell_i,ev["exit_price"]),
                        xytext=(12,38 if count%2 else 52),textcoords="offset points",
                        fontsize=9,color=COLOR_SELL,ha="left",fontweight="bold",
                        arrowprops=dict(arrowstyle="-",color=COLOR_SELL,lw=.75))
        why=({"STOP":"손절","STOP_GAP":"갭 손절","TARGET":"2R 익절",
              "TARGET_GAP":"갭 익절","TIMEOUT_20D":"20일 종료",
              "OPEN_ASOF":"미청산(11/12 기준)"}).get(ev["outcome"],ev["outcome"])
        desc=(f"#{count} {record.date} {record.setup} | {ev['entry_date']} 진입 {ev['entry']:,.0f} "
              f"| 손절 {ev['stop']:,.0f} / 목표 {ev['target']:,.0f} | {why}")
        if ev["exit_date"]:
            desc+=f" {ev['exit_date']} {ev['exit_price']:,.0f} ({ev['net_return']:+.2f}%)"
        if ev["mismatch"]:
            desc+=" [원본 재검증 필요]"
        trade_notes.append(desc)
    ax.grid(alpha=.14)
    av.grid(alpha=.12)
    ax.set_xlim(-2,n+5)
    ax.set_ylabel("현재 환산 가격(원)")
    av.set_ylabel("거래량(백만주)")
    av.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val,_:f"{val:,.0f}"))
    av.yaxis.set_major_formatter(plt.FuncFormatter(lambda val,_:f"{val:,.1f}"))
    av.tick_params(labelsize=9)
    ax.tick_params(labelsize=9)
    plt.setp(ax.get_xticklabels(),visible=False)
    ticks=list(range(0,n,max(1,n//11)))
    if ticks[-1]!=n-1:
        ticks.append(n-1)
    av.set_xticks(ticks)
    av.set_xticklabels([str(t.date.iloc[i].date()) for i in ticks],
                       rotation=24,ha="right",fontsize=9)
    from matplotlib.lines import Line2D
    handles=[
        Line2D([0],[0],color=EMA_COLORS[c],lw=2,label=c.upper())
        for c in ("ema112","ema224","ema448")
    ]+[
        Line2D([0],[0],marker="^",color="white",markerfacecolor=COLOR_BUY,
               markersize=10,label="모의매수"),
        Line2D([0],[0],marker="v",color="white",markerfacecolor=COLOR_SELL,
               markersize=10,label="모의매도"),
        Line2D([0],[0],marker="*",color="white",markerfacecolor=COLOR_WATCH,
               markersize=12,label="관찰 종가"),
    ]
    ax.legend(handles=handles,loc="upper right",ncol=6,fontsize=10,frameon=True,
              framealpha=.94)
    fig.text(.055,.963,f"{name} ({ticker})   |   2026 캔들 / 모의 매수·매도 위치",
             fontsize=21,weight="bold",color="#172C42")
    fig.text(.055,.927,f"단테 V4 백테스트 이벤트 | 실주문 없음 | 2026-11-12 종가까지 | "
             f"{'최근 확대' if recent else '2026년 전체'} | 왕복비용 0.4%p",
             fontsize=11,color="#596E82")
    fig.text(.055,.155,"매매 위치 요약 (각 #은 독립 신호 시뮬레이션이며 동시에 매매한 포트폴리오가 아님)",
             fontsize=10.5,weight="bold",color="#243D54")
    if trade_notes:
        for i,note in enumerate(trade_notes[:5]):
            fig.text(.06,.128-i*.022,note,fontsize=9.1,color="#33495F")
    else:
        fig.text(.06,.128,"이 기간에는 V4의 독립 모의매매 진입 신호 없음.",
                 fontsize=10,color="#596E82")
    fig.text(.055,.025,
             "출처: GitHub dante-backtest V4 신호 + FinanceData/marcap 가격 · 수정 OHLC 현재가격 환산 · "
             "도식화는 20거래일 손절/2R 목표 우선 규칙",
             fontsize=9,color="#657487")
    fig.savefig(output,dpi=145,facecolor="white")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ko",type=Path,required=True)
    p.add_argument("--kq",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    set_korean_font()
    from dante_classic_scanner import load
    panel=pd.concat([load(a.ko,"KO"),load(a.kq,"KQ")],ignore_index=True)
    panel=panel[panel.code.isin(TARGETS)].copy()
    events=pd.read_csv(a.manifest,dtype={"code":str},encoding="utf-8-sig")
    a.out.mkdir(exist_ok=True,parents=True)
    logs=[]
    for code,name in TARGETS.items():
        group=panel[panel.code.eq(code)].copy()
        if group.empty:
            raise ValueError(f"missing OHLCV for {code}")
        # For any code reuse, select the episode with latest available trading day.
        last=group.date.max()
        episode=group[group.date.eq(last)].series_id.iloc[-1]
        x,factor=prepare(group[group.series_id.eq(episode)])
        x=x[x.date.le(pd.Timestamp("2026-11-12"))].reset_index(drop=True)
        if x.date.max()<pd.Timestamp("2026-11-10"):
            raise ValueError(f"stale OHLCV for {code}: {x.date.max()}")
        rs=events[events.code.eq(code)].sort_values("date")
        applied=[]
        for rr in rs.itertuples(index=False):
            v=simulate_record(x,rr,factor)
            applied.append((rr,v))
            row=dict(code=code,name=name,signal_date=rr.date,
                     setup=rr.setup,profile=rr.profile,
                     original_outcome=rr.outcome_20d if pd.notna(rr.outcome_20d) else "",
                     original_net_20d=rr.net_20d if pd.notna(rr.net_20d) else "")
            row.update({k:v.get(k) for k in ("entry_date","exit_date","entry","stop","target",
                                            "exit_price","outcome","net_return",
                                            "mismatch","source_price_deviation_pct","error")})
            logs.append(row)
        print(code,name,"last",str(x.date.max().date()),"events",len(applied),flush=True)
        for recent,tag in ((False,"2026"),(True,"recent")):
            plot_chart(x,applied,code,name,a.out/f"{code}_{tag}_candles.png",recent=recent)
    df=pd.DataFrame(logs)
    df.to_csv(a.out/"simulated_positions.csv",index=False,encoding="utf-8-sig")
    if df.mismatch.fillna(False).any():
        print("WARNING: at least one historical trade differs from original V4; inspect as data-vintage variance",flush=True)
    if (df.source_price_deviation_pct.abs()>1.0).any():
        print("WARNING: some entry prices differ from original V4 by >1%; inspect adjustment",flush=True)
    print(df[["code","signal_date","setup","entry_date","exit_date","outcome","net_return","mismatch"]].to_string(index=False),flush=True)
    (a.out/"NOTICE.txt").write_text(
        "Charts show research simulations, NOT actual orders or actionable recommendations.\n"
        "All current classic watchlist entries are RESEARCH_ONLY.\n"
        "Signals refer to separate V4 research setups and independent tests, not a combined trading portfolio.\n"
        "Entry: next trading day adjusted open. Exit: within 20 trading days, stop-first if same bar hit both,\n"
        "or 2R target, else time exit. Last bars 2026-11-12. Trading fees 0.4%p round trip; no slippage.\n"
        "If neither stop nor target reached and <20 sessions have elapsed, status OPEN_ASOF, NOT a recommendation to buy.\n",
        encoding="utf-8")


if __name__=="__main__":
    main()
