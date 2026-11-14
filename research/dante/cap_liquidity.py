#!/usr/bin/env python3
"""Point-in-time market-cap relative liquidity diagnostics for KOSPI/KOSDAQ.

Rules here are RESEARCH heuristic thresholds; NOT Dante published constants.
Capitalization is TOTAL listed capitalization, NOT free-float capitalization.
Data from FinanceData/marcap can encode 'Marcap' in won or millions of won:
determine scale by checking (raw close × listed shares), never hardcode units.

Live source overlays may lack cap and shares; use at most five recent prior
listed-share observations to estimate capitalization and explicitly label it.
No future rows/forward-looking cap, no order execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Thresholds deliberately simple and separate from technical signal detection.
# A prospective validation is required before these become investment criteria.
CAP_TIERS = (
    ("SMALL_LT_100B", 100_000_000_000., 100_000_000., 0.0010),
    ("MID_100B_TO_1T", 1_000_000_000_000., 300_000_000., 0.0004),
    ("LARGE_GE_1T", np.inf, 1_000_000_000., 0.00015),
)
CAP_MATCH_TOLERANCE=0.15


def _num(s, index):
    if s is None:
        return pd.Series(np.nan,index=index,dtype=float)
    return pd.to_numeric(s,errors="coerce").astype(float)


def normalize_daily_capitalization(frame):
    """Return independently checkable daily KRW market cap with source labels.

    Raw marcap units accepted ONLY when consistent with price*share count.
    Mark share-count-only estimates separately, not "verified source" data.
    Works for missing marcap on Naver recent-price overlays without lookahead.
    """
    x=frame.copy()
    idx=x.index
    raw=_num(x.get("marcap"),idx)
    close=_num(x.get("close"),idx)
    rawshares=_num(x.get("listed_shares"),idx)
    # Naver appends a handful of days without share/cap observations. No stale
    # perpetual share estimate, and never borrow shares from a future record.
    shares=rawshares.ffill(limit=5).where(lambda v:v.gt(0))
    implied=(close*shares).where(close.gt(0)&shares.gt(0))
    raw=raw.where(raw.gt(0))
    rel=lambda a,b: ((a/b)-1).abs()
    in_won=raw.notna()&implied.notna()&rel(raw,implied).le(CAP_MATCH_TOLERANCE)
    in_million=(~in_won)&raw.notna()&implied.notna()&rel(raw*1_000_000.,implied).le(CAP_MATCH_TOLERANCE)
    # Price × reported or last RECENT listed shares is a transparent estimate.
    fallback=(~in_won)&(~in_million)&implied.notna()
    cap=pd.Series(np.nan,index=idx,dtype=float)
    cap.loc[in_won]=raw.loc[in_won]
    cap.loc[in_million]=raw.loc[in_million]*1_000_000.
    cap.loc[fallback]=implied.loc[fallback]
    quality=pd.Series("CAP_MISSING",index=idx,dtype=object)
    quality.loc[in_won]="KRX_MARCAP_WON_CHECKED"
    quality.loc[in_million]="MARCAP_MILLION_CONVERTED_CHECKED"
    quality.loc[fallback & rawshares.notna()]="CLOSE_X_LISTED_SHARES_ESTIMATE"
    quality.loc[fallback & rawshares.isna()]="LAST_REPORTED_SHARES_ESTIMATE"
    quality.loc[raw.notna()&implied.notna()&(~in_won)&(~in_million)]="MARCAP_MISMATCH_ESTIMATE"
    x["market_cap_won"]=cap
    x["market_cap_quality"]=quality
    x["listed_shares_pti"]=shares
    x["cap_reported_conflict"]=(raw.notna()&implied.notna()&(~in_won)&(~in_million))
    return x


def market_cap_tier(cap):
    if not np.isfinite(cap) or cap<=0:return "CAP_UNKNOWN"
    for name,upper,absolute,ratio in CAP_TIERS:
        if cap<upper:return name
    raise AssertionError("unexpected cap")


def attach_liquidity_features(group):
    """Chronological, single-issuer-episode DataFrame; no future data."""
    z=normalize_daily_capitalization(group.reset_index(drop=True))
    amt=_num(z.get("amount"),z.index)
    vol=_num(z.get("volume"),z.index)
    # Previous 20 bars only; the new breakout candle cannot inflate its own
    # historical baseline or alter its prior-day reference cap.
    z["pre20_amount_won"]=amt.shift(1).rolling(20,min_periods=20).mean()
    z["pre20_amount_median_won"]=amt.shift(1).rolling(20,min_periods=20).median()
    z["pre20_volume"]=vol.shift(1).rolling(20,min_periods=20).mean()
    z["cap_before_event_won"]=z.market_cap_won.shift(1)
    cap=z.cap_before_event_won.where(z.cap_before_event_won.gt(0))
    z["daily_turnover_to_cap_pct"]=100*amt/z.market_cap_won.where(z.market_cap_won.gt(0))
    z["relative_turnover_pre20_pct"]=100*z.pre20_amount_won/cap
    z["listed_share_volume_turnover_pct"]=100*vol/z.listed_shares_pti.where(z.listed_shares_pti.gt(0))
    z["amount_spike_vs_prior20"]=amt/z.pre20_amount_won.where(z.pre20_amount_won.gt(0))
    xcap=z.cap_before_event_won.to_numpy(dtype=float)
    tier=np.full(len(z),"CAP_UNKNOWN",dtype=object)
    req_abs=np.full(len(z),np.nan,dtype=float)
    req_rate=np.full(len(z),np.nan,dtype=float)
    lower=0.
    for name,upper,absolute,rate in CAP_TIERS:
        ok=np.isfinite(xcap)&(xcap>=lower)&(xcap<upper)
        tier[ok]=name
        req_abs[ok]=absolute
        req_rate[ok]=100*rate
        lower=upper
    z["cap_tier"]=tier
    z["required_pre20_amount_won"]=req_abs
    z["required_pre20_cap_ratio_pct"]=req_rate
    z["adaptive_liquidity_pass"]=(
        z.pre20_amount_won.ge(z.required_pre20_amount_won)&
        z.relative_turnover_pre20_pct.ge(z.required_pre20_cap_ratio_pct)
    )
    z["legacy_50e_pass"]=amt.rolling(20,min_periods=20).mean().ge(5_000_000_000.)
    z["legacy_previous20_50e_pass"]=z.pre20_amount_won.ge(5_000_000_000.)
    z["liquidity_warning"]=np.select(
        [z.cap_tier.eq("CAP_UNKNOWN"),z.pre20_amount_won.lt(100_000_000.),
         ~z.adaptive_liquidity_pass],
        ["MISSING_CAP","VERY_THIN_BELOW_100M","BELOW_CAP_TIER_BENCHMARK"],
        default="RESEARCH_BENCHMARK_PASS"
    )
    return z


def is_liquid(r, mode="legacy", legacy_min_amount=5_000_000_000.):
    if mode=="legacy":
        # Preserve the old production logic, which includes today's amount.
        return bool(pd.notna(r.get("amt20")) and r["amt20"]>=legacy_min_amount)
    if mode=="cap-relative":
        return bool(r.get("adaptive_liquidity_pass",False))
    if mode=="observe":
        # no hard amount exclusion; mark thin liquidity as research-only.
        return bool(pd.notna(r.get("pre20_amount_won")) and r["pre20_amount_won"]>0)
    raise ValueError("Unknown liquidity mode: "+str(mode))


def describe_liquidity(r):
    return {
        "cap_tier":r.get("cap_tier","CAP_UNKNOWN"),
        "market_cap_won":r.get("market_cap_won",np.nan),
        "cap_before_event_won":r.get("cap_before_event_won",np.nan),
        "market_cap_quality":r.get("market_cap_quality","CAP_MISSING"),
        "pre20_amount_won":r.get("pre20_amount_won",np.nan),
        "pre20_amount_median_won":r.get("pre20_amount_median_won",np.nan),
        "relative_turnover_pre20_pct":r.get("relative_turnover_pre20_pct",np.nan),
        "daily_turnover_to_cap_pct":r.get("daily_turnover_to_cap_pct",np.nan),
        "listed_share_volume_turnover_pct":r.get("listed_share_volume_turnover_pct",np.nan),
        "amount_spike_vs_prior20":r.get("amount_spike_vs_prior20",np.nan),
        "liquidity_warning":r.get("liquidity_warning","MISSING_CAP"),
        "adaptive_liquidity_pass":r.get("adaptive_liquidity_pass",False),
        "legacy_50e_pass":r.get("legacy_50e_pass",False),
        "listed_shares_pti":r.get("listed_shares_pti",np.nan),
    }
