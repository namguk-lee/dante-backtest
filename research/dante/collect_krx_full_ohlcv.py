#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable KOSPI/KOSDAQ + delisted KRX OHLCV collector.

The collector is deliberately separated from the backtester. It builds a
point-in-time-ish research universe from current KOSPI/KOSDAQ listings plus
KRX delisting episodes, caches each listing episode, and merges the cache into
KO/KQ parquet files.

FinanceDataReader documents KRX price data as adjusted-price data. We therefore
store adjusted_close=close for the FDR KRX feed. Delisted episodes are queried
through KRX-DELISTING and clipped to their listing/delisting dates so a reused
stock code does not silently bridge separate listing episodes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import time

import pandas as pd
import FinanceDataReader as fdr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01-01", help="Data warm-up start")
    p.add_argument("--end", default=None)
    p.add_argument("--out", type=Path, default=Path("krx_data"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--delay", type=float, default=0.20)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-failure-rate", type=float, default=0.15)
    return p.parse_args()


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_listing(df: pd.DataFrame, market_hint: str | None, active: bool) -> pd.DataFrame:
    x = df.copy()
    symbol_col = _first_existing(x, ["Symbol", "Code"])
    if symbol_col is None:
        raise ValueError(f"Listing lacks Symbol/Code: {list(x.columns)}")
    x["code"] = x[symbol_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)

    market_col = _first_existing(x, ["Market", "Exchange"])
    if market_col:
        x["exchange"] = x[market_col].astype(str).str.upper()
    else:
        x["exchange"] = market_hint
    x["exchange"] = x["exchange"].replace({"KOSPI": "KO", "KOSDAQ": "KQ"})
    x = x[x["exchange"].isin(["KO", "KQ"])].copy()

    name_col = _first_existing(x, ["Name"])
    list_col = _first_existing(x, ["ListingDate", "Listing Date"])
    delist_col = _first_existing(x, ["DelistingDate", "Delisting Date"])
    x["name"] = x[name_col].astype(str) if name_col else ""
    x["listing_date"] = pd.to_datetime(x[list_col], errors="coerce") if list_col else pd.NaT
    x["delisting_date"] = pd.to_datetime(x[delist_col], errors="coerce") if delist_col else pd.NaT
    x["active"] = bool(active)
    x["source"] = "ACTIVE" if active else "DELISTED"

    def series_id(row: pd.Series) -> str:
        ld = row["listing_date"]
        d = ld.strftime("%Y%m%d") if pd.notna(ld) else "UNKNOWN"
        return f"{row['exchange']}_{row['code']}_{d}_{row['source'][0]}"

    x["series_id"] = x.apply(series_id, axis=1)
    return x[["series_id", "code", "exchange", "name", "active", "source", "listing_date", "delisting_date"]]


def _bounded_dates(rec: dict, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    qs = start
    qe = end
    ld = rec.get("listing_date")
    dd = rec.get("delisting_date")
    if pd.notna(ld):
        qs = max(qs, pd.Timestamp(ld))
    if pd.notna(dd):
        qe = min(qe, pd.Timestamp(dd))
    return qs, qe


def _cache_complete(path: Path, qs: pd.Timestamp, qe: pd.Timestamp) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    try:
        q = pd.read_parquet(path, columns=["date"])
        if q.empty:
            return False, 0
        d = pd.to_datetime(q["date"], errors="coerce").dropna()
        if d.empty:
            return False, 0
        start_ok = d.min() <= qs + pd.Timedelta(days=10)
        end_ok = d.max() >= qe - pd.Timedelta(days=10)
        return bool(start_ok and end_ok), len(q)
    except Exception:
        return False, 0


def fetch_one(rec: dict, start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path,
              retries: int, delay: float, force: bool = False) -> dict:
    code = rec["code"]
    ex = rec["exchange"]
    active = bool(rec["active"])
    series_id = rec["series_id"]
    p = cache_dir / ex / f"{series_id}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    qs, qe = _bounded_dates(rec, start, end)

    if qs > qe:
        return {"series_id": series_id, "code": code, "exchange": ex, "rows": 0,
                "cached": False, "error": "outside_window"}

    if not force:
        complete, n = _cache_complete(p, qs, qe)
        if complete:
            return {"series_id": series_id, "code": code, "exchange": ex, "rows": n,
                    "cached": True, "error": None}

    last_error = None
    for attempt in range(retries):
        try:
            if active:
                q = fdr.DataReader(code, qs.strftime("%Y-%m-%d"), qe.strftime("%Y-%m-%d"))
            else:
                q = fdr.DataReader(f"KRX-DELISTING:{code}", qs.strftime("%Y-%m-%d"), qe.strftime("%Y-%m-%d"))
            if q is None or q.empty:
                return {"series_id": series_id, "code": code, "exchange": ex, "rows": 0,
                        "cached": False, "error": "empty"}

            q = q.reset_index()
            ren = {}
            for canonical, aliases in {
                "date": ["Date", "index"], "open": ["Open"], "high": ["High"],
                "low": ["Low"], "close": ["Close"], "volume": ["Volume"], "amount": ["Amount"],
            }.items():
                src = _first_existing(q, aliases)
                if src:
                    ren[src] = canonical
            q = q.rename(columns=ren)
            needed = ["date", "open", "high", "low", "close", "volume"]
            missing = [c for c in needed if c not in q.columns]
            if missing:
                raise ValueError(f"missing {missing}; columns={list(q.columns)}")

            keep = needed + (["amount"] if "amount" in q.columns else [])
            q = q[keep].copy()
            q["date"] = pd.to_datetime(q["date"], errors="coerce")
            for c in ["open", "high", "low", "close", "volume"] + (["amount"] if "amount" in q.columns else []):
                q[c] = pd.to_numeric(q[c], errors="coerce")
            q = q.dropna(subset=needed)
            q = q[(q["date"] >= qs) & (q["date"] <= qe)].sort_values("date").drop_duplicates("date")
            if q.empty:
                return {"series_id": series_id, "code": code, "exchange": ex, "rows": 0,
                        "cached": False, "error": "empty_after_episode_clip"}

            q["adjusted_close"] = q["close"]
            q["series_id"] = series_id
            q["code"] = code
            q["exchange"] = ex
            q["name"] = rec.get("name", "")
            q["active"] = active
            q["listing_date"] = rec.get("listing_date")
            q["delisting_date"] = rec.get("delisting_date")
            q.to_parquet(p, index=False)
            time.sleep(delay + random.random() * delay)
            return {"series_id": series_id, "code": code, "exchange": ex, "rows": len(q),
                    "cached": False, "error": None}
        except Exception as exc:
            last_error = repr(exc)
            time.sleep((attempt + 1) * 1.25 + random.random())

    return {"series_id": series_id, "code": code, "exchange": ex, "rows": 0,
            "cached": False, "error": last_error}


def _listing_has_symbol(df: pd.DataFrame | None) -> bool:
    return (
        isinstance(df, pd.DataFrame)
        and not df.empty
        and _first_existing(df, ["Symbol", "Code"]) is not None
    )


def load_universe(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ko_raw = fdr.StockListing("KOSPI")
    kq_raw = fdr.StockListing("KOSDAQ")
    print("KOSPI listing:", getattr(ko_raw, "shape", None), list(getattr(ko_raw, "columns", [])), flush=True)
    print("KOSDAQ listing:", getattr(kq_raw, "shape", None), list(getattr(kq_raw, "columns", [])), flush=True)
    ko = normalize_listing(ko_raw, "KO", True)
    kq = normalize_listing(kq_raw, "KQ", True)

    raw_de = None
    try:
        raw_de = fdr.StockListing(
            "KRX-DELISTING",
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        print(
            "KRX-DELISTING ranged:",
            getattr(raw_de, "shape", None),
            list(getattr(raw_de, "columns", [])),
            flush=True,
        )
    except Exception as exc:
        print("KRX-DELISTING ranged failed:", repr(exc), flush=True)

    if not _listing_has_symbol(raw_de):
        print("KRX-DELISTING ranged unavailable; retrying full listing", flush=True)
        try:
            raw_de = fdr.StockListing("KRX-DELISTING")
            print(
                "KRX-DELISTING full:",
                getattr(raw_de, "shape", None),
                list(getattr(raw_de, "columns", [])),
                flush=True,
            )
        except Exception as exc:
            print("KRX-DELISTING full failed:", repr(exc), flush=True)
            raw_de = None

    if not _listing_has_symbol(raw_de):
        raise RuntimeError(
            "KRX-DELISTING universe unavailable from FinanceDataReader. "
            "Refusing active-only backtest because it would introduce survivorship bias."
        )

    de = normalize_listing(raw_de, None, False)
    de = de[de["delisting_date"].isna() | (de["delisting_date"] >= start)].copy()

    u = pd.concat([ko, kq, de], ignore_index=True)
    u = u.drop_duplicates(["series_id"], keep="last").sort_values(["exchange", "code", "listing_date", "active"])
    return u.reset_index(drop=True)


def merge_exchange(cache: Path, exchange: str, out_path: Path) -> tuple[int, int]:
    frames = []
    for p in sorted((cache / exchange).glob("*.parquet")):
        try:
            q = pd.read_parquet(p)
            if not q.empty:
                frames.append(q)
        except Exception as exc:
            print("skip_cache", p, repr(exc), flush=True)
    if not frames:
        return 0, 0
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["series_id", "date"]).drop_duplicates(["series_id", "date"], keep="last")
    out.to_parquet(out_path, index=False)
    return len(out), out["series_id"].nunique()


def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "symbols"
    cache.mkdir(exist_ok=True)
    start = pd.Timestamp(a.start)
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize()

    universe = load_universe(start, end)
    universe.to_csv(a.out / "universe.csv", index=False, encoding="utf-8-sig")
    print(
        f"universe episodes={len(universe):,} KO={(universe.exchange == 'KO').sum():,} "
        f"KQ={(universe.exchange == 'KQ').sum():,} active={universe.active.sum():,}",
        flush=True,
    )

    logs: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = {
            pool.submit(fetch_one, rec, start, end, cache, a.retries, a.delay, a.force): rec
            for rec in universe.to_dict("records")
        }
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                logs.append(fut.result())
            except Exception as exc:
                rec = futures[fut]
                logs.append({"series_id": rec["series_id"], "code": rec["code"], "exchange": rec["exchange"],
                             "rows": 0, "cached": False, "error": repr(exc)})
            if i % 50 == 0 or i == total:
                ok = sum(x["rows"] > 0 for x in logs)
                print(f"[{i}/{total}] ok={ok} failed_or_empty={i-ok}", flush=True)

    log = pd.DataFrame(logs)
    log.to_csv(a.out / "collection_log.csv", index=False, encoding="utf-8-sig")

    for ex, name in [("KO", "ko_eod.parquet"), ("KQ", "kq_eod.parquet")]:
        rows, series = merge_exchange(cache, ex, a.out / name)
        print(name, f"rows={rows:,}", f"series={series:,}", flush=True)

    attempted = int((log["error"] != "outside_window").sum()) if not log.empty else 0
    failed = int(((log["rows"] <= 0) & (log["error"] != "outside_window")).sum()) if not log.empty else 0
    failure_rate = failed / attempted if attempted else 1.0
    print(f"collection failure_rate={failure_rate:.2%} ({failed}/{attempted})", flush=True)
    if failure_rate > a.max_failure_rate:
        raise SystemExit(f"Collection failure rate {failure_rate:.2%} exceeds {a.max_failure_rate:.2%}")


if __name__ == "__main__":
    main()
