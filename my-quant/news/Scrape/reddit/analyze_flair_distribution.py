#!/usr/bin/env python
"""Analyze flair distributions in reddit_posts SQLite DB.

Outputs are written under <db_dir>/static/reddit by default:
- flair_summary.md
- flair_summary.json
- flair_distribution_raw.csv
- flair_distribution_basic.csv
- flair_distribution_canonical.csv
- flair_distribution_by_subreddit.csv
- charts/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from config import DB_PATH as DEFAULT_DB_PATH
except Exception:
    DEFAULT_DB_PATH = "data/reddit_posts.db"


def normalize_basic(flair: Optional[str]) -> str:
    """Trim flair and map null/empty values to (EMPTY)."""
    if flair is None:
        return "(EMPTY)"
    value = flair.strip()
    return value if value else "(EMPTY)"


def normalize_canonical(flair: Optional[str]) -> str:
    """Build a canonical flair key for fuzzy de-duplication."""
    basic = normalize_basic(flair)
    if basic == "(EMPTY)":
        return basic
    basic = re.sub(r"\s+", " ", basic)
    return basic.lower()


def sort_counter_items(counter: Counter[str]):
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def safe_filename(value: str) -> str:
    """Convert a subreddit/flair string to a filesystem-safe file name."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return safe or "unknown"


def shorten_label(value: str, max_len: int = 56) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def plot_flair_distribution(
    counter: Counter[str],
    title: str,
    output_path: Path,
    top_n: int,
) -> bool:
    """Render one horizontal bar chart for a flair counter."""
    top_n = max(1, top_n)
    items = sort_counter_items(counter)[:top_n]
    if not items:
        return False

    items = list(reversed(items))
    labels = [shorten_label(flair) for flair, _ in items]
    values = [count for _, count in items]

    fig_height = max(4.0, 0.45 * len(values) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(range(len(values)), values, color="#4C78A8")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Count")
    ax.set_title(title)

    max_value = max(values) if values else 0
    offset = max(1, int(max_value * 0.02))
    for bar, value in zip(bars, values):
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reddit flair distribution from SQLite DB")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to SQLite db file (default: config.DB_PATH or data/reddit_posts.db)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <db_dir>/static/reddit)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N flair items shown in summary/charts")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only export tables and summary, skip PNG charts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    out_dir = Path(args.out_dir) if args.out_dir else db_path.parent / "static" / "reddit"
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    table = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reddit_posts'"
    ).fetchone()
    if not table:
        conn.close()
        raise RuntimeError("Table 'reddit_posts' does not exist in the database.")

    rows = cur.execute("SELECT subreddit, flair, datetime FROM reddit_posts").fetchall()
    total = len(rows)

    raw_counter: Counter[str] = Counter()
    basic_counter: Counter[str] = Counter()
    canonical_counter: Counter[str] = Counter()
    subreddit_basic_counter: dict[str, Counter[str]] = {}

    datetimes = []
    for row in rows:
        subreddit = row["subreddit"]
        flair = row["flair"]
        dt = row["datetime"]

        raw_key = "(NULL)" if flair is None else flair
        basic_key = normalize_basic(flair)
        canonical_key = normalize_canonical(flair)

        raw_counter[raw_key] += 1
        basic_counter[basic_key] += 1
        canonical_counter[canonical_key] += 1

        if subreddit not in subreddit_basic_counter:
            subreddit_basic_counter[subreddit] = Counter()
        subreddit_basic_counter[subreddit][basic_key] += 1

        if dt:
            datetimes.append(dt)

    raw_sorted = sort_counter_items(raw_counter)
    basic_sorted = sort_counter_items(basic_counter)
    canonical_sorted = sort_counter_items(canonical_counter)

    nonempty_flair_posts = total - basic_counter.get("(EMPTY)", 0)
    empty_flair_posts = basic_counter.get("(EMPTY)", 0)

    with (out_dir / "flair_distribution_raw.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flair_raw", "count", "pct"])
        for flair, count in raw_sorted:
            pct = round(count * 100.0 / total, 4) if total else 0.0
            writer.writerow([flair, count, pct])

    with (out_dir / "flair_distribution_basic.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flair_basic", "count", "pct"])
        for flair, count in basic_sorted:
            pct = round(count * 100.0 / total, 4) if total else 0.0
            writer.writerow([flair, count, pct])

    with (out_dir / "flair_distribution_canonical.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["flair_canonical", "count", "pct"])
        for flair, count in canonical_sorted:
            pct = round(count * 100.0 / total, 4) if total else 0.0
            writer.writerow([flair, count, pct])

    with (out_dir / "flair_distribution_by_subreddit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subreddit", "flair_basic", "count", "pct_in_subreddit"])
        for subreddit in sorted(subreddit_basic_counter):
            sub_counter = subreddit_basic_counter[subreddit]
            sub_total = sum(sub_counter.values())
            for flair, count in sort_counter_items(sub_counter):
                pct = round(count * 100.0 / sub_total, 4) if sub_total else 0.0
                writer.writerow([subreddit, flair, count, pct])

    charts_dir = out_dir / "charts"
    chart_files: list[str] = []
    if not args.no_plots:
        charts_dir.mkdir(parents=True, exist_ok=True)

        overall_chart = charts_dir / "overall_top_flair.png"
        if plot_flair_distribution(
            basic_counter,
            f"All Subreddits Flair Distribution (Top {max(1, args.top_n)})",
            overall_chart,
            args.top_n,
        ):
            chart_files.append(str(overall_chart.relative_to(out_dir)))

        for idx, subreddit in enumerate(sorted(subreddit_basic_counter), start=1):
            filename = f"subreddit_{idx:02d}_{safe_filename(subreddit)}.png"
            output_path = charts_dir / filename
            if plot_flair_distribution(
                subreddit_basic_counter[subreddit],
                f"r/{subreddit} Flair Distribution (Top {max(1, args.top_n)})",
                output_path,
                args.top_n,
            ):
                chart_files.append(str(output_path.relative_to(out_dir)))

    top_n = max(1, args.top_n)
    top_basic = []
    for flair, count in basic_sorted[:top_n]:
        top_basic.append(
            {
                "flair": flair,
                "count": count,
                "pct": round(count * 100.0 / total, 4) if total else 0.0,
            }
        )

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db_path": str(db_path),
        "out_dir": str(out_dir),
        "total_posts": total,
        "nonempty_flair_posts": nonempty_flair_posts,
        "empty_flair_posts": empty_flair_posts,
        "distinct_flair_raw": len(raw_sorted),
        "distinct_flair_basic": len(basic_sorted),
        "distinct_flair_canonical": len(canonical_sorted),
        "date_min": min(datetimes) if datetimes else None,
        "date_max": max(datetimes) if datetimes else None,
        "top_basic": top_basic,
        "subreddit_totals": {
            subreddit: sum(counter.values()) for subreddit, counter in sorted(subreddit_basic_counter.items())
        },
        "chart_files": chart_files,
    }

    with (out_dir / "flair_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Reddit Flair Distribution Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Database: {summary['db_path']}",
        f"- Output dir: {summary['out_dir']}",
        f"- Total posts: {summary['total_posts']}",
        f"- Non-empty flair posts: {summary['nonempty_flair_posts']}",
        f"- Empty flair posts: {summary['empty_flair_posts']}",
        f"- Distinct flair (raw): {summary['distinct_flair_raw']}",
        f"- Distinct flair (basic trim): {summary['distinct_flair_basic']}",
        f"- Distinct flair (canonical): {summary['distinct_flair_canonical']}",
        f"- Date range: {summary['date_min']} ~ {summary['date_max']}",
        "",
        f"## Top {top_n} flair (basic trim)",
        "",
        "| Rank | Flair | Count | Pct |",
        "|---:|---|---:|---:|",
    ]
    for i, item in enumerate(top_basic, start=1):
        md_lines.append(f"| {i} | {item['flair']} | {item['count']} | {item['pct']}% |")

    if chart_files:
        md_lines.extend([
            "",
            "## Generated charts",
            "",
        ])
        for chart_file in chart_files:
            md_lines.append(f"- {chart_file}")

    (out_dir / "flair_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    conn.close()

    print(f"Done. Output directory: {out_dir}")
    print("Generated files:")
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            print(f"- {path.name}")
    if chart_files:
        print(f"- charts/ ({len(chart_files)} PNG files)")


if __name__ == "__main__":
    main()
