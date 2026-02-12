#!/usr/bin/env python
"""Build filtered Reddit payload JSON for downstream agents."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "news_tool_controller.json"
SUBREDDIT_PATTERN = re.compile(r"/r/([^/]+)/", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered Reddit JSON payload from SQLite")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to controller JSON (default: my-quant/news/news_tool_controller.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run query and show stats without writing output")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Controller file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if "reddit_payload" not in data:
        raise KeyError("Missing 'reddit_payload' section in controller file")
    return data["reddit_payload"]


def resolve_path(raw_path: str, config_dir: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (config_dir / p).resolve()


def build_order_clause(order_by: str) -> str:
    if order_by == "datetime_asc":
        return "datetime ASC"
    if order_by == "score_desc":
        return "score DESC, datetime DESC"
    return "datetime DESC"


def get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def extract_subreddit_from_text(value: Any) -> str | None:
    if value is None:
        return None
    match = SUBREDDIT_PATTERN.search(str(value))
    if not match:
        return None
    return match.group(1)


def extract_exact_subreddit(url: Any, permalink: Any, fallback: str) -> str:
    return (
        extract_subreddit_from_text(url)
        or extract_subreddit_from_text(permalink)
        or fallback
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)

    if not cfg.get("enabled", True):
        print("reddit_payload.enabled is false; nothing to do")
        return

    db_path = resolve_path(cfg["db_path"], config_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    output_path = resolve_path(cfg["output_json_path"], config_dir)
    source_table = cfg.get("source_table", "reddit_posts")
    include_fields = cfg.get("include_fields") or [
        "post_id",
        "datetime",
        "subreddit",
        "title",
        "content",
        "flair",
        "author",
        "score",
        "num_comments",
        "url",
    ]

    keywords = [str(x) for x in (cfg.get("flair_keywords") or []) if str(x).strip()]
    keywords = [x.strip() for x in keywords]
    case_sensitive = bool(cfg.get("case_sensitive", False))
    match_mode = str(cfg.get("match_mode", "contains_any"))
    exclude_empty_flair = bool(cfg.get("exclude_empty_flair", True))

    subreddits = [str(x).strip() for x in (cfg.get("subreddits") or []) if str(x).strip()]
    start_datetime = cfg.get("start_datetime")
    end_datetime = cfg.get("end_datetime")
    min_score = cfg.get("min_score")
    limit_total = cfg.get("limit_total")
    limit_per_subreddit = cfg.get("limit_per_subreddit")
    order_by = build_order_clause(str(cfg.get("order_by", "datetime_desc")))
    pack_by_subreddit = bool(cfg.get("pack_by_subreddit", True))

    subreddit_from_url = bool(cfg.get("subreddit_from_url", True))
    route_by_original_subreddit = bool(cfg.get("route_by_original_subreddit", True))
    route_field_name = str(cfg.get("route_field_name", "route_subreddit"))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    table_columns = get_table_columns(conn, source_table)
    missing_fields = [f for f in include_fields if f not in table_columns]
    if missing_fields:
        raise ValueError(f"Fields not in table {source_table}: {missing_fields}")

    query_fields = list(include_fields)
    internal_required_fields: list[str] = []
    if route_by_original_subreddit:
        internal_required_fields.append("subreddit")
    if subreddit_from_url:
        internal_required_fields.extend(["subreddit", "url", "permalink"])

    for field in internal_required_fields:
        if field not in table_columns:
            raise ValueError(f"Field '{field}' required by routing mode but not found in table {source_table}")
        if field not in query_fields:
            query_fields.append(field)

    conditions: list[str] = []
    params: list[Any] = []

    if exclude_empty_flair:
        conditions.append("TRIM(COALESCE(flair, '')) <> ''")

    if keywords:
        keyword_conditions: list[str] = []
        for kw in keywords:
            if match_mode == "exact_any":
                if case_sensitive:
                    keyword_conditions.append("COALESCE(flair, '') = ?")
                    params.append(kw)
                else:
                    keyword_conditions.append("LOWER(COALESCE(flair, '')) = ?")
                    params.append(kw.lower())
            else:
                if case_sensitive:
                    keyword_conditions.append("COALESCE(flair, '') LIKE ?")
                    params.append(f"%{kw}%")
                else:
                    keyword_conditions.append("LOWER(COALESCE(flair, '')) LIKE ?")
                    params.append(f"%{kw.lower()}%")
        conditions.append("(" + " OR ".join(keyword_conditions) + ")")

    if subreddits:
        placeholders = ",".join(["?"] * len(subreddits))
        conditions.append(f"subreddit IN ({placeholders})")
        params.extend(subreddits)

    if start_datetime:
        conditions.append("datetime >= ?")
        params.append(start_datetime)

    if end_datetime:
        conditions.append("datetime <= ?")
        params.append(end_datetime)

    if min_score is not None:
        conditions.append("score >= ?")
        params.append(min_score)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    select_fields = ", ".join(query_fields)
    sql = f"SELECT {select_fields} FROM {source_table} WHERE {where_clause} ORDER BY {order_by}"

    if isinstance(limit_total, int) and limit_total > 0:
        sql += " LIMIT ?"
        params.append(limit_total)

    rows = conn.execute(sql, params).fetchall()
    matched_before_group_limit = len(rows)

    records: list[dict[str, Any]] = []
    for row in rows:
        source_record = dict(row)
        record = {k: source_record.get(k) for k in include_fields}

        original_subreddit = str(source_record.get("subreddit") or "unknown")
        exact_subreddit = original_subreddit
        if subreddit_from_url:
            exact_subreddit = extract_exact_subreddit(
                source_record.get("url"),
                source_record.get("permalink"),
                original_subreddit,
            )
        record["subreddit"] = exact_subreddit
        record["subreddit_exact"] = exact_subreddit

        route_value = original_subreddit if route_by_original_subreddit else exact_subreddit
        record[route_field_name] = route_value

        records.append(record)

    if isinstance(limit_per_subreddit, int) and limit_per_subreddit > 0:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            route_key = str(rec.get(route_field_name, "unknown"))
            if len(grouped[route_key]) < limit_per_subreddit:
                grouped[route_key].append(rec)
        final_records = [item for route_key in sorted(grouped) for item in grouped[route_key]]
    else:
        final_records = records

    counts_by_route_subreddit = Counter(str(r.get(route_field_name, "unknown")) for r in final_records)
    counts_by_exact_subreddit = Counter(str(r.get("subreddit", "unknown")) for r in final_records)
    counts_by_flair = Counter(str(r.get("flair", "")) for r in final_records)

    total_rows_in_db = conn.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
    conn.close()

    payload: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "controller_path": str(config_path),
        "db_path": str(db_path),
        "filters": {
            "flair_keywords": keywords,
            "case_sensitive": case_sensitive,
            "match_mode": match_mode,
            "exclude_empty_flair": exclude_empty_flair,
            "subreddits": subreddits,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "min_score": min_score,
            "limit_total": limit_total,
            "limit_per_subreddit": limit_per_subreddit,
            "order_by": order_by,
            "subreddit_from_url": subreddit_from_url,
            "route_by_original_subreddit": route_by_original_subreddit,
            "route_field_name": route_field_name,
        },
        "stats": {
            "total_rows_in_db": total_rows_in_db,
            "matched_before_group_limit": matched_before_group_limit,
            "final_records": len(final_records),
            "route_subreddit_counts": dict(sorted(counts_by_route_subreddit.items())),
            "subreddit_exact_counts": dict(sorted(counts_by_exact_subreddit.items())),
            "top_flair": [
                {"flair": flair, "count": count}
                for flair, count in counts_by_flair.most_common(20)
            ],
        },
    }

    if pack_by_subreddit:
        by_route_subreddit: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in final_records:
            by_route_subreddit[str(rec.get(route_field_name, "unknown"))].append(rec)
        sorted_grouped = dict(sorted(by_route_subreddit.items()))
        payload["items_by_subreddit"] = sorted_grouped
        payload["items_by_route_subreddit"] = sorted_grouped

    payload["items"] = final_records

    print(f"Matched rows: {matched_before_group_limit}")
    print(f"Final rows: {len(final_records)}")
    print(f"Route groups: {len(counts_by_route_subreddit)}")
    print(f"Exact subreddit count: {len(counts_by_exact_subreddit)}")

    if args.dry_run:
        print("Dry-run enabled. Output file not written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = bool(cfg.get("output_pretty", True))
    if pretty:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"Payload written to: {output_path}")


if __name__ == "__main__":
    main()
