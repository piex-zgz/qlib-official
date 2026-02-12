# AGENT Memory (Long-term Notes)

Updated: 2026-02-12
Scope: D:\Quant-qlib-official

## Project Focus
- Build News Agent data-prep tools under `my-quant/news`.
- Current source focus: Reddit SQLite (`data/reddit_posts.db`).

## Stable Conventions
- Use controller config first: `my-quant/news/news_tool_controller.json`.
- Build payload via: `my-quant/news/tools/build_reddit_payload.py`.
- Keep output path stable for downstream: `data/static/reddit/reddit_news_payload.json`.

## Reddit Payload Rules (Current)
- Flair filter: fuzzy contains `new`/`news`, case-insensitive.
- `subreddit`: extract exact subreddit from URL/permalink (`/r/<name>/...`).
- `route_subreddit`: keep original grouped subreddit for routing split.
- Grouping for first-stage dispatch uses `route_subreddit`.

## Important Paths
- Plan file: `my-quant/news/AGENT_PLAN.md`
- Flair analysis script: `my-quant/news/Scrape/reddit/analyze_flair_distribution.py`
- Payload builder: `my-quant/news/tools/build_reddit_payload.py`

## Next TODO
1. Add incremental export mode (since last run).
2. Add route-level whitelist/blacklist and per-route logic mapping.
3. Add tests for config parsing + SQL filter + payload schema.
