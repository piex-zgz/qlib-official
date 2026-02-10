# -*- coding: utf-8 -*-
"""
财经快讯爬虫主入口

支持命令行参数调用
"""

import argparse
import sys
import json
from datetime import date, datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def cmd_fetch(args):
    """抓取命令"""
    from database import NewsDatabase
    from api_scraper import CLSScraper
    from akshare_scraper import AKShareScraper

    db = NewsDatabase()

    sources = []
    if args.cls:
        sources.append("cls")
    if args.em:
        sources.append("em")

    if not sources:
        sources = ["cls", "em"]

    total = 0
    for source in sources:
        try:
            if source == "cls":
                scraper = CLSScraper()
            else:
                scraper = AKShareScraper()

            items = scraper.fetch(limit=args.limit)
            if items:
                count = db.add_news(items, source=source)
                print(f"[{scraper.DISPLAY_NAME}] 获取 {len(items)} 条，新增 {count} 条")
                total += count
            else:
                print(f"[{source}] 未获取到数据")

        except Exception as e:
            print(f"[{source}] 失败: {e}")

    print(f"\n共新增 {total} 条快讯")


def cmd_increment(args):
    """增量获取命令"""
    from database import NewsDatabase
    from api_scraper import CLSScraper
    from akshare_scraper import AKShareScraper

    db = NewsDatabase()

    sources = []
    if args.cls:
        sources.append("cls")
    if args.em:
        sources.append("em")

    if not sources:
        sources = ["cls", "em"]

    total = 0
    for source in sources:
        try:
            if source == "cls":
                scraper = CLSScraper()
            else:
                scraper = AKShareScraper()

            items = scraper.fetch_incremental(db=db)
            if items:
                count = db.add_news(items, source=source)
                print(f"[{scraper.DISPLAY_NAME}] 新增 {count} 条")
                total += count
            else:
                print(f"[{source}] 无新数据")

        except Exception as e:
            print(f"[{source}] 失败: {e}")

    print(f"\n共新增 {total} 条快讯")


def cmd_query(args):
    """查询命令"""
    from database import NewsDatabase

    db = NewsDatabase()

    items = db.query(
        start_date=args.start,
        end_date=args.end,
        source=args.source,
        category=args.category,
        keyword=args.keyword,
        limit=args.limit,
    )

    print(f"\n查询结果: {len(items)} 条\n")
    for i, item in enumerate(items, 1):
        print(f"{i}. [{item['datetime']}] [{item['source']}][{item['category']}]")
        print(f"   {item['content'][:100]}...")
        print()


def cmd_status(args):
    """状态命令"""
    from database import NewsDatabase

    db = NewsDatabase()
    stats = db.get_stats()

    print("\n" + "=" * 50)
    print("财经快讯数据库状态")
    print("=" * 50)
    print(f"总计快讯数: {stats['total']}")
    print(f"  - 财联社 (cls): {stats['by_source'].get('cls', 0)} 条")
    print(f"  - 东方财富 (em): {stats['by_source'].get('em', 0)} 条")
    print(f"\n最近 10 天:")
    for row in stats['by_date'][:10]:
        print(f"  {row['date']}: {row['count']} 条")
    print(f"\n最近抓取记录:")
    for row in stats['recent_fetches']:
        status_icon = "✓" if row['status'] == "success" else "✗"
        print(f"  [{status_icon}] {row['source']} - {row['fetch_time'][:19]} - {row['count']} 条")
    print()


def cmd_export(args):
    """导出命令"""
    from database import NewsDatabase

    db = NewsDatabase()

    output_path = db.export(
        start_date=args.start,
        end_date=args.end,
        source=args.source,
        format=args.format,
    )

    print(f"\n已导出到: {output_path}")


def cmd_monitor(args):
    """监控命令（后台运行）"""
    from scheduler import SchedulerApp

    interval = args.interval
    sources = []

    if args.cls:
        sources.append("cls")
    if args.em:
        sources.append("em")

    if not sources:
        sources = ["cls", "em"]

    app = SchedulerApp(interval_minutes=interval, sources=sources)
    app.start(interval_minutes=interval)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="财经快讯采集系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 抓取最新快讯（所有数据源）
  python main.py fetch

  # 只抓取财联社
  python main.py fetch --cls

  # 只抓取东方财富
  python main.py fetch --em

  # 增量获取（去重，只获取新数据）
  python main.py increment

  # 查询最近快讯
  python main.py query --limit 20

  # 按关键词查询
  python main.py query --keyword 比特币

  # 按日期范围查询
  python main.py query --start 2026-02-01 --end 2026-02-05

  # 查看数据库状态
  python main.py status

  # 导出数据
  python main.py export --start 2026-02-01

  # 启动定时监控（每 5 分钟）
  python main.py monitor

  # 定时监控（每 5 分钟，只监控财联社）
  python main.py monitor --cls --interval 5
        """,
    )

    # 全局参数
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="数据库路径",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # fetch 命令
    fetch_parser = subparsers.add_parser("fetch", help="抓取最新快讯")
    fetch_parser.add_argument("--cls", action="store_true", help="只抓取财联社")
    fetch_parser.add_argument("--em", action="store_true", help="只抓取东方财富")
    fetch_parser.add_argument("--limit", type=int, default=100, help="最大数量")
    fetch_parser.set_defaults(func=cmd_fetch)

    # increment 命令
    inc_parser = subparsers.add_parser("increment", help="增量获取快讯（去重）")
    inc_parser.add_argument("--cls", action="store_true", help="只增量获取财联社")
    inc_parser.add_argument("--em", action="store_true", help="只增量获取东方财富")
    inc_parser.set_defaults(func=cmd_increment)

    # query 命令
    query_parser = subparsers.add_parser("query", help="查询快讯")
    query_parser.add_argument("--start", type=str, help="开始日期 (YYYY-MM-DD)")
    query_parser.add_argument("--end", type=str, help="结束日期 (YYYY-MM-DD)")
    query_parser.add_argument("--source", type=str, choices=["cls", "em"], help="数据源")
    query_parser.add_argument("--category", type=str, help="分类")
    query_parser.add_argument("--keyword", type=str, help="关键词")
    query_parser.add_argument("--limit", type=int, default=50, help="最大数量")
    query_parser.set_defaults(func=cmd_query)

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看状态")
    status_parser.set_defaults(func=cmd_status)

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出数据")
    export_parser.add_argument("--start", type=str, help="开始日期")
    export_parser.add_argument("--end", type=str, help="结束日期")
    export_parser.add_argument("--source", type=str, choices=["cls", "em"], help="数据源")
    export_parser.add_argument("--format", type=str, choices=["csv", "json"], default="csv", help="导出格式")
    export_parser.set_defaults(func=cmd_export)

    # monitor 命令
    monitor_parser = subparsers.add_parser("monitor", help="启动定时监控")
    monitor_parser.add_argument("--cls", action="store_true", help="只监控财联社")
    monitor_parser.add_argument("--em", action="store_true", help="只监控东方财富")
    monitor_parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="抓取间隔（分钟）",
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    # 解析参数
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 设置数据库路径
    if hasattr(args, 'db') and args.db:
        import os
        os.environ.setdefault("NEWS_DB_PATH", args.db)

    # 执行命令
    args.func(args)


if __name__ == "__main__":
    main()
