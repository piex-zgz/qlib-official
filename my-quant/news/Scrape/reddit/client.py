#!/usr/bin/env python
"""
Reddit News Scraper - 命令行工具
"""

import sys
import logging
import time
from datetime import datetime

from config import (
    DEFAULT_SUBREDDITS,
    LOG_LEVEL,
    DB_PATH,
    DEFAULT_LIMIT,
    INCREMENT_HOURS,
    SUBREDDIT_INTERVAL
)
from database import RedditDatabase
from scraper import RedditScraper
from scheduler import RedditScheduler, run_scheduler

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Reddit')


def fetch(limit=DEFAULT_LIMIT, subreddits=None):
    """抓取最新帖子"""
    subreddits = subreddits or DEFAULT_SUBREDDITS
    print(f"正在抓取 r/{subreddits} ...")
    scraper = RedditScraper(subreddits)
    db = RedditDatabase(DB_PATH)

    # 判断是否多个板块
    sr_list = scraper.list_subreddits()
    if len(sr_list) > 1:
        # 多板块：循环抓取，每个板块间隔 SUBREDDIT_INTERVAL 秒
        print(f"检测到 {len(sr_list)} 个板块，将循环抓取...")
        posts = scraper.fetch_all(limit=limit, interval_seconds=SUBREDDIT_INTERVAL)
    else:
        posts = scraper.fetch(limit=limit)

    if not posts:
        print("没有获取到帖子")
        return
    count = db.add_posts(posts)
    print(f"抓取完成: 获取 {len(posts)} 条，新增 {count} 条")
    print("\n最新帖子:")
    for i, post in enumerate(posts[:5], 1):
        print(f"{i}. [{post['subreddit']}] {post['title'][:60]}...")
        print(f"   {post['datetime']} | {post['url']}")


def increment(hours=None, subreddits=None):
    """增量获取新帖子"""
    if hours is None:
        hours = INCREMENT_HOURS
    subreddits = subreddits or DEFAULT_SUBREDDITS
    print(f"正在增量获取 r/{subreddits} (过去 {hours} 小时)...")
    scraper = RedditScraper(subreddits)
    db = RedditDatabase(DB_PATH)

    # 判断是否多个板块
    sr_list = scraper.list_subreddits()
    if len(sr_list) > 1:
        # 多板块：循环增量抓取
        print(f"检测到 {len(sr_list)} 个板块，将循环增量抓取...")
        posts = []
        for sr in sr_list:
            print(f"[Reddit] 增量抓取 r/{sr}...")
            sr_scraper = RedditScraper(sr)
            sr_posts = sr_scraper.fetch_incremental(db, hours=hours)
            if sr_posts:
                posts.extend(sr_posts)
                print(f"[Reddit] r/{sr}: 新增 {len(sr_posts)} 条")
            else:
                print(f"[Reddit] r/{sr}: 没有新帖子")
            # 板块之间等待
            if sr != sr_list[-1]:
                print(f"[Reddit] 等待 {SUBREDDIT_INTERVAL} 秒...")
                time.sleep(SUBREDDIT_INTERVAL)
    else:
        posts = scraper.fetch_incremental(db, hours=hours)

    if not posts:
        print("没有新帖子")
        return
    count = db.add_posts(posts)
    print(f"增量获取完成: 新增 {count} 条")


def query(start_date=None, end_date=None, subreddit=None, keyword=None, limit=20):
    """查询帖子"""
    db = RedditDatabase(DB_PATH)
    posts = db.query(
        start_date=start_date,
        end_date=end_date,
        subreddit=subreddit,
        keyword=keyword,
        limit=limit
    )
    if not posts:
        print("没有找到匹配的帖子")
        return
    print(f"找到 {len(posts)} 条帖子:\n")
    for i, post in enumerate(posts, 1):
        print(f"{i}. [{post['subreddit']}] {post['title']}")
        print(f"   {post['datetime']} | by {post.get('author', 'N/A')}")
        print(f"   {post['url']}")
        if post.get('content'):
            content = post['content'][:150] + '...' if len(post['content']) > 150 else post['content']
            print(f"   {content}")
        print()


def status():
    """查看数据库状态"""
    db = RedditDatabase(DB_PATH)
    stats = db.get_stats()
    print("=== Reddit Posts 状态 ===")
    print(f"总数: {stats['total']}")
    print(f"今日: {stats['today']}")
    print("\n按板块统计:")
    for item in stats['by_subreddit'][:10]:
        print(f"  {item['subreddit']}: {item['cnt']} 条")


def test(subreddits=None):
    """测试连接"""
    subreddits = subreddits or DEFAULT_SUBREDDITS
    print(f"测试 r/{subreddits} 连接...")
    scraper = RedditScraper(subreddits)
    if scraper.test_connection():
        print("连接成功!")
    else:
        print("连接失败!")


def monitor(interval=30, once=False, subreddits=None):
    """启动定时监控"""
    subreddits = subreddits or DEFAULT_SUBREDDITS
    if once:
        scheduler = RedditScheduler(subreddits, interval)
        scheduler.run_once()
    else:
        print(f"启动定时监控 r/{subreddits}, 间隔 {interval} 分钟")
        print("按 Ctrl+C 停止")
        run_scheduler(subreddits, interval, once=False)


def main():
    """主函数"""
    import fire
    fire.Fire({
        'fetch': fetch,
        'increment': increment,
        'query': query,
        'status': status,
        'test': test,
        'monitor': monitor,
    })


if __name__ == "__main__":
    main()
