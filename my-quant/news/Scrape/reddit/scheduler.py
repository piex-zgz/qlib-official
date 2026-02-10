"""
Reddit Scheduler Module - 定时调度器
"""

import signal
import sys
import threading
import logging
from datetime import datetime

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
except ImportError:
    print("需要安装 APScheduler: pip install apscheduler")
    sys.exit(1)

from scraper import RedditScraper
from database import RedditDatabase
from config import (
    MONITOR_INTERVAL_MINUTES,
    DEFAULT_SUBREDDITS,
    LOG_LEVEL,
    SUBREDDIT_INTERVAL,
    INCREMENT_LIMIT
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RedditScheduler')


def log(msg):
    """打印日志（兼容后台线程）"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


class RedditScheduler:
    """Reddit 定时调度器"""

    def __init__(self, subreddits=None, interval_minutes=None, db_path=None):
        """
        初始化调度器

        Args:
            subreddits: Reddit板块列表
            interval_minutes: 抓取间隔（分钟）
            db_path: 数据库路径
        """
        self.subreddits = subreddits or DEFAULT_SUBREDDITS
        self.interval_minutes = interval_minutes or MONITOR_INTERVAL_MINUTES
        self.db_path = db_path

        self.scraper = RedditScraper(self.subreddits)
        self.db = RedditDatabase(self.db_path)

        self.scheduler = BackgroundScheduler()
        self._running = False
        self._stop_event = threading.Event()

    def _fetch_once(self):
        """执行一次抓取"""
        log(f"开始抓取 r/{self.subreddits} ...")
        try:
            sr_list = self.scraper.list_subreddits()
            if len(sr_list) > 1:
                # 多板块：循环抓取
                log(f"检测到 {len(sr_list)} 个板块，将循环抓取...")
                posts = self.scraper.fetch_all(limit=INCREMENT_LIMIT, interval_seconds=SUBREDDIT_INTERVAL)
            else:
                posts = self.scraper.fetch_incremental(self.db)

            if posts:
                count = self.db.add_posts(posts)
                log(f"新增 {count} 条帖子 (共 {len(posts)} 条)")
            else:
                log("没有新帖子")
        except Exception as e:
            log(f"抓取出错: {e}")

    def start(self, interval_minutes=None):
        """
        启动调度器

        Args:
            interval_minutes: 可选的间隔覆盖
        """
        if interval_minutes:
            self.interval_minutes = interval_minutes

        # 添加定时任务
        self.scheduler.add_job(
            self._fetch_once,
            IntervalTrigger(minutes=self.interval_minutes),
            id='reddit_fetch',
            name=f'Reddit Fetch ({self.subreddits})',
            replace_existing=True
        )

        # 立即执行一次
        self._fetch_once()

        self._running = True
        self.scheduler.start()
        log(f"调度器已启动，间隔 {self.interval_minutes} 分钟")

    def stop(self):
        """停止调度器"""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            self._stop_event.set()
            log("调度器已停止")

    def run_once(self):
        """只执行一次抓取（不启动调度）"""
        self._fetch_once()

    def get_status(self):
        """获取状态"""
        stats = self.db.get_stats()
        return {
            'running': self._running,
            'interval_minutes': self.interval_minutes,
            'subreddits': self.subreddits,
            'stats': stats
        }


def run_scheduler(subreddits=None, interval_minutes=None, once=False):
    """运行调度器（单次或持续）"""
    if interval_minutes is None:
        interval_minutes = MONITOR_INTERVAL_MINUTES
    scheduler = RedditScheduler(subreddits, interval_minutes)
    running = [True]  # 用列表包装以便在闭包中修改

    def signal_handler(signum, frame):
        print("\n正在停止调度器...")
        running[0] = False
        scheduler.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if once:
        scheduler.run_once()
    else:
        scheduler.start()

    # 保持运行
    try:
        while running[0]:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Reddit News Scheduler')
    parser.add_argument('--subreddits', '-s', type=str, default=None,
                        help='Reddit板块列表，用+连接')
    parser.add_argument('--interval', '-i', type=int, default=MONITOR_INTERVAL_MINUTES,
                        help='抓取间隔（分钟）')
    parser.add_argument('--once', action='store_true',
                        help='只执行一次')

    args = parser.parse_args()

    run_scheduler(args.subreddits, args.interval, args.once)
