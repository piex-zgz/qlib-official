# -*- coding: utf-8 -*-
"""
定时调度器模块

支持定时增量获取财经快讯
"""

import time
import threading
import logging
import signal
import sys
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class NewsScheduler:
    """财经快讯定时调度器"""

    def __init__(
        self,
        interval_minutes: int = 5,
        sources: List[str] = None,
        db_path: str = None,
    ):
        """
        初始化调度器

        Args:
            interval_minutes: 抓取间隔（分钟）
            sources: 数据源列表 ["cls", "em"]
            db_path: 数据库路径
        """
        self.interval_minutes = interval_minutes
        self.sources = sources or ["cls", "em"]
        self.db_path = db_path
        self.scheduler = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 初始化爬虫和数据库
        self._init_components()

    def _init_components(self):
        """初始化组件"""
        # 延迟导入以避免循环依赖
        try:
            from .database import NewsDatabase
            from .api_scraper import CLSScraper
            from .akshare_scraper import AKShareScraper
        except ImportError:
            from database import NewsDatabase
            from api_scraper import CLSScraper
            from akshare_scraper import AKShareScraper

        self.db = NewsDatabase(db_path=self.db_path)

        self.scrapers = {}
        if "cls" in self.sources:
            self.scrapers["cls"] = CLSScraper()
        if "em" in self.sources:
            self.scrapers["em"] = AKShareScraper()

    def _fetch_all_sources(self):
        """抓取所有数据源"""
        logger.info(f"[Scheduler] 开始抓取数据源: {self.sources}")

        for source, scraper in self.scrapers.items():
            try:
                start_time = time.time()

                # 增量获取
                items = scraper.fetch_incremental(db=self.db)

                if items:
                    # 保存到数据库
                    self.db.add_news(items, source=source)
                    self.db.log_fetch(
                        source=source,
                        count=len(items),
                        status="success",
                        duration=time.time() - start_time,
                    )
                    logger.info(f"[Scheduler] {source}: 新增 {len(items)} 条")
                else:
                    self.db.log_fetch(
                        source=source,
                        count=0,
                        status="success",
                        duration=time.time() - start_time,
                    )

            except Exception as e:
                logger.error(f"[Scheduler] {source} 抓取失败: {e}")
                self.db.log_fetch(
                    source=source,
                    count=0,
                    status="error",
                    error_message=str(e),
                )

    def start(self, interval_minutes: int = None):
        """
        启动调度器

        Args:
            interval_minutes: 抓取间隔（分钟）
        """
        if interval_minutes is None:
            interval_minutes = self.interval_minutes

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self._fetch_all_sources,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="fetch_news",
            name="抓取财经快讯",
            replace_existing=True,
            max_instances=1,
        )

        # 立即执行一次
        logger.info("[Scheduler] 执行首次抓取...")
        self.executor.submit(self._fetch_all_sources)

        self.scheduler.start()
        self.running = True

        logger.info(f"[Scheduler] 启动成功，每 {interval_minutes} 分钟抓取一次")

    def stop(self):
        """停止调度器"""
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
        self.executor.shutdown(wait=False)
        self.running = False
        self.db.close()
        logger.info("[Scheduler] 已停止")

    def run_once(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        立即执行一次抓取

        Returns:
            各数据源的抓取结果
        """
        results = {}

        for source, scraper in self.scrapers.items():
            try:
                items = scraper.fetch_incremental(db=self.db)
                if items:
                    self.db.add_news(items, source=source)
                results[source] = items
            except Exception as e:
                logger.error(f"[Scheduler] {source} 抓取失败: {e}")
                results[source] = []

        return results

    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        stats = self.db.get_stats()
        stats["running"] = self.running
        stats["interval_minutes"] = self.interval_minutes
        stats["sources"] = list(self.scrapers.keys())
        return stats


class SchedulerApp:
    """调度器应用（支持后台运行）"""

    def __init__(self, interval_minutes: int = 5, sources: List[str] = None):
        self.interval_minutes = interval_minutes
        self.sources = sources or ["cls", "em"]
        self.scheduler = None

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"收到信号 {signum}，正在停止...")
        self.stop()
        sys.exit(0)

    def start(self, interval_minutes: int = None):
        """
        启动应用

        Args:
            interval_minutes: 抓取间隔
        """
        if interval_minutes is None:
            interval_minutes = self.interval_minutes

        logger.info(f"启动财经快讯监控...")
        logger.info(f"  数据源: {self.sources}")
        logger.info(f"  抓取间隔: {interval_minutes} 分钟")

        self.scheduler = NewsScheduler(
            interval_minutes=interval_minutes,
            sources=self.sources,
        )
        self.scheduler.start(interval_minutes=interval_minutes)

        # 保持运行
        try:
            while True:
                time.sleep(60)
                self._print_status()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """停止应用"""
        if self.scheduler:
            self.scheduler.stop()

    def _print_status(self):
        """打印状态"""
        status = self.scheduler.get_status()
        logger.info(
            f"[状态] 总计: {status['total']} | "
            f"cls: {status['by_source'].get('cls', 0)} | "
            f"em: {status['by_source'].get('em', 0)}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="财经快讯定时监控")
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="抓取间隔（分钟）",
    )
    parser.add_argument(
        "--sources", "-s",
        nargs="+",
        default=["cls", "em"],
        choices=["cls", "em"],
        help="数据源列表",
    )
    parser.add_argument(
        "--once", "-o",
        action="store_true",
        help="只执行一次",
    )

    args = parser.parse_args()

    if args.once:
        # 只执行一次
        print("=== 执行一次抓取 ===")
        scheduler = NewsScheduler(interval_minutes=args.interval, sources=args.sources)
        results = scheduler.run_once()
        for source, items in results.items():
            print(f"{source}: {len(items)} 条")
        scheduler.stop()
    else:
        # 启动定时监控
        app = SchedulerApp(interval_minutes=args.interval, sources=args.sources)
        app.start()
