# -*- coding: utf-8 -*-
"""
统一爬虫客户端

整合 API 和 Selenium 两种方案，提供统一的接口
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import yaml

try:
    from .api_scraper import APIScraper
    from .selenium_scraper import SeleniumScraper
    from .akshare_scraper import AKShareScraper
    from .storage import TelegraphStorage
except ImportError:
    from api_scraper import APIScraper
    from selenium_scraper import SeleniumScraper
    from akshare_scraper import AKShareScraper
    from storage import TelegraphStorage


class TelegraphClient:
    """电报数据爬虫客户端"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        method: str = "api",
        base_dir: str = "../data/macro_news",
    ):
        """
        初始化客户端

        Args:
            config_path: 配置文件路径
            method: 默认爬取方法 ("api" 或 "selenium")
            base_dir: 数据存储基础目录
        """
        self.method = method
        self.storage = TelegraphStorage(base_dir=base_dir)

        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化爬虫
        self.api_scraper = self._init_api_scraper()
        self.selenium_scraper = None  # 懒加载
        self.akshare_scraper = self._init_akshare_scraper()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "scraper": {
                "timeout": 30,
                "request_interval": 2,
                "max_retries": 3,
                "default_method": "api",
            },
            "selenium": {
                "headless": True,
                "window_size": [1400, 900],
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                # 合并配置
                default_config.update(user_config)

        # 设置日志
        logging.basicConfig(
            level=getattr(logging, default_config.get("logging", {}).get("level", "INFO")),
            format=default_config.get("logging", {}).get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
        )

        return default_config

    def _init_api_scraper(self) -> APIScraper:
        """初始化 API 爬虫"""
        scraper_config = self.config.get("scraper", {})
        return APIScraper(
            timeout=scraper_config.get("timeout", 30),
            request_interval=scraper_config.get("request_interval", 2),
            max_retries=scraper_config.get("max_retries", 3),
        )

    def _init_selenium_scraper(self) -> SeleniumScraper:
        """初始化 Selenium 爬虫"""
        selenium_config = self.config.get("selenium", {})
        return SeleniumScraper(
            headless=selenium_config.get("headless", True),
            window_size=tuple(selenium_config.get("window_size", [1400, 900])),
            timeout=self.config.get("scraper", {}).get("timeout", 30),
        )

    def _init_akshare_scraper(self) -> AKShareScraper:
        """初始化 AKShare 爬虫"""
        scraper_config = self.config.get("scraper", {})
        return AKShareScraper(
            timeout=scraper_config.get("timeout", 30),
            request_interval=scraper_config.get("request_interval", 1.0),
            max_retries=scraper_config.get("max_retries", 3),
        )

    def fetch_daily(
        self,
        dt: date,
        method: Optional[str] = None,
        target_count: int = 1000,
        max_scrolls: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取单日所有电报数据

        Args:
            dt: 日期
            method: 爬取方法（默认使用初始化时指定的方法）
            target_count: Selenium 目标数量
            max_scrolls: Selenium 最大滚动次数

        Returns:
            电报数据列表
        """
        method = method or self.method

        print(f"\n{'='*50}")
        print(f"获取 {dt} 的电报数据 (方法: {method})")
        print(f"{'='*50}")

        if method == "api":
            return self._fetch_by_api(dt)
        elif method == "selenium":
            return self._fetch_by_selenium(dt, target_count=target_count, max_scrolls=max_scrolls)
        elif method == "akshare":
            return self._fetch_by_akshare(dt)
        else:
            raise ValueError(f"不支持的方法: {method}")

    def _fetch_by_api(self, dt: date) -> List[Dict[str, Any]]:
        """
        使用 API 方式获取数据

        Args:
            dt: 日期

        Returns:
            电报数据列表
        """
        try:
            items = self.api_scraper.fetch_daily_telegraph(dt)
            self.storage.save_daily_telegraph(items, dt, raw=True)
            return items
        except Exception as e:
            print(f"[Client] API 方式失败: {e}")
            return []

    def _fetch_by_selenium(
        self,
        dt: date,
        target_count: int = 1000,
        max_scrolls: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        使用 Selenium 方式获取数据

        Args:
            dt: 日期
            target_count: 目标数量
            max_scrolls: 最大滚动次数

        Returns:
            电报数据列表
        """
        try:
            with self._init_selenium_scraper() as scraper:
                items = scraper.fetch_daily_telegraph(
                    dt,
                    target_count=target_count,
                    max_scrolls=max_scrolls,
                )
                self.storage.save_daily_telegraph(items, dt, raw=True)
                return items
        except Exception as e:
            print(f"[Client] Selenium 方式失败: {e}")
            return []

    def _fetch_by_akshare(self, dt: date) -> List[Dict[str, Any]]:
        """
        使用 AKShare 方式获取数据

        Args:
            dt: 日期

        Returns:
            快讯数据列表
        """
        try:
            items = self.akshare_scraper.fetch_daily_telegraph(dt)
            self.storage.save_daily_telegraph(items, dt, raw=True)
            return items
        except Exception as e:
            print(f"[Client] AKShare 方式失败: {e}")
            return []

    def fetch_range(
        self,
        start_date: date,
        end_date: date,
        method: Optional[str] = None,
        check_existing: bool = True,
    ) -> Dict[date, List[Dict[str, Any]]]:
        """
        获取日期范围内的所有电报数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            method: 爬取方法
            check_existing: 是否跳过已存在的数据

        Returns:
            日期到数据列表的映射
        """
        results = {}
        dates = self.storage.get_date_range(start_date, end_date)

        print(f"\n{'='*50}")
        print(f"获取 {start_date} 到 {end_date} 的电报数据")
        print(f"共 {len(dates)} 天")
        print(f"{'='*50}")

        for dt in dates:
            # 检查是否已存在
            if check_existing:
                existing = self.storage.load_daily_telegraph(dt)
                if not existing.empty:
                    print(f"[Client] {dt} 数据已存在，跳过")
                    results[dt] = existing.to_dict("records")
                    continue

            items = self.fetch_daily(dt, method=method)
            results[dt] = items

        return results

    def fetch_today(self, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取今天的电报数据

        Args:
            method: 爬取方法

        Returns:
            电报数据列表
        """
        return self.fetch_daily(date.today(), method=method)

    def fetch_incremental(
        self,
        method: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        增量获取今天的快讯数据

        用于定时任务，每次获取当天的最新数据并保存。
        无论是否已存在数据，都会重新获取最新数据。

        Args:
            method: 爬取方法
            limit: 最大数量

        Returns:
            快讯数据列表
        """
        dt = date.today()
        method = method or self.method

        print(f"\n{'='*50}")
        print(f"增量获取 {dt} 的快讯数据 (方法: {method})")
        print(f"{'='*50}")

        if method == "akshare":
            # AKShare 方式
            items = self._fetch_by_akshare(dt)
            # 保存最新快照
            self.storage.save_latest(items, dt)
            return items
        else:
            # 其他方式
            items = self.fetch_daily(dt, method=method)
            self.storage.save_latest(items, dt)
            return items

    def fetch_latest(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取最新电报

        Args:
            limit: 最大数量

        Returns:
            电报数据列表
        """
        try:
            items = self.api_scraper.fetch_latest(limit)
            self.storage.save_latest(items)
            return items
        except Exception as e:
            print(f"[Client] 获取最新电报失败: {e}")
            return []

    def list_saved_dates(self) -> List[date]:
        """列出已保存数据的日期"""
        return self.storage.list_available_dates()


if __name__ == "__main__":
    # 测试
    client = TelegraphClient()

    # 获取今天数据
    print("=== 获取今天数据 ===")
    today_data = client.fetch_today()
    print(f"今天获取 {len(today_data)} 条")

    # 列出已保存的日期
    saved = client.list_saved_dates()
    print(f"\n已保存 {len(saved)} 天的数据")
