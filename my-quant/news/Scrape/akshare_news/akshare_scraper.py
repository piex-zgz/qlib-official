# -*- coding: utf-8 -*-
"""
AKShare 财经快讯爬虫

使用 AKShare 获取东方财富全球财经快讯
数据源: ak.stock_info_global_em()
"""

import time
import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    ak = None
    pd = None

logger = logging.getLogger(__name__)


class AKShareScraper:
    """AKShare 方式获取财经快讯"""

    SOURCE = "em"  # 数据源标识
    DISPLAY_NAME = "东方财富"

    def __init__(
        self,
        timeout: int = 30,
        request_interval: float = 1.0,
        max_retries: int = 3,
    ):
        """
        初始化

        Args:
            timeout: 请求超时时间（秒）
            request_interval: 请求间隔（秒）
            max_retries: 最大重试次数
        """
        if ak is None:
            raise ImportError("请安装 akshare: pip install akshare")

        self.timeout = timeout
        self.request_interval = request_interval
        self.max_retries = max_retries

    def fetch(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        获取最新快讯

        Args:
            limit: 最大数量

        Returns:
            快讯数据列表
        """
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                logger.info(f"AKShare: 获取最新快讯 (尝试 {attempt + 1}/{self.max_retries})")

                # 获取东方财富全球财经快讯
                df = ak.stock_info_global_em()

                if df is None or df.empty:
                    logger.warning("AKShare: 未获取到数据")
                    return []

                # 转换格式
                items = self._convert_dataframe(df)

                # 限制数量
                items = items[:limit]

                duration = time.time() - start_time
                logger.info(f"AKShare: 获取到 {len(items)} 条快讯 (耗时 {duration:.2f}s)")

                return items

            except Exception as e:
                logger.error(f"AKShare: 获取失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_interval)
                else:
                    raise

        return []

    def fetch_incremental(
        self,
        db=None,
        since_minutes: int = 5,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        增量获取快讯

        Args:
            db: 数据库实例
            since_minutes: 获取最近多少分钟的快讯
            limit: 最大数量

        Returns:
            快讯数据列表
        """
        items = self.fetch(limit=limit)

        if db is not None and items:
            # 获取当前数据库中最新的快讯时间
            latest = db.query(limit=1, source=self.SOURCE)
            if latest:
                latest_dt = datetime.fromisoformat(latest[0]['datetime'])
                # 过滤出新快讯
                new_items = []
                for item in items:
                    item_dt = datetime.fromisoformat(item['datetime'][:19])
                    if item_dt > latest_dt:
                        new_items.append(item)
                logger.info(f"AKShare: 过滤后新增 {len(new_items)} 条")
                items = new_items

        return items

    def _convert_dataframe(self, df) -> List[Dict[str, Any]]:
        """
        转换 DataFrame 为标准格式
        """
        items = []

        # 查找列
        time_col = None
        title_col = None
        source_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "时间" in col or "time" in col_lower:
                time_col = col
                break

        for col in df.columns:
            col_lower = col.lower()
            if "标题" in col or "title" in col_lower:
                title_col = col
                break

        if title_col is None:
            title_col = df.columns[0]

        for col in df.columns:
            col_lower = col.lower()
            if "来源" in col or "source" in col_lower:
                source_col = col
                break

        for idx, row in df.iterrows():
            try:
                # 提取时间
                if time_col and not pd.isna(row[time_col]):
                    time_str = str(row[time_col])
                else:
                    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 提取内容
                content = str(row[title_col]) if not pd.isna(row[title_col]) else ""

                # 提取来源
                source = str(row[source_col]) if source_col and not pd.isna(row[source_col]) else self.DISPLAY_NAME

                item = {
                    "datetime": time_str,
                    "content": content,
                    "category": self._detect_category(content),
                    "source": self.DISPLAY_NAME,
                    "url": "",
                }
                items.append(item)

            except Exception as e:
                logger.warning(f"转换行数据失败: {e}")
                continue

        return items

    def _detect_category(self, content: str) -> str:
        """检测快讯分类"""
        import re

        if content.startswith("【"):
            match = re.search(r"【([^【】]+)】", content)
            if match:
                return match.group(1).strip()

        return "快讯"


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== 测试 AKShare 快讯获取 ===")

    scraper = AKShareScraper()
    items = scraper.fetch(limit=20)

    print(f"\n获取到 {len(items)} 条快讯:\n")
    for i, item in enumerate(items[:10], 1):
        print(f"{i}. [{item['datetime']}] {item['content'][:60]}...")
