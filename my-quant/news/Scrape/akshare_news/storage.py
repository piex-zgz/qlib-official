# -*- coding: utf-8 -*-
"""
数据存储模块

负责将电报数据按日期存储到 CSV 文件中
"""

import os
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path


class TelegraphStorage:
    """电报数据存储类"""

    def __init__(self, base_dir: str = None):
        """
        初始化存储类

        Args:
            base_dir: 数据存储的基础目录
        """
        if base_dir is None:
            # 默认使用项目根目录下的 data/macro_news
            base_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "macro_news"
        self.base_dir = Path(base_dir)
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """确保基础目录存在"""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_date_dir(self, dt: date) -> Path:
        """
        获取日期目录路径

        Args:
            dt: 日期

        Returns:
            日期目录路径
        """
        date_str = dt.strftime("%Y%m%d")
        return self.base_dir / date_str

    def save_daily_telegraph(
        self,
        data: List[Dict[str, Any]],
        dt: date,
        raw: bool = True,
    ) -> str:
        """
        保存单日电报数据

        Args:
            data: 电报数据列表
            dt: 日期
            raw: 是否保存为原始数据（否则为清洗后数据）

        Returns:
            保存的文件路径
        """
        if not data:
            print(f"[Storage] {dt} 无数据可保存")
            return ""

        date_dir = self._get_date_dir(dt)
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = "telegraph_raw.csv" if raw else "telegraph_clean.csv"
        filepath = date_dir / filename

        df = pd.DataFrame(data)

        # 调整列顺序
        preferred_cols = ["datetime", "content", "category", "source", "url"]
        cols = [c for c in preferred_cols if c in df.columns] + \
               [c for c in df.columns if c not in preferred_cols]
        df = df[cols]

        # 保存为 CSV
        df.to_csv(filepath, index=False, encoding="utf-8-sig")

        print(f"[Storage] 已保存 {dt}: {len(df)} 条记录 -> {filepath}")
        return str(filepath)

    def load_daily_telegraph(
        self,
        dt: date,
        raw: bool = True,
    ) -> pd.DataFrame:
        """
        加载单日电报数据

        Args:
            dt: 日期
            raw: 是否读取原始数据

        Returns:
            电报数据 DataFrame
        """
        date_dir = self._get_date_dir(dt)
        filename = "telegraph_raw.csv" if raw else "telegraph_clean.csv"
        filepath = date_dir / filename

        if not filepath.exists():
            return pd.DataFrame()

        return pd.read_csv(filepath)

    def save_latest(
        self,
        data: List[Dict[str, Any]],
        dt: Optional[date] = None,
    ) -> str:
        """
        保存最新数据快照

        Args:
            data: 电报数据列表
            dt: 日期（默认为今天）

        Returns:
            保存的文件路径
        """
        if dt is None:
            dt = date.today()

        filepath = self.base_dir / "telegraph_latest.csv"

        if data:
            df = pd.DataFrame(data)
            # 只保留最近的数据
            if len(df) > 100:
                df = df.head(100)

            df.to_csv(filepath, index=False, encoding="utf-8-sig")
        else:
            # 创建空文件
            df = pd.DataFrame(columns=["datetime", "content", "category", "source", "url"])
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

        print(f"[Storage] 最新快照已更新: {filepath}")
        return str(filepath)

    def list_available_dates(self) -> List[date]:
        """
        列出已保存数据的日期

        Returns:
            日期列表
        """
        dates = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
                try:
                    dates.append(datetime.strptime(d.name, "%Y%m%d").date())
                except ValueError:
                    continue
        return sorted(dates)

    def get_date_range(self, start: date, end: date) -> List[date]:
        """
        获取日期范围内的所有日期

        Args:
            start: 开始日期
            end: 结束日期

        Returns:
            日期列表
        """
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current = date.fromordinal(current.toordinal() + 1)
        return dates


if __name__ == "__main__":
    # 测试
    storage = TelegraphStorage()

    # 列出已保存的日期
    dates = storage.list_available_dates()
    print(f"已保存 {len(dates)} 天的数据:")
    for d in dates[-5:]:
        print(f"  {d}")

    # 测试保存
    test_data = [
        {
            "datetime": "2026-02-05 09:30:00",
            "content": "测试电报内容 1",
            "category": "快讯",
            "source": "财联社",
            "url": "https://www.cls.cn/telegraph/123",
        },
        {
            "datetime": "2026-02-05 09:31:00",
            "content": "测试电报内容 2",
            "category": "深度",
            "source": "财联社",
            "url": "https://www.cls.cn/telegraph/124",
        },
    ]

    storage.save_daily_telegraph(test_data, date(2026, 2, 5))
    storage.save_latest(test_data, date(2026, 2, 5))
