# -*- coding: utf-8 -*-
"""
财联社 API 爬虫

通过调用财联社 API 获取电报数据
"""

import time
import json
import hashlib
import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class CLSScraper:
    """财联社快讯爬虫"""

    SOURCE = "cls"  # 数据源标识
    DISPLAY_NAME = "财联社"

    def __init__(
        self,
        timeout: int = 30,
        request_interval: int = 2,
        max_retries: int = 3,
    ):
        """
        初始化

        Args:
            timeout: 请求超时时间（秒）
            request_interval: 请求间隔（秒）
            max_retries: 最大重试次数
        """
        if requests is None:
            raise ImportError("请安装 requests: pip install requests")

        self.timeout = timeout
        self.request_interval = request_interval
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.cls.cn/",
        })

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送 API 请求"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                last_error = f"请求失败: {e}"
                logger.warning(f"[CLS] 第 {attempt + 1} 次尝试失败: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.request_interval * (attempt + 1))

        logger.error(f"[CLS] 请求失败: {last_error}")
        return None

    def _generate_sign(self) -> str:
        """生成请求签名"""
        import random
        random_str = str(random.random())[2:]
        return hashlib.md5(random_str.encode()).hexdigest()

    def fetch(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取最新快讯

        Args:
            limit: 最大数量

        Returns:
            快讯数据列表
        """
        start_time = time.time()

        # 财联社 API
        url = "https://www.cls.cn/nodeapi/telegraphList"
        params = {
            "timestamp": int(time.time()),
            "sign": self._generate_sign(),
            "limit": limit,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(f"[CLS] 获取最新快讯 (尝试 {attempt + 1}/{self.max_retries})")

                data = self._make_request(url, params)
                if data is None:
                    continue

                # API 可能返回 error 或 code 字段
                error_code = data.get("error", data.get("code", -1))
                if error_code != 0:
                    logger.warning(f"[CLS] API 返回错误: {data.get('message', data)}")
                    continue

                # 解析数据
                items = self._parse_response(data)

                duration = time.time() - start_time
                logger.info(f"[CLS] 获取到 {len(items)} 条快讯 (耗时 {duration:.2f}s)")

                return items[:limit]

            except Exception as e:
                logger.error(f"[CLS] 获取失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_interval)

        return []

    def fetch_incremental(
        self,
        db=None,
        since_minutes: int = 5,
        limit: int = 50,
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
                logger.info(f"[CLS] 过滤后新增 {len(new_items)} 条")
                items = new_items

        return items

    def _parse_response(self, data: Dict) -> List[Dict[str, Any]]:
        """解析 API 响应"""
        items = []
        telegraph_list = data.get("data", {}).get("roll_data", [])

        for row in telegraph_list:
            try:
                # 解析时间
                datetime_str = ""
                ct = row.get("ctime", "")
                if ct:
                    try:
                        dt = datetime.fromtimestamp(int(ct))
                        datetime_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 优先使用 title，其次使用 content
                content = row.get("title", "") or row.get("content", "") or row.get("brief", "")
                if not content:
                    continue

                # 清理内容（去除乱码标记）
                content = self._clean_content(content)

                # 获取分类
                category = row.get("category", "")
                if not category:
                    category = self._detect_category(content)

                item = {
                    "datetime": datetime_str,
                    "content": content,
                    "category": category,
                    "source": self.DISPLAY_NAME,
                    "url": row.get("shareurl", ""),
                }
                items.append(item)

            except Exception as e:
                logger.warning(f"[CLS] 解析数据失败: {e}")
                continue

        return items

    def _clean_content(self, content: str) -> str:
        """清理内容"""
        # 移除乱码字符
        import re
        # 尝试修复常见编码问题
        try:
            # 检测是否为乱码并尝试修复
            if '锟' in content or '鑴' in content or '投诉' in content[:10]:
                # 尝试 gbk 解码再编码
                content = content.encode('latin1').decode('gbk', errors='ignore')
        except:
            pass
        return content.strip()

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

    print("=== 测试财联社快讯获取 ===")

    scraper = CLSScraper()
    items = scraper.fetch(limit=20)

    print(f"\n获取到 {len(items)} 条快讯:\n")
    for i, item in enumerate(items[:10], 1):
        print(f"{i}. [{item['datetime']}] {item['content'][:60]}...")
