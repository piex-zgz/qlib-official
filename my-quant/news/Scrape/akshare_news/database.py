# -*- coding: utf-8 -*-
"""
新闻数据库模块

使用 SQLite 存储财经快讯，支持多数据源
"""

import sqlite3
import json
import threading
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NewsDatabase:
    """财经快讯数据库"""

    # 数据库模式
    SCHEMA = """
    -- 快讯主表
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uuid TEXT UNIQUE NOT NULL,           -- 唯一标识
        datetime TEXT NOT NULL,               -- 快讯时间
        content TEXT NOT NULL,               -- 快讯内容
        category TEXT,                        -- 分类
        source TEXT NOT NULL,                 -- 数据源 (cls/em)
        url TEXT,                             -- 原文链接
        fetch_time TEXT NOT NULL,             -- 抓取时间
        date_only TEXT NOT NULL,              -- 日期 (用于分区查询)
        created_at TEXT NOT NULL              -- 记录创建时间
    );

    -- 创建索引
    CREATE INDEX IF NOT EXISTS idx_news_date ON news(date_only);
    CREATE INDEX IF NOT EXISTS idx_news_datetime ON news(datetime);
    CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);
    CREATE INDEX IF NOT EXISTS idx_news_category ON news(category);
    CREATE INDEX IF NOT EXISTS idx_news_uuid ON news(uuid);

    -- 数据源配置表
    CREATE TABLE IF NOT EXISTS data_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        last_fetch_time TEXT,
        last_fetch_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        created_at TEXT NOT NULL
    );

    -- 抓取日志表
    CREATE TABLE IF NOT EXISTS fetch_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        fetch_time TEXT NOT NULL,
        count INTEGER NOT NULL,
        status TEXT NOT NULL,
        error_message TEXT,
        duration REAL
    );
    """

    def __init__(self, db_path: str = None):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径（默认: data/news.db）
        """
        if db_path is None:
            # 默认使用项目根目录下的 data/news.db
            db_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "news.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(str(self.db_path))
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self):
        """初始化数据库"""
        conn = self._get_connection()
        conn.executescript(self.SCHEMA)
        conn.commit()

        # 初始化数据源
        self._init_sources(conn)

    def _init_sources(self, conn: sqlite3.Connection):
        """初始化数据源配置"""
        sources = [
            ('cls', '财联社', datetime.now().isoformat()),
            ('em', '东方财富', datetime.now().isoformat()),
        ]
        conn.executemany(
            """INSERT OR IGNORE INTO data_sources
               (name, display_name, last_fetch_time, created_at)
               VALUES (?, ?, ?, ?)""",
            [(s[0], s[1], s[2], datetime.now().isoformat()) for s in sources]
        )
        conn.commit()

    def _generate_uuid(self, content: str, datetime_str: str, source: str) -> str:
        """生成唯一标识"""
        import hashlib
        raw = f"{source}:{datetime_str}:{content[:100]}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def add_news(
        self,
        items: List[Dict[str, Any]],
        source: str,
        fetch_time: Optional[str] = None,
    ) -> int:
        """
        添加快讯数据

        Args:
            items: 快讯列表
            source: 数据源标识 (cls/em)
            fetch_time: 抓取时间

        Returns:
            插入数量
        """
        if not items:
            return 0

        if fetch_time is None:
            fetch_time = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()
        inserted = 0

        for item in items:
            try:
                # 生成 UUID
                uuid = self._generate_uuid(
                    item.get('content', ''),
                    item.get('datetime', ''),
                    source
                )

                # 解析日期
                dt_str = item.get('datetime', '')
                try:
                    dt = datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
                    date_only = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    date_only = date.today().strftime("%Y-%m-%d")

                cursor.execute(
                    """INSERT OR IGNORE INTO news
                       (uuid, datetime, content, category, source, url, fetch_time, date_only, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid,
                        item.get('datetime', ''),
                        item.get('content', ''),
                        item.get('category', '快讯'),
                        source,
                        item.get('url', ''),
                        fetch_time,
                        date_only,
                        datetime.now().isoformat(),
                    )
                    # 检查是否真的插入了（INSERT OR IGNORE 可能不插入）
                )
                if cursor.rowcount > 0:
                    inserted += 1

            except Exception as e:
                logger.warning(f"插入快讯失败: {e}")
                continue

        conn.commit()
        logger.info(f"[DB] 插入 {inserted}/{len(items)} 条 {source} 快讯")
        return inserted

    def log_fetch(
        self,
        source: str,
        count: int,
        status: str,
        error_message: Optional[str] = None,
        duration: Optional[float] = None,
    ):
        """记录抓取日志"""
        conn = self._get_connection()
        conn.execute(
            """INSERT INTO fetch_logs (source, fetch_time, count, status, error_message, duration)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                source,
                datetime.now().isoformat(),
                count,
                status,
                error_message,
                duration,
            )
        )
        conn.commit()

        # 更新数据源状态
        conn.execute(
            """UPDATE data_sources SET last_fetch_time = ?, last_fetch_count = ? WHERE name = ?""",
            (datetime.now().isoformat(), count, source)
        )
        conn.commit()

    def query(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        查询快讯

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            source: 数据源
            category: 分类
            keyword: 关键词
            limit: 最大数量
            offset: 偏移量

        Returns:
            快讯列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if start_date:
            conditions.append("date_only >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("date_only <= ?")
            params.append(end_date)

        if source:
            conditions.append("source = ?")
            params.append(source)

        if category:
            conditions.append("category = ?")
            params.append(category)

        if keyword:
            conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT id, uuid, datetime, content, category, source, url, fetch_time, date_only, created_at
            FROM news
            WHERE {where_clause}
            ORDER BY datetime DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def count(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        """统计数量"""
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if start_date:
            conditions.append("date_only >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("date_only <= ?")
            params.append(end_date)

        if source:
            conditions.append("source = ?")
            params.append(source)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(f"SELECT COUNT(*) FROM news WHERE {where_clause}", params)
        return cursor.fetchone()[0]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 总数
        cursor.execute("SELECT COUNT(*) FROM news")
        total = cursor.fetchone()[0]

        # 按数据源统计
        cursor.execute("SELECT source, COUNT(*) as cnt FROM news GROUP BY source")
        by_source = {row['source']: row['cnt'] for row in cursor.fetchall()}

        # 按日期统计
        cursor.execute("SELECT date_only, COUNT(*) as cnt FROM news GROUP BY date_only ORDER BY date_only DESC LIMIT 30")
        by_date = [{'date': row['date_only'], 'count': row['cnt']} for row in cursor.fetchall()]

        # 最近抓取日志
        cursor.execute("""
            SELECT source, fetch_time, count, status
            FROM fetch_logs
            ORDER BY fetch_time DESC
            LIMIT 10
        """)
        recent_fetches = [dict(row) for row in cursor.fetchall()]

        return {
            'total': total,
            'by_source': by_source,
            'by_date': by_date,
            'recent_fetches': recent_fetches,
        }

    def export(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: Optional[str] = None,
        format: str = 'csv',
    ) -> str:
        """
        导出数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源
            format: 导出格式 (csv/json)

        Returns:
            导出文件路径
        """
        import csv

        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if start_date:
            conditions.append("date_only >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("date_only <= ?")
            params.append(end_date)

        if source:
            conditions.append("source = ?")
            params.append(source)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor.execute(
            f"""SELECT datetime, content, category, source, url, date_only, fetch_time
                FROM news WHERE {where_clause} ORDER BY datetime DESC""",
            params
        )

        if format == 'csv':
            output_path = self.db_path.parent / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['datetime', 'content', 'category', 'source', 'url', 'date_only', 'fetch_time'])
                for row in cursor.fetchall():
                    writer.writerow(row)
            return str(output_path)

        elif format == 'json':
            output_path = self.db_path.parent / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            data = [dict(row) for row in cursor.fetchall()]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return str(output_path)

        return ""

    def close(self):
        """关闭连接"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


if __name__ == "__main__":
    # 测试
    db = NewsDatabase()

    # 统计
    stats = db.get_stats()
    print(f"\n数据库统计:")
    print(f"  总快讯数: {stats['total']}")
    print(f"  按来源: {stats['by_source']}")
    print(f"  最近日期: {stats['by_date'][:5]}")

    # 查询最近 10 条
    print(f"\n最近 10 条快讯:")
    for item in db.query(limit=10):
        print(f"  [{item['datetime']}] [{item['source']}] {item['content'][:50]}...")
