"""
Reddit Posts Database Module
"""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path

from config import DB_PATH


class RedditDatabase:
    """Reddit帖子数据库操作"""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = DB_PATH
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_db()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE NOT NULL,
                    post_id TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    subreddit TEXT NOT NULL,
                    author TEXT,
                    url TEXT NOT NULL,
                    fetch_time TEXT NOT NULL,
                    date_only TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_date
                ON reddit_posts(date_only)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_subreddit
                ON reddit_posts(subreddit)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_datetime
                ON reddit_posts(datetime)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reddit_uuid
                ON reddit_posts(uuid)
            """)
            conn.commit()

    def _generate_uuid(self, post_id, subreddit, datetime_str):
        """生成唯一标识符"""
        content = f"reddit:{post_id}:{subreddit}:{datetime_str}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def add_post(self, post):
        """
        添加帖子，返回是否新增成功

        Args:
            post: {
                'post_id': str,
                'datetime': str,  # YYYY-MM-DD HH:MM:SS
                'title': str,
                'content': str,
                'subreddit': str,
                'author': str,
                'url': str,
            }

        Returns:
            bool: 是否新增（True）或已存在（False）
        """
        if not post.get('post_id'):
            return False

        fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_only = post['datetime'][:10] if 'datetime' in post else fetch_time[:10]
        uuid = self._generate_uuid(
            post['post_id'],
            post['subreddit'],
            post['datetime']
        )
        created_at = fetch_time

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO reddit_posts
                (uuid, post_id, datetime, title, content, subreddit, author, url, fetch_time, date_only, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid,
                    post['post_id'],
                    post['datetime'],
                    post['title'],
                    post.get('content', ''),
                    post['subreddit'],
                    post.get('author', ''),
                    post['url'],
                    fetch_time,
                    date_only,
                    created_at
                )
            )
            conn.commit()
            return cursor.rowcount > 0

    def add_posts(self, posts):
        """
        批量添加帖子

        Args:
            posts: 帖子字典列表

        Returns:
            int: 新增数量
        """
        count = 0
        for post in posts:
            if self.add_post(post):
                count += 1
        return count

    def get_latest_post_time(self, subreddit=None):
        """获取最新帖子时间"""
        with self._get_connection() as conn:
            if subreddit:
                result = conn.execute(
                    """
                    SELECT datetime FROM reddit_posts
                    WHERE subreddit = ?
                    ORDER BY datetime DESC
                    LIMIT 1
                    """,
                    (subreddit,)
                ).fetchone()
            else:
                result = conn.execute(
                    """
                    SELECT datetime FROM reddit_posts
                    ORDER BY datetime DESC
                    LIMIT 1
                    """
                ).fetchone()
            return result['datetime'] if result else None

    def query(self, start_date=None, end_date=None, subreddit=None,
             keyword=None, limit=100):
        """
        查询帖子

        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            subreddit: 板块名称
            keyword: 关键词
            limit: 返回数量限制

        Returns:
            list: 帖子字典列表
        """
        conditions = []
        params = []

        if start_date:
            conditions.append("datetime >= ?")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("datetime <= ?")
            params.append(f"{end_date} 23:59:59")
        if subreddit:
            conditions.append("subreddit = ?")
            params.append(subreddit)
        if keyword:
            conditions.append("(title LIKE ? OR content LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM reddit_posts
                WHERE {where_clause}
                ORDER BY datetime DESC
                LIMIT ?
                """,
                params + [limit]
            ).fetchall()
            return [dict(row) for row in rows]

    def count(self, start_date=None, end_date=None, subreddit=None):
        """统计帖子数量"""
        conditions = []
        params = []

        if start_date:
            conditions.append("datetime >= ?")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("datetime <= ?")
            params.append(f"{end_date} 23:59:59")
        if subreddit:
            conditions.append("subreddit = ?")
            params.append(subreddit)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_connection() as conn:
            result = conn.execute(
                f"SELECT COUNT(*) as cnt FROM reddit_posts WHERE {where_clause}",
                params
            ).fetchone()
            return result['cnt']

    def get_stats(self):
        """获取统计信息"""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM reddit_posts").fetchone()['cnt']
            today = datetime.now().strftime('%Y-%m-%d')
            today_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM reddit_posts WHERE date_only = ?",
                (today,)
            ).fetchone()['cnt']
            subreddits = conn.execute(
                "SELECT subreddit, COUNT(*) as cnt FROM reddit_posts GROUP BY subreddit ORDER BY cnt DESC"
            ).fetchall()
            return {
                'total': total,
                'today': today_count,
                'by_subreddit': [dict(r) for r in subreddits]
            }

    def delete_old_posts(self, days=30):
        """删除旧帖子"""
        cutoff = datetime.now().strftime('%Y-%m-%d')
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM reddit_posts WHERE date_only < ?",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
