"""
Reddit RSS Scraper Module
"""

import feedparser
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

from config import (
    DEFAULT_SUBREDDITS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    REQUEST_INTERVAL,
    DEFAULT_LIMIT,
    INCREMENT_HOURS,
    INCREMENT_LIMIT,
    RSS_LIMIT
)


def _ensure_list(subreddits):
    """确保subreddits是列表"""
    if subreddits is None:
        subreddits = DEFAULT_SUBREDDITS
    if isinstance(subreddits, str):
        subreddits = [subreddits]
    return subreddits


def _to_query_string(subreddits):
    """将板块列表转为URL查询字符串"""
    if isinstance(subreddits, list):
        return '+'.join(subreddits)
    return subreddits


class RedditScraper:
    """Reddit RSS 爬虫"""

    SOURCE = "reddit"
    DISPLAY_NAME = "Reddit"

    def __init__(self, subreddits=None, config=None):
        """
        初始化爬虫

        Args:
            subreddits: Reddit板块，字符串如 "LocalLlaMA+OpenAI" 或列表如 ["LocalLlaMA", "OpenAI"]
            config: 配置字典，可选
        """
        if subreddits is None:
            subreddits = _ensure_list(DEFAULT_SUBREDDITS)
        else:
            subreddits = _ensure_list(subreddits)
        self.subreddits = subreddits

        if config is None:
            config = {
                'max_retries': MAX_RETRIES,
                'timeout': REQUEST_TIMEOUT,
                'request_interval': REQUEST_INTERVAL,
                'default_limit': DEFAULT_LIMIT
            }
        self.config = config

    def _parse_entry(self, entry):
        """
        解析单个RSS条目

        Args:
            entry: feedparser entry

        Returns:
            dict: 标准化的帖子数据
        """
        # 提取发布时间
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6])
            datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 提取post_id从URL
        url = entry.link if hasattr(entry, 'link') else ''
        post_id = ''
        if url:
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                post_id = path_parts[-1] if path_parts[-1] else path_parts[-2]

        # 提取板块名称
        subreddit = self.subreddits[0] if isinstance(self.subreddits, list) else self.subreddits.split('+')[0]
        if hasattr(entry, 'id') and entry.id:
            # 从id中提取板块
            id_str = str(entry.id)
            if '/r/' in id_str:
                start = id_str.find('/r/') + 3
                end = id_str.find('/', start)
                if end == -1:
                    end = id_str.find('.', start)
                if end == -1:
                    end = len(id_str)
                subreddit = id_str[start:end]

        # 提取内容摘要
        content = ''
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description

        # 清理HTML标签（简单处理）
        import re
        content = re.sub(r'<[^>]+>', '', content)
        content = content.strip()[:1000] if content else ''

        # 提取作者
        author = ''
        if hasattr(entry, 'author'):
            author = entry.author

        return {
            'post_id': post_id,
            'datetime': datetime_str,
            'title': entry.title if hasattr(entry, 'title') else '',
            'content': content,
            'subreddit': subreddit,
            'author': author,
            'url': url,
            'source': self.SOURCE
        }

    def fetch(self, limit=None, since_datetime=None):
        """
        获取帖子

        Args:
            limit: 返回数量限制
            since_datetime: 只返回此时间之后的帖子 (datetime str)

        Returns:
            list: 帖子字典列表
        """
        if limit is None:
            limit = self.config.get('default_limit', DEFAULT_LIMIT)

        max_retries = self.config.get('max_retries', MAX_RETRIES)
        request_interval = self.config.get('request_interval', REQUEST_INTERVAL)

        # 构建RSS URL，添加limit参数
        query_str = _to_query_string(self.subreddits)
        rss_url = f"https://www.reddit.com/r/{query_str}/new.rss?limit={RSS_LIMIT}"

        for attempt in range(max_retries):
            try:
                # feedparser 自动处理 User-Agent
                feed = feedparser.parse(rss_url)

                if hasattr(feed, 'status') and feed.status != 200:
                    print(f"[Reddit] RSS请求失败，状态码: {feed.status}")
                    time.sleep(request_interval * (attempt + 1))
                    continue

                posts = []
                for entry in feed.entries[:limit]:
                    post = self._parse_entry(entry)

                    # 过滤时间
                    if since_datetime and post['datetime'] <= since_datetime:
                        continue

                    posts.append(post)

                return posts

            except Exception as e:
                print(f"[Reddit] 抓取失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(request_interval * (attempt + 1))

        return []

    def fetch_latest(self, limit=DEFAULT_LIMIT):
        """获取最新帖子"""
        return self.fetch(limit=limit)

    def fetch_incremental(self, db, hours=None):
        """
        增量获取新帖子

        Args:
            db: RedditDatabase实例
            hours: 获取多少小时内的帖子 (默认使用 INCREMENT_HOURS)

        Returns:
            list: 新帖子列表
        """
        if hours is None:
            hours = INCREMENT_HOURS
        limit = INCREMENT_LIMIT

        latest_dt = db.get_latest_post_time()
        if latest_dt is None:
            # 首次抓取，获取过去N小时
            posts = self.fetch(limit=limit)
            # 过滤出过去N小时内的
            cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            posts = [p for p in posts if p['datetime'] > cutoff]
        else:
            # 非首次抓取，获取所有新帖子（数据库会MD5去重）
            posts = self.fetch(limit=limit)

        return posts

    def list_subreddits(self):
        """获取监控的板块列表"""
        if isinstance(self.subreddits, list):
            return self.subreddits
        return self.subreddits.split('+')

    def fetch_all(self, limit=None, interval_seconds=60):
        """
        循环抓取所有板块，每个板块间隔一定时间

        Args:
            limit: 每个板块返回数量限制
            interval_seconds: 板块之间等待时间（秒），默认60秒（1分钟）

        Returns:
            list: 所有帖子字典列表
        """
        if limit is None:
            limit = self.config.get('default_limit', DEFAULT_LIMIT)

        all_posts = []
        subreddits = self.list_subreddits()

        for i, sr in enumerate(subreddits):
            print(f"[Reddit] 抓取板块 {i+1}/{len(subreddits)}: r/{sr}")
            posts = self._fetch_single(sr, limit)

            if posts:
                all_posts.extend(posts)
                print(f"[Reddit] r/{sr}: 获取 {len(posts)} 条")
            else:
                print(f"[Reddit] r/{sr}: 没有新帖子")

            # 如果不是最后一个板块，等待一段时间
            if i < len(subreddits) - 1:
                print(f"[Reddit] 等待 {interval_seconds} 秒后抓取下一个板块...")
                time.sleep(interval_seconds)

        return all_posts

    def _fetch_single(self, subreddit, limit=None):
        """
        抓取单个板块

        Args:
            subreddit: 单个板块名
            limit: 返回数量限制

        Returns:
            list: 帖子字典列表
        """
        if limit is None:
            limit = self.config.get('default_limit', DEFAULT_LIMIT)

        max_retries = self.config.get('max_retries', MAX_RETRIES)
        request_interval = self.config.get('request_interval', REQUEST_INTERVAL)

        rss_url = f"https://www.reddit.com/r/{subreddit}/new.rss?limit={RSS_LIMIT}"

        for attempt in range(max_retries):
            try:
                feed = feedparser.parse(rss_url)

                if hasattr(feed, 'status') and feed.status != 200:
                    print(f"[Reddit] RSS请求失败，状态码: {feed.status}")
                    time.sleep(request_interval * (attempt + 1))
                    continue

                posts = []
                for entry in feed.entries[:limit]:
                    post = self._parse_entry_for_subreddit(entry, subreddit)
                    posts.append(post)

                return posts

            except Exception as e:
                print(f"[Reddit] 抓取失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(request_interval * (attempt + 1))

        return []

    def _parse_entry_for_subreddit(self, entry, subreddit):
        """为单个板块解析RSS条目"""
        # 提取发布时间
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6])
            datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 提取post_id从URL
        url = entry.link if hasattr(entry, 'link') else ''
        post_id = ''
        if url:
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 2:
                post_id = path_parts[-1] if path_parts[-1] else path_parts[-2]

        # 提取内容摘要
        content = ''
        if hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description

        # 清理HTML标签
        import re
        content = re.sub(r'<[^>]+>', '', content)
        content = content.strip()[:1000] if content else ''

        # 提取作者
        author = ''
        if hasattr(entry, 'author'):
            author = entry.author

        return {
            'post_id': post_id,
            'datetime': datetime_str,
            'title': entry.title if hasattr(entry, 'title') else '',
            'content': content,
            'subreddit': subreddit,
            'author': author,
            'url': url,
            'source': self.SOURCE
        }

    def test_connection(self):
        """测试连接"""
        try:
            subreddits = self.list_subreddits()
            if len(subreddits) == 1:
                rss_url = f"https://www.reddit.com/r/{subreddits[0]}/new.rss?limit={RSS_LIMIT}"
            else:
                query_str = _to_query_string(subreddits)
                rss_url = f"https://www.reddit.com/r/{query_str}/new.rss?limit={RSS_LIMIT}"
            feed = feedparser.parse(rss_url)
            if hasattr(feed, 'status'):
                return feed.status == 200
            return len(feed.entries) > 0
        except Exception as e:
            print(f"[Reddit] 连接测试失败: {e}")
            return False
