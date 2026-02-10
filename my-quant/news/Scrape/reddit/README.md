# Reddit News Scraper

基于 RSS 的 Reddit 新闻抓取工具，支持多板块循环抓取、定时调度和 SQLite 存储。

## 功能特性

- **RSS 抓取**: 使用 Reddit RSS API 获取最新帖子
- **多板块支持**: 支持单个或多个板块，可配置循环抓取间隔
- **增量更新**: 自动跳过已存在的帖子（MD5 去重）
- **定时调度**: 支持后台定时抓取（APScheduler）
- **数据统计**: 支持按板块、日期查询和统计

## 安装依赖

```bash
pip install feedparser apscheduler fire
```

## 快速开始

```bash
# 抓取最新帖子
python client.py fetch

# 增量获取新帖子（默认获取过去24小时）
python client.py increment

# 定时监控（每30分钟）
python client.py monitor --interval 30

# 查询帖子
python client.py query --subreddit LocalLlaMA --keyword AI --limit 20

# 查看数据库状态
python client.py status
```

## 命令行参数

### fetch - 抓取最新帖子

```bash
python client.py fetch --limit 50 --subreddits "LocalLlaMA+OpenAI"
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --limit | 抓取数量 | 50 |
| --subreddits | 板块（+连接或列表） | config.py中配置 |

### increment - 增量获取

```bash
python client.py increment --hours 24 --subreddits "LocalLlaMA+OpenAI"
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --hours | 获取过去N小时 | 24 |
| --subreddits | 板块 | config.py中配置 |

### monitor - 定时监控

```bash
python client.py monitor --interval 15 --once
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --interval | 抓取间隔（分钟） | 30 |
| --once | 只执行一次 | False |

### query - 查询帖子

```bash
python client.py query --subreddit LocalLlaMA --keyword "LLM" --start_date 2024-01-01 --limit 20
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --subreddit | 板块名称 | 全部 |
| --keyword | 关键词 | 全部 |
| --start_date | 开始日期 | 全部 |
| --end_date | 结束日期 | 全部 |
| --limit | 返回数量 | 20 |

## 配置文件 (config.py)

```python
# 数据目录
DATA_DIR = r"D:\Quant-qlib-official\data"

# 数据库路径
DB_PATH = os.path.join(DATA_DIR, "reddit_posts.db")

# 默认板块（字符串或列表）
DEFAULT_SUBREDDITS = "LocalLlaMA+OpenAI+MachineLearning"
# DEFAULT_SUBREDDITS = ["LocalLlaMA", "OpenAI", "MachineLearning"]

# 多板块抓取间隔（秒）
SUBREDDIT_INTERVAL = 60

# 抓取配置
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
REQUEST_INTERVAL = 2
DEFAULT_LIMIT = 50
RSS_LIMIT = 100  # RSS API 最大约100条

# 增量抓取
INCREMENT_HOURS = 24  # 首次获取过去24小时
INCREMENT_LIMIT = 500 # 增量抓取最大数量

# 定时监控
MONITOR_INTERVAL_MINUTES = 30
MONITOR_ENABLED = False
```

## 数据库结构

```sql
CREATE TABLE reddit_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,      -- MD5(source:post_id:subreddit:datetime)
    post_id TEXT NOT NULL,          -- Reddit 帖子ID
    datetime TEXT NOT NULL,         -- 发布时间
    title TEXT NOT NULL,            -- 帖子标题
    content TEXT,                   -- 帖子摘要
    subreddit TEXT NOT NULL,        -- 所属板块（抓取时使用的板块）
    author TEXT,                    -- 作者
    url TEXT NOT NULL,              -- 帖子链接
    fetch_time TEXT NOT NULL,       -- 抓取时间
    date_only TEXT NOT NULL,        -- 日期分区
    created_at TEXT NOT NULL        -- 创建时间
);

-- 索引
CREATE INDEX idx_reddit_date ON reddit_posts(date_only);
CREATE INDEX idx_reddit_subreddit ON reddit_posts(subreddit);
CREATE INDEX idx_reddit_datetime ON reddit_posts(datetime);
CREATE INDEX idx_reddit_uuid ON reddit_posts(uuid);
```

## 多板块说明

当配置多个板块时（使用 `+` 连接或列表）：

1. **字符串格式**: `"LocalLlaMA+OpenAI+MachineLearning"` - API 会同时返回所有板块的帖子
2. **列表格式**: `["LocalLlaMA", "OpenAI", "MachineLearning"]` - 会逐个板块分别抓取，每个板块间隔 `SUBREDDIT_INTERVAL` 秒

列表格式的优势：
- 可以独立追踪每个板块的抓取状态
- 更灵活的错误处理（一个板块失败不影响其他板块）
- 便于按板块进行增量更新

## 数据存储

- 数据库位置: `D:\Quant-qlib-official\data\reddit_posts.db`
- 数据文件体积小，适合长期积累

## License

MIT
