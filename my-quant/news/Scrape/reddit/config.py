# Reddit 新闻抓取配置

import os

# 数据目录 (绝对路径)
DATA_DIR = r"D:\Quant-qlib-official\data"

# 数据库路径
DB_PATH = os.path.join(DATA_DIR, "reddit_posts.db")

# RSS URL: https://www.reddit.com/r/LocalLlaMA+OpenAI+MachineLearning/new.rss
# 板块用 + 连接，也可以传入列表，会循环抓取

# 抓取配置
DEFAULT_SUBREDDITS = "LocalLlaMA+OpenAI+MachineLearning"  # 单个字符串
# DEFAULT_SUBREDDITS = ["LocalLlaMA", "OpenAI", "MachineLearning"]  # 或列表，每个板块间隔1分钟
SUBREDDIT_INTERVAL = 60  # 多板块时，每个板块之间等待秒数
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
REQUEST_INTERVAL = 2
DEFAULT_LIMIT = 50
RSS_LIMIT = 100  # Reddit RSS API 最大支持约100条

# 增量抓取配置
INCREMENT_HOURS = 24  # 首次抓取时获取过去多少小时内的帖子
INCREMENT_LIMIT = 500  # 增量抓取时获取的最大帖子数量

# 定时监控
MONITOR_INTERVAL_MINUTES = 30
MONITOR_ENABLED = False

# 日志
LOG_LEVEL = "INFO"
