# News Agent 计划与进度（更新于 2026-02-12）

## 目标
构建一个可配置的 News Agent 数据准备层：
1. 从 Reddit 数据库读取内容
2. 按 flair 关键词做筛选（当前：new/news 模糊匹配）
3. 打包成下游可消费 JSON
4. 先按“原始路由分组”分流，再保留“URL 真实 subreddit”用于精细逻辑

## 已完成
- [x] 删除过时历史板块 `LocalLlaMA+OpenAI+MachineLearning` 数据
- [x] 完成 flair 统计脚本并输出分 subreddit 图表
- [x] 新增总控配置：`my-quant/news/news_tool_controller.json`
- [x] 新增构建工具：`my-quant/news/tools/build_reddit_payload.py`
- [x] 支持 flair 模糊匹配（contains_any）
- [x] 支持 URL 提取真实 subreddit（如 `LocalLLaMA`）
- [x] 支持按原始 `subreddit` 分流字段（`route_subreddit`）

## 当前产物
- 配置文件：`my-quant/news/news_tool_controller.json`
- 工具脚本：`my-quant/news/tools/build_reddit_payload.py`
- 输出文件：`data/static/reddit/reddit_news_payload.json`

## 当前规则（核心）
- 关键词：`["news", "new"]`
- 匹配方式：模糊匹配（大小写不敏感）
- `subreddit` 字段：URL/permalink 提取的真实板块
- `route_subreddit` 字段：原始抓取路由分组（用于后续接入不同逻辑）

## 下一步
1. 增加“增量导出”模式（按上次运行时间过滤）
2. 增加“黑白名单路由规则”（按 route_subreddit 分派不同处理器）
3. 给下游 Agent 增加统一消费入口（读取 payload -> 分流 -> 执行）
4. 增加基础测试（配置解析、SQL 筛选、JSON 结构）

## 常用命令
- Dry run：`python my-quant/news/tools/build_reddit_payload.py --dry-run`
- 正式导出：`python my-quant/news/tools/build_reddit_payload.py`
- flair 图表：`python my-quant/news/Scrape/reddit/analyze_flair_distribution.py --top-n 12`
