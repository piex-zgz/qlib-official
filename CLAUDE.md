# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qlib is Microsoft's AI-oriented quantitative investment platform. It provides:
- **Data Layer**: High-performance storage and retrieval of financial data (daily/minute-level OHLCV)
- **ML Models**: Supervised learning (LightGBM, XGBoost, LSTM, Transformer, etc.) for alpha seeking
- **Reinforcement Learning**: Trading agents for order execution and portfolio optimization
- **Backtesting**: High-performance backtesting engine with execution simulation
- **Workflow**: End-to-end experiment management with MLflow integration

## Architecture

```
qlib/
├── data/          # Data layer - storage, cache, dataset, feature operators (ops.py)
├── model/         # ML models - base trainer, ensemble methods, interpretability
├── workflow/      # Experiment management - recorder, MLflow exp_manager
├── strategy/      # Trading strategies based on model predictions
├── backtest/      # Backtesting framework - executor, exchange, account, report
├── rl/            # Reinforcement learning - simulator, trainer, strategies
├── contrib/       # Community-contributed components (data handlers, evaluators)
└── utils/         # Shared utilities
```

Key entry points:
- `qlib.init()` - Initialize qlib with config (data paths, etc.)
- `qlib.data.D` - Data API for calendar, instruments, features
- `qrun` CLI command - Run automated research workflows from YAML config

## Development Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install Cython modules (required before running tests)
make prerequisite

# Run tests (from tests/ directory)
cd tests && python -m pytest . -m "not slow"

# Run a single test
python -m pytest tests/test_workflow.py -v

# Download sample data
python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn

# Run a model workflow
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

# Run linting (black, pylint, flake8, mypy, nbqa)
make lint

# Check black formatting only
make black

# Build documentation
make docs-gen
```

## My Quant Project (`examples/my-quant/`)

自研量化投资策略框架，基于 qlib + akshare。

### 项目定位
- 独立于 qlib 主仓库的个人量化项目
- 完整的数据获取→训练→回测→可视化流程
- 支持多种模型：LightGBM、XGBoost、Linear、CatBoost、LSTM、GRU

### 核心文件
| 文件 | 功能 |
|------|------|
| `train.py` | 主入口，支持 `train`、`dump_bin`、`plot`、`create_config` 命令 |
| `live_backtest.py` | 实盘模拟持续回测系统，支持 `init`、`daily`、`simulate` 命令 |
| `get_data.py` | 用 akshare 下载 A 股日线数据（后复权） |
| `normalize.py` | 对齐交易日历、填充停牌数据 |
| `plot_utils.py` | 可视化模块，生成累计收益/IC分析/交易详情等图表 |
| `config.yaml` | 配置文件 |

### 工作流程
```
下载数据 → 规范化 → dump_bin → 训练回测 → 可视化
```

### 关键配置 (`config.yaml`)
- `data.qlib_dir`: qlib 数据目录
- `train.model`: 模型类型（LightGBM/XGBoost/LSTM/GRU等）
- `train.segments`: 训练/验证/测试集时间划分
- `backtest.topk`: TopkDropout 策略参数

### 输出
- 图表保存在 `examples/my-quant/plots/<日期>/<实验名>/`
- 实验记录在 `data/qlib_data/mlruns/`
- 图表文件：`01_cumulative_returns.png`、`02_monthly_returns.png`、`03_ic_analysis.png`、`06_detailed_trades.png`、`06_trade_details.csv`、statistics.txt

### 常用命令

**训练回测流程：**
```bash
# 训练
python train.py train --config config.yaml

# 为已有实验重新生成图表
python train.py plot --experiment_name my-quant-LightGBM

# 切换模型
python train.py create_config --model GRU
```

**实盘模拟持续回测：**
```bash
# 1. 首次初始化（训练模型 + 初始化账户）
python live_backtest.py init

# 2. 每日运行（获取最新数据、预测、交易、生成图表）
python live_backtest.py daily

# 3. 补运行某一天
python live_backtest.py daily --date 2024-01-15

# 4. 历史模拟测试
python live_backtest.py simulate --start_date 2024-01-01 --end_date 2024-01-31
```

### 实盘模拟系统特性

**核心逻辑：**
- **T+1 交易规则**：在 T 日收盘后用 T 日数据预测，在 T+1 日执行交易
- **定期重新训练**：支持每日/每周/每月重新训练模型（通过 `config.yaml` 中的 `live_backtest.retrain_frequency` 配置）
- **状态持久化**：账户状态、持仓、历史收益保存在 `.live_backtest_state.json`
- **每日图表输出**：保存在 `plots/YYYY-MM-DD/live/` 目录

**配置项（`config.yaml`）：**
```yaml
live_backtest:
  enabled: true
  download_data: true          # 是否每天下载最新数据
  retrain_frequency: "weekly"  # 重新训练频率: daily/weekly/monthly/never
  initial_cash: 100000000      # 初始资金
  topk: 50                     # 每次选择的股票数量
  n_drop: 5                    # 调仓时换出股票数量
  state_file: .live_backtest_state.json
```

**已知问题修复记录（2026-02-08）：**
1. ✅ 修复 T+1 交易规则违反问题（原先用同日数据预测和交易）
2. ✅ 修复价格数据获取 bug（卖出股票时无法获取价格，导致虚假亏损）
3. ✅ 修复预测数据 MultiIndex 处理问题
4. ✅ 修复 JSON 状态文件数字类型转换问题
5. ✅ 添加 pending 交易列表持久化（数据未就绪时保存信号，待数据更新后计算收益）
6. ✅ 添加 `check_pending` 命令手动触发 pending 收益计算
7. ✅ 修复 JSON 序列化 tuple key 问题（MultiIndex 转 dict 时的 tuple key）

## Code Standards
