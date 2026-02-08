#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实盘模拟持续回测脚本

功能：
- 定期重新训练模型（每周/每月，不是每天）
- 每天执行虚拟交易并更新持仓
- 每天保存收益图表

使用方法：
    # 首次初始化
    python live_backtest.py init

    # 每日运行
    python live_backtest.py daily

    # 补运行某一天
    python live_backtest.py daily --date 2024-01-15

    # 历史模拟（测试用）
    python live_backtest.py simulate --start 2024-01-01 --end 2024-01-31
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from loguru import logger

import yaml
import fire
import qlib
from qlib.constant import REG_CN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LiveBacktest:
    """实盘模拟持续回测类"""

    def __init__(self, config_path: str = None):
        """
        初始化

        Args:
            config_path: 配置文件路径
        """
        # 设置基础路径
        self.base_dir = Path(__file__).parent

        # 加载配置
        if config_path is None:
            config_path = self.base_dir / "config.yaml"
        else:
            config_path = Path(config_path)

        self.config = self._load_config(config_path)
        self.config_path = config_path

        # 状态文件路径
        state_file = self.config.get('live_backtest', {}).get(
            'state_file', '.live_backtest_state.json'
        )
        self.state_file = self.base_dir / state_file

        # 初始化日志
        self._setup_logger()

        # 初始化 qlib
        qlib_dir = Path(self.config['data']['qlib_dir']).expanduser()
        logger.info(f"初始化 qlib，数据目录: {qlib_dir}")
        qlib.init(provider_uri=str(qlib_dir), region=REG_CN)

    def _setup_logger(self):
        """设置日志"""
        import sys
        log_file = self.base_dir / "logs" / f"live_backtest_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(str(log_file), rotation="1 day", encoding="utf-8")
        logger.add(sys.stdout, level="INFO")

    def _load_config(self, config_path: Path) -> dict:
        """加载配置文件"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _load_state(self) -> dict:
        """加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # 将字符串转换回数字类型
            self._convert_state_types(state)
            return state
        return None

    def _convert_state_types(self, state: dict):
        """将状态中的字符串转换回数字类型，处理版本迁移"""
        # 1. 迁移旧版本：添加 last_data_update_date 字段
        if 'last_data_update_date' not in state:
            logger.info("检测到旧版本状态文件，添加 last_data_update_date 字段...")
            # 从 qlib 数据库获取最后日期
            try:
                from qlib.data import D
                calendar = D.calendar(freq='day')
                state['last_data_update_date'] = calendar[-1].strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"无法从 qlib 获取日期: {e}，使用 last_update_date")
                state['last_data_update_date'] = state.get('last_update_date', '2024-01-01')
            logger.info(f"迁移后 last_data_update_date = {state['last_data_update_date']}")

        # 2. 转换数字类型
        if 'account' in state:
            account = state['account']
            if 'cash' in account and isinstance(account['cash'], str):
                account['cash'] = float(account['cash'])
            if 'total_value' in account and isinstance(account['total_value'], str):
                account['total_value'] = float(account['total_value'])

        if 'history' in state:
            history = state['history']
            if 'returns' in history:
                history['returns'] = [float(r) if isinstance(r, str) else r for r in history['returns']]
            if 'total_values' in history:
                history['total_values'] = [float(v) if isinstance(v, str) else v for v in history['total_values']]

        # 3. 迁移旧版本：添加 pending_trades 字段
        if 'pending_trades' not in state:
            logger.info("检测到旧版本状态文件，添加 pending_trades 字段...")
            state['pending_trades'] = []
            logger.info("pending_trades 字段已初始化为空列表")

        # 4. 转换 predictions 为 pandas Series（如果存储为 dict）
        # 注意：predictions 现在不持久化，只有 pending_trades 中的 pred_df 需要处理
        if 'predictions' in state:
            pred = state['predictions']
            if isinstance(pred, dict):
                try:
                    state['predictions'] = pd.Series(pred)
                    logger.debug("predictions 已从 dict 恢复为 pandas Series")
                except Exception as e:
                    logger.warning(f"恢复 predictions 失败: {e}")
                    del state['predictions']

    def _save_state(self, state: dict):
        """保存状态"""
        import copy

        # 创建深拷贝，避免修改原对象
        state_copy = copy.deepcopy(state)

        # 递归处理所有 pandas 对象和 tuple key
        state_copy = self._process_for_json(state_copy)

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state_copy, f, ensure_ascii=False, indent=2)
        logger.info(f"状态已保存到: {self.state_file}")

    def _process_for_json(self, obj):
        """递归处理对象，确保可以 JSON 序列化"""
        import copy

        obj_copy = copy.deepcopy(obj)

        if isinstance(obj_copy, pd.Series):
            # Series -> dict
            return obj_copy.to_dict()

        elif isinstance(obj_copy, pd.DataFrame):
            # DataFrame -> dict
            return obj_copy.to_dict(orient='index')

        elif isinstance(obj_copy, dict):
            # 递归处理 dict
            return {k: self._process_for_json(v) for k, v in obj_copy.items()}

        elif isinstance(obj_copy, (list, tuple)):
            # 递归处理 list/tuple
            return [self._process_for_json(item) for item in obj_copy]

        else:
            # 原始类型，直接返回
            return obj_copy

    def _get_latest_trading_day(self) -> str:
        """获取最新的交易日（从 akshare 实时获取）"""
        import akshare as ak

        try:
            # 从 akshare 获取交易日历
            logger.debug("从 akshare 获取交易日历...")
            calendar_df = ak.tool_trade_date_hist_sina()
            calendar_df['trade_date'] = pd.to_datetime(calendar_df['trade_date'])

            # 只取今天及之前的交易日
            today = datetime.now().date()
            valid_dates = calendar_df[calendar_df['trade_date'].dt.date <= today]

            if len(valid_dates) == 0:
                raise ValueError("没有找到有效的交易日")

            latest = valid_dates['trade_date'].max()
            latest_str = latest.strftime('%Y-%m-%d')
            logger.debug(f"最新交易日: {latest_str}")
            return latest_str

        except Exception as e:
            logger.warning(f"从 akshare 获取交易日历失败: {e}，回退到本地 qlib 数据")
            # 降级方案：使用本地 qlib 数据
            from qlib.data import D
            calendar = D.calendar(freq='day')
            if len(calendar) == 0:
                raise ValueError("无法获取交易日历，请检查数据是否已准备")
            return calendar[-1].strftime('%Y-%m-%d')

    def _normalize_date(self, date_str: str) -> str:
        """
        规范化日期格式为 YYYY-MM-DD

        Args:
            date_str: 输入日期，支持 YYYYMMDD 或 YYYY-MM-DD 格式

        Returns:
            规范化的 YYYY-MM-DD 格式日期
        """
        if date_str is None:
            return None

        date_str = str(date_str).strip()

        # 如果是 YYYYMMDD 格式
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # 如果是 YYYY-MM-DD 格式，直接返回
        if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str

        # 其他格式，尝试使用 pandas 解析
        try:
            return pd.Timestamp(date_str).strftime('%Y-%m-%d')
        except Exception:
            logger.warning(f"无法解析日期格式: {date_str}")
            return None

    def _get_local_data_latest_date(self) -> str:
        """
        检查本地数据的最后日期

        Returns:
            str: 本地数据的最后日期（YYYY-MM-DD 格式），如果没有数据返回 None
        """
        data_dir = Path(self.config.get('data', {}).get('save_dir', 'data/source'))

        if not data_dir.exists():
            return None

        # 获取一个股票文件来检查日期（使用 SH000300 指数文件更可靠）
        benchmark_file = data_dir / "SH000300.csv"
        if benchmark_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(benchmark_file)
                if 'date' in df.columns:
                    latest = df['date'].max()
                    logger.info(f"本地数据最后日期: {latest}")
                    return latest
            except Exception as e:
                logger.warning(f"读取本地数据日期失败: {e}")

        # 或者检查任意一个股票文件
        csv_files = list(data_dir.glob("SH*.csv")) + list(data_dir.glob("SZ*.csv"))
        if csv_files:
            try:
                import pandas as pd
                sample_file = csv_files[0]
                df = pd.read_csv(sample_file, usecols=['date'])
                latest = df['date'].max()
                logger.info(f"本地数据最后日期（来自 {sample_file.name}）: {latest}")
                return latest
            except:
                pass

        return None

    def _update_data(self, target_date: str = None, force_full: bool = False) -> bool:
        """
        自动更新数据（股票 + 指数）

        Args:
            target_date: 更新到这个日期（格式：YYYY-MM-DD 或 YYYYMMDD），默认为今天
            force_full: 是否强制完整下载（跳过本地数据检查）

        Returns:
            bool: 更新是否成功
        """
        import subprocess
        import sys

        try:
            # 1. 确定目标日期
            data_config = self.config.get('data', {})

            if target_date is None:
                target_date = self._get_latest_trading_day()
            else:
                target_date = self._normalize_date(target_date)

            # 转换为 YYYYMMDD 格式（用于下载脚本）
            target_date_num = target_date.replace('-', '')

            # 2. 确定开始下载的日期
            if not force_full:
                # 增量模式：从上次更新日期开始
                last_data_date = self.state.get('last_data_update_date')
                if last_data_date is None:
                    # 如果状态文件没有，尝试从本地数据获取
                    last_data_date = self._get_local_data_latest_date()
                    if last_data_date:
                        logger.info(f"从本地数据获取到上次更新日期: {last_data_date}")
                    else:
                        last_data_date = '20240101'
                        logger.warning("无法获取上次更新日期，使用默认值")

                start_date_num = last_data_date.replace('-', '')
                logger.info(f"增量数据更新: {start_date_num} → {target_date_num}")

                # 如果已经是最新，跳过
                if start_date_num >= target_date_num:
                    logger.info(f"数据已是最新（{last_data_date}），无需更新")
                    return True
            else:
                # 完整模式：先检查本地数据
                local_date = self._get_local_data_latest_date()
                if local_date:
                    # 本地有数据，从本地最后日期开始
                    start_date_num = local_date.replace('-', '')
                    logger.info(f"完整模式：本地数据最新日期为 {local_date}")
                    logger.info(f"从 {local_date} 开始增量下载: {start_date_num} → {target_date_num}")

                    # 检查是否需要下载
                    if start_date_num >= target_date_num:
                        logger.info(f"数据已是最新（{local_date}），无需更新")
                        return True
                else:
                    # 本地没有数据，从配置的开始日期
                    start_date = str(data_config.get('start_date', '20080101'))
                    start_date_num = start_date
                    logger.info(f"完整数据下载（本地无数据）: {start_date} → {target_date_num}")

            # 2. 下载股票数据
            logger.info("步骤 1/5: 下载股票数据...")
            get_data_script = self.base_dir / "get_data.py"
            result = subprocess.run(
                [sys.executable, str(get_data_script), "download",
                 "--start_date", start_date_num,
                 "--end_date", target_date_num,
                 "--max_workers", "12"],
                check=False  # 不检查返回值，输出会直接显示
            )
            logger.info("股票数据下载完成")

            # 3. 下载指数数据
            logger.info("步骤 2/5: 下载指数数据...")
            benchmark_script = self.base_dir / "download_benchmark.py"
            subprocess.run(
                [sys.executable, str(benchmark_script)],
                check=False,
                env={**subprocess.os.environ,
                     "BENCHMARK_START_DATE": start_date_num,
                     "BENCHMARK_END_DATE": target_date_num}
            )
            logger.info("指数数据下载完成")

            # 3.5 更新 CSI300 成分股列表（使用 qlib 的 collector）
            logger.info("步骤 2.5/5: 更新 CSI300 成分股列表...")
            qlib_dir = Path(self.config['data']['qlib_dir']).expanduser()
            collector_script = qlib_dir.parent / "scripts" / "data_collector" / "cn_index" / "collector.py"
            if collector_script.exists():
                subprocess.run(
                    [sys.executable, str(collector_script),
                     "--index_name", "CSI300",
                     "--qlib_dir", str(qlib_dir),
                     "--method", "parse_instruments"],
                    check=False
                )
                logger.info("CSI300 成分股列表已更新")
            else:
                logger.warning(f"找不到 collector 脚本: {collector_script}")

            # 4. 规范化数据
            logger.info("步骤 3/5: 规范化数据...")
            normalize_script = self.base_dir / "normalize.py"
            subprocess.run(
                [sys.executable, str(normalize_script), "normalize"],
                check=False
            )
            logger.info("数据规范化完成")

            # 5. 更新 qlib 二进制数据
            logger.info("步骤 4/5: 更新 qlib 二进制数据...")
            train_script = self.base_dir / "train.py"
            subprocess.run(
                [sys.executable, str(train_script), "dump_bin"],
                check=False
            )
            logger.info("qlib 数据库更新完成")

            # 6. 更新状态
            logger.info("步骤 5/5: 更新状态...")
            self.state['last_data_update_date'] = target_date
            self._save_state(self.state)

            logger.info(f"数据更新成功！当前数据截止: {target_date}")

            # 7. 检查并计算 pending 收益
            logger.info("检查并计算待处理的 pending 收益...")
            calculated = self._calculate_pending_returns(target_date)
            if calculated > 0:
                logger.info(f"已计算 {calculated} 个 pending 交易的收益")
                self._save_state(self.state)

            return True

        except Exception as e:
            logger.error(f"数据更新异常: {e}")
            logger.error(f"完整堆栈跟踪:\n{traceback.format_exc()}")
            return False

    def _calculate_pending_returns(self, target_date: str) -> int:
        """
        计算待处理的 pending 收益

        当数据更新后，检查是否有 pending 的交易可以计算收益了

        Args:
            target_date: 更新后的数据截止日期（YYYY-MM-DD）

        Returns:
            int: 计算的 pending 交易数量
        """
        pending_trades = self.state.get('pending_trades', [])

        if not pending_trades:
            logger.debug("没有待计算的 pending 交易")
            return 0

        logger.info(f"检查 {len(pending_trades)} 个待计算的 pending 交易...")

        from qlib.data import D
        calendar = D.calendar(freq='day')
        calendar_dates = [d.strftime('%Y-%m-%d') for d in calendar]

        calculated_count = 0
        remaining_trades = []

        for pending in pending_trades:
            pred_date = pending.get('pred_date')  # 预测数据日期 (T)
            trade_date = pending.get('trade_date')  # 交易日期 (T+1)

            logger.info(f"检查 pending 交易: pred_date={pred_date}, trade_date={trade_date}")

            # 检查 T+1 的数据是否已更新
            if trade_date not in calendar_dates:
                logger.warning(f"交易日 {trade_date} 不在日历中，跳过")
                remaining_trades.append(pending)
                continue

            # 检查数据是否已更新到 trade_date 之后
            if target_date < trade_date:
                logger.info(f"数据尚未更新到 {trade_date}，等待...")
                remaining_trades.append(pending)
                continue

            # T+1 数据已可用，计算收益
            try:
                # 恢复 pred_df
                pred_df_dict = pending.get('pred_df', {})
                if pred_df_dict:
                    if isinstance(pred_df_dict, dict):
                        # 尝试从 predictions 恢复
                        if self.state.get('predictions') is not None:
                            pred_df = self.state['predictions']
                            if hasattr(pred_df.index, 'get_level_values'):
                                pred_dates = pred_df.index.get_level_values('datetime').unique()
                                pred_dates_str = [d.strftime('%Y-%m-%d') for d in pred_dates]
                                if pred_date in pred_dates_str:
                                    pred_df = pred_df.xs(pd.Timestamp(pred_date), level='datetime')
                        else:
                            # 需要从 pred_df_dict 恢复
                            try:
                                # pred_df_dict 是 {stock_code: score} 格式
                                # 需要转换为 MultiIndex 格式
                                pred_df = pd.Series(pred_df_dict)
                                # 转换为 MultiIndex 格式
                                pred_df = pd.DataFrame({'score': pred_df})
                                pred_df.index = pd.MultiIndex.from_product(
                                    [[pd.Timestamp(pred_date)], pred_df.index],
                                    names=['datetime', 'instrument']
                                )
                                pred_df = pred_df['score']
                            except Exception as e:
                                logger.warning(f"无法恢复 pred_df: {e}")
                                continue
                    else:
                        pred_df = pred_df_dict
                else:
                    # 使用当前预测
                    pred_df = self.state.get('predictions', pd.Series())
                    if not pred_df.empty and hasattr(pred_df.index, 'get_level_values'):
                        pred_dates = pred_df.index.get_level_values('datetime').unique()
                        pred_dates_str = [d.strftime('%Y-%m-%d') for d in pred_dates]
                        if pred_date in pred_dates_str:
                            # 保持 MultiIndex 格式
                            original_pred_df = pred_df
                            pred_df = original_pred_df.xs(pd.Timestamp(pred_date), level='datetime')
                            # 转换回 MultiIndex
                            pred_df = pd.DataFrame({'score': pred_df})
                            pred_df.index = pd.MultiIndex.from_product(
                                [[pd.Timestamp(pred_date)], pred_df.index],
                                names=['datetime', 'instrument']
                            )
                            pred_df = pred_df['score']

                if pred_df.empty:
                    logger.warning(f"pred_df 为空，无法计算收益")
                    remaining_trades.append(pending)
                    continue

                # 运行 qlib 回测来计算收益
                logger.info(f"[pending] 使用 qlib 计算 {trade_date} 的收益...")
                backtest_result = self._run_qlib_backtest_with_pred(pred_df, trade_date)

                if backtest_result and not backtest_result.get('pending', False):
                    # 计算收益并更新状态
                    self._apply_backtest_result(trade_date, backtest_result)
                    calculated_count += 1
                    logger.info(f"[pending] {trade_date} 收益计算完成")
                else:
                    remaining_trades.append(pending)

            except Exception as e:
                logger.error(f"计算 pending 收益失败: {e}")
                import traceback
                traceback.print_exc()
                remaining_trades.append(pending)

        # 更新 pending_trades 列表
        self.state['pending_trades'] = remaining_trades
        if remaining_trades:
            logger.info(f"仍有 {len(remaining_trades)} 个 pending 交易等待数据更新")

        return calculated_count

    def _run_qlib_backtest_with_pred(self, pred_df: pd.Series, date: str) -> dict:
        """
        使用指定的预测数据运行 qlib 单日回测

        Args:
            pred_df: 预测数据 Series
            date: 交易日期

        Returns:
            dict: 回测结果
        """
        from qlib.contrib.evaluate import backtest_daily
        from qlib.contrib.strategy import TopkDropoutStrategy
        from qlib.backtest.position import Position

        # 获取当前持仓
        positions = self.state['account'].get('positions', {})
        current_holdings = {s: p for s, p in positions.items() if isinstance(p, dict) and p.get('shares', 0) > 0}

        # 初始化策略
        backtest_config = self.config.get('backtest', {})
        live_config = self.config.get('live_backtest', {})

        STRATEGY_CONFIG = {
            "topk": live_config.get('topk', 10),
            "n_drop": live_config.get('n_drop', 5),
            "signal": pred_df,
        }

        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

        # 初始化账户
        initial_cash = backtest_config.get('account', 100000000)
        account_value = self.state['account'].get('total_value', initial_cash)

        if not current_holdings:
            account = account_value
        else:
            account = Position()
            account.reset_stock()
            for stock, pos_info in current_holdings.items():
                shares = pos_info.get('shares', 0)
                if shares > 0:
                    account.update_pos(stock, shares)

        # 运行回测
        exchange_kwargs = backtest_config.get('exchange_kwargs', {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        })

        logger.info(f"[qlib回测] 运行回测: {date}")
        report_df, positions_df = backtest_daily(
            start_time=date,
            end_time=date,
            strategy=strategy_obj,
            account=account,
            benchmark=backtest_config.get('benchmark', 'SH000300'),
            exchange_kwargs=exchange_kwargs,
        )

        # 提取结果
        result = {
            'date': date,
            'trades': [],
            'pending': False,
            'portfolio': {
                'cash': self.state['account'].get('cash', 0),
                'total_value': account_value,
                'positions': current_holdings,
            }
        }

        try:
            if report_df is not None and not report_df.empty:
                if 'total_value' in report_df.columns:
                    result['portfolio']['total_value'] = float(report_df['total_value'].iloc[-1])
                if 'cash' in report_df.columns:
                    result['portfolio']['cash'] = float(report_df['cash'].iloc[-1])

            if positions_df is not None and not positions_df.empty:
                latest_date = list(positions_df.keys())[-1]
                if latest_date in positions_df:
                    new_positions = positions_df[latest_date]
                    if hasattr(new_positions, 'get_stock_list'):
                        stock_list = new_positions.get_stock_list()
                        new_holdings = {}
                        for stock in stock_list:
                            shares = new_positions.get_stock_amount(stock)
                            if shares > 0:
                                new_holdings[stock] = {'shares': shares}
                        result['portfolio']['positions'] = new_holdings

        except Exception as e:
            logger.warning(f"提取回测结果失败: {e}")

        return result

    def _apply_backtest_result(self, date: str, backtest_result: dict):
        """
        应用回测结果到状态

        Args:
            date: 交易日期
            backtest_result: 回测结果
        """
        portfolio = backtest_result.get('portfolio', {})

        # 更新账户
        self.state['account']['cash'] = portfolio.get('cash', self.state['account']['cash'])
        self.state['account']['total_value'] = portfolio.get('total_value', self.state['account']['total_value'])
        self.state['account']['positions'] = portfolio.get('positions', self.state['account']['positions'])

        # 计算收益
        history = self.state['history']
        prev_total = history['total_values'][-1] if history['total_values'] else self.config['backtest']['account']
        current_total = self.state['account']['total_value']

        daily_return = (current_total - prev_total) / prev_total

        # 更新历史
        history['dates'].append(date)
        history['returns'].append(daily_return)
        history['total_values'].append(current_total)
        self.state['history'] = history
        self.state['last_update_date'] = date

        logger.info(f"[pending] 收益已应用: date={date}, return={daily_return*100:.4f}%, "
                   f"total_value={current_total:,.0f}")

    def _build_handler_config(self, end_date: str = None) -> dict:
        """构建 handler 配置"""
        data_config = self.config.get('data', {})
        train_config = self.config.get('train', {})

        start_date = str(data_config.get('start_date', '20080101'))
        start_time = self._normalize_date(start_date)

        if end_date is None:
            end_date = str(data_config.get('end_date', '20250101'))
        end_time = self._normalize_date(end_date)

        return {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": train_config['segments']['train'][0],
            "fit_end_time": train_config['segments']['train'][1],
            "instruments": train_config.get('instruments', 'csi300'),
        }

    def _build_dataset_config(self, handler_config: dict, train_config: dict = None) -> dict:
        """构建数据集配置"""
        if train_config is None:
            train_config = self.config.get('train', {})

        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": handler_config,
                },
                "segments": {
                    "train": train_config['segments']['train'],
                    "valid": train_config['segments']['valid'],
                    "test": train_config['segments']['test'],
                },
            },
        }

        # 对于时序模型(LSTM, GRU等)
        model_name = train_config.get('model', '').lower()
        if model_name in ['lstm', 'gru']:
            dataset_config["class"] = "TSDatasetH"
            dataset_config["kwargs"]["step_len"] = train_config.get('step_len', 20)

        return dataset_config

    def _build_model_config(self, train_config: dict = None) -> dict:
        """构建模型配置"""
        if train_config is None:
            train_config = self.config.get('train', {})

        model_name = train_config.get('model', 'LightGBM')

        model_configs = {
            'LightGBM': {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.02,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                },
            },
            'XGBoost': {
                "class": "XGBModel",
                "module_path": "qlib.contrib.model.xgboost",
                "kwargs": {
                    "eval_metric": "rmse",
                    "colsample_bytree": 0.8879,
                    "eta": 0.0421,
                    "max_depth": 8,
                    "n_estimators": 647,
                    "subsample": 0.8789,
                    "nthread": 20,
                },
            },
            'Linear': {
                "class": "LinearModel",
                "module_path": "qlib.contrib.model.linear",
                "kwargs": {"reg": 0.0001},
            },
            'CatBoost': {
                "class": "CatBoostModel",
                "module_path": "qlib.contrib.model.catboost",
                "kwargs": {
                    "loss_function": "RMSE",
                    "iterations": 500,
                    "learning_rate": 0.03,
                    "depth": 6,
                    "l2_leaf_reg": 3,
                    "random_seed": 42,
                    "thread_count": 20,
                },
            },
            'GRU': {
                "class": "GRU",
                "module_path": "qlib.contrib.model.pytorch_gru",
                "kwargs": {
                    "d_feat": 20,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 200,
                    "lr": 1e-3,
                    "early_stop": 10,
                    "batch_size": 800,
                    "metric": "loss",
                    "loss": "mse",
                    "n_jobs": 20,
                    "GPU": 0,
                },
            },
            'LSTM': {
                "class": "LSTM",
                "module_path": "qlib.contrib.model.pytorch_lstm_ts",
                "kwargs": {
                    "d_feat": 20,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 200,
                    "lr": 1e-3,
                    "early_stop": 10,
                    "batch_size": 800,
                    "metric": "loss",
                    "loss": "mse",
                    "n_jobs": 20,
                    "GPU": 0,
                },
            },
        }

        if 'model_config' in train_config:
            return train_config['model_config']

        # 处理模型名称映射（兼容多种命名方式）
        model_name_lower = model_name.lower()
        model_name_mapping = {
            'lightgbm': 'LightGBM',
            'lgbm': 'LightGBM',
            'xgboost': 'XGBoost',
            'xgb': 'XGBoost',
            'linear': 'Linear',
            'lstm': 'LSTM',
            'gru': 'GRU',
            'pytorch_gru': 'GRU',
            'pytorch_lstm': 'LSTM',
            'catboost': 'CatBoost',
            'catboost_model': 'CatBoost',
        }

        model_key = model_name_mapping.get(model_name_lower, 'LightGBM')
        if model_key not in model_configs:
            logger.warning(f"未知模型名称: {model_name}, 使用 LightGBM")
            model_key = 'LightGBM'

        return model_configs[model_key]

    def init(self, download_data: bool = True):
        """首次初始化

        Args:
            download_data: 是否自动下载最新数据，默认 True
        """
        logger.info("=" * 60)
        logger.info("开始初始化实盘模拟系统")
        logger.info("=" * 60)

        # 初始化 state（必须在调用 _update_data 之前）
        self.state = {}

        # 删除历史状态文件（init 是首次初始化，需要全新开始）
        if self.state_file.exists():
            logger.info(f"删除历史状态文件: {self.state_file}")
            self.state_file.unlink()

        # 0. 下载最新数据（新增）
        target_date = None
        if download_data:
            logger.info("步骤 0/3: 下载最新数据...")
            target_date = self._get_latest_trading_day()
            logger.info(f"目标日期: {target_date}")

            success = self._update_data(target_date, force_full=True)
            if not success:
                logger.warning("数据下载失败，继续使用已有数据")
                target_date = None
        else:
            logger.info("跳过数据下载（download_data=False）")

        # 1. 训练初始模型
        logger.info("步骤 1/3: 训练初始模型...")
        result = self.train_model()

        # 如果 train_model 返回了路径和日期，使用它们
        if result is not None:
            model_path, last_train_date = result
        else:
            # 如果 self.state 已更新，使用其中的值
            if hasattr(self, 'state') and self.state is not None:
                model_path = self.state.get('model_path')
                last_train_date = self.state.get('last_train_date')
            else:
                model_path = self._get_latest_model_path()
                last_train_date = datetime.now().strftime('%Y-%m-%d')

        # 2. 初始化账户状态
        logger.info("步骤 2/3: 初始化账户...")
        backtest_config = self.config.get('backtest', {})
        initial_cash = backtest_config.get('account', 100000000)

        # 如果没有下载数据，尝试从 qlib 获取最后日期
        if target_date is None:
            try:
                from qlib.data import D
                calendar = D.calendar(freq='day')
                target_date = calendar[-1].strftime('%Y-%m-%d')
            except:
                target_date = datetime.now().strftime('%Y-%m-%d')

        state = {
            'last_update_date': None,
            'last_data_update_date': target_date,  # 新增字段
            'last_train_date': last_train_date,
            'account': {
                'cash': initial_cash,
                'total_value': initial_cash,
                'positions': {}
            },
            'history': {
                'dates': [],
                'returns': [],
                'total_values': []
            },
            'model_path': model_path,
            'pending_trades': [],  # 待计算收益的交易列表
        }

        self.state = state
        self._save_state(state)

        logger.info("步骤 3/3: 完成初始化")
        logger.info("=" * 60)
        logger.info("初始化完成！")
        logger.info(f"初始资金: {initial_cash:,.0f}")
        logger.info(f"模型路径: {model_path}")
        logger.info("下一步：运行 python live_backtest.py daily 开始每日回测")
        logger.info("=" * 60)

    def _get_latest_model_path(self) -> str:
        """获取最新的模型路径"""
        model_dir = self.base_dir / "models"
        if not model_dir.exists():
            return None

        models = list(model_dir.glob("model_*.pkl"))
        if not models:
            return None

        latest = max(models, key=lambda x: x.stat().st_mtime)
        return str(latest)

    def daily(self, date: str = None, state: dict = None):
        """
        每日运行流程

        注意：这里实现的是 T+1 交易逻辑
        - 在 T 日收盘后，用 T 日数据预测明日（T+1）哪些股票会涨
        - 在 T+1 日开盘时买入，用 T+1 日的价格执行交易
        - 在 T+1 日收盘后计算收益

        Args:
            date: 交易日期（T+1日），默认为最新交易日
            state: 外部传入的状态，如果为 None 则从文件加载
        """
        # 0. 确定运行日期
        if date is None:
            date = self._get_latest_trading_day()
        else:
            # 使用规范化方法处理日期
            normalized = self._normalize_date(date)
            if normalized is None:
                logger.error(f"日期格式错误: {date}")
                return
            date = normalized

        logger.info("=" * 60)
        logger.info(f"[{date}] 开始每日回测")
        logger.info("=" * 60)

        # 0.1 加载状态
        if state is not None:
            # 使用外部传入的状态
            self.state = state
        else:
            self.state = self._load_state()

        if self.state is None:
            logger.error("未找到状态文件，请先运行 init 初始化")
            return

        # 0.2 检查是否已经处理过这个日期（防止重复）
        history = self.state.get('history', {})
        if date in history.get('dates', []):
            logger.warning(f"⚠️  日期 {date} 已经处理过，跳过")
            logger.info(f"如需重新处理，请先删除状态文件或手动编辑历史记录")
            return

        # 0.3 检查并更新数据（新增）
        live_config = self.config.get('live_backtest', {})
        if live_config.get('download_data', False):
            logger.info("检查数据更新...")
            last_data_date = self.state.get('last_data_update_date')

            if last_data_date is None:
                logger.warning("状态文件中没有 last_data_update_date，尝试从 qlib 获取")
                from qlib.data import D
                try:
                    calendar = D.calendar(freq='day')
                    last_data_date = calendar[-1].strftime('%Y-%m-%d')
                    self.state['last_data_update_date'] = last_data_date
                except:
                    last_data_date = '2024-01-01'

            # 比较日期
            if date > last_data_date:
                logger.info(f"需要更新数据: {last_data_date} → {date}")
                success = self._update_data(date, force_full=False)
                if not success:
                    logger.warning("数据更新失败，继续使用已有数据")
            elif date == last_data_date:
                logger.info(f"✅ 数据已是最新（{last_data_date}），无需更新")
            else:
                logger.info(f"目标日期 {date} 早于数据更新日期 {last_data_date}，无需更新")
        else:
            logger.info("自动数据更新已禁用（config.live_backtest.download_data=False）")

        logger.info(f"当前账户: 现金={self.state['account']['cash']:,.0f}, "
                   f"总资产={self.state['account']['total_value']:,.0f}")

        # 1. 检查是否需要重新训练
        logger.info("步骤 1/5: 检查是否需要重新训练...")
        if self.should_retrain(date):
            logger.info("需要重新训练，开始训练...")
            self.train_model()
        else:
            logger.info("使用现有模型")

        # 2. 确定预测数据日期和交易日期
        # 逻辑：
        # - date 参数是你想要用于预测的数据日期（T）
        # - trade_date 是 T 的下一个交易日（T+1），是实际执行交易的日期
        # - 如果 T+1 的数据已可用，则计算收益
        logger.info("步骤 2/5: 获取预测信号...")
        from qlib.data import D
        calendar = D.calendar(freq='day')
        calendar_dates = [d.strftime('%Y-%m-%d') for d in calendar]

        # 检查 date 是否在交易日历中
        if date not in calendar_dates:
            logger.error(f"[daily] 日期 {date} 不在交易日历中，无法进行回测")
            logger.info(f"可用的交易日历范围: {calendar_dates[0]} ~ {calendar_dates[-1]}")
            return

        # 找到 date 在日历中的位置，确定 T+1
        date_idx = list(calendar).index(pd.Timestamp(date))

        if date_idx >= len(calendar) - 1:
            # date 是最后一天，T+1 不存在
            logger.info(f"[daily] {date} 是交易日历的最后一天")
            logger.info(f"[daily] 使用 {date} 的数据预测下一交易日（等待 T+1 数据更新后计算收益）")
            trade_date = None
        else:
            # T+1 存在
            trade_date = calendar[date_idx + 1].strftime('%Y-%m-%d')
            logger.info(f"[daily] 预测数据日期: {date} (T)")
            logger.info(f"[daily] 交易/收益计算日期: {trade_date} (T+1)")

        # 使用 date 的数据生成预测
        predictions = self.predict(date)

        if predictions is None or predictions.empty:
            logger.warning("预测结果为空，跳过今日交易")
            return

        logger.info(f"预测股票数量: {len(predictions)}")
        topk = self.config.get('live_backtest', {}).get('topk', 10)
        logger.info(f"选择 Top-{topk} 股票作为交易信号")

        # 保存预测和预测日期
        self.state['predictions'] = predictions
        self.state['pred_date'] = date
        logger.info(f"预测已保存: 用 {date} (T) 数据预测")

        # 3. 检查 T+1 数据是否可用，决定是否计算收益
        if trade_date is not None:
            # T+1 在日历中，检查数据是否已更新到 T+1
            latest_qlib_date = self._get_latest_data_date_from_qlib()

            if latest_qlib_date and latest_qlib_date >= trade_date:
                # T+1 数据已可用，执行交易并计算收益
                logger.info(f"[daily] T+1 ({trade_date}) 数据已可用，执行交易...")
                logger.info(f"[daily] 使用 qlib 回测系统执行交易...")
                backtest_result = self._run_qlib_backtest(trade_date)

                if backtest_result and not backtest_result.get('pending', False):
                    # 成功计算收益
                    logger.info("步骤 3/5: 执行虚拟交易... (已完成)")
                    logger.info("步骤 4/5: 计算收益... (已完成)")
                    daily_return = self.calculate_return(trade_date, backtest_result)

                    if daily_return is not None:
                        logger.info(f"当日收益率: {daily_return*100:.4f}%")
                        total_return = (self.state['account']['total_value'] / self.config['backtest']['account'] - 1) * 100
                        logger.info(f"累计收益率: {total_return:.2f}%")
                else:
                    # pending 或失败
                    logger.info("步骤 3/5: 执行虚拟交易... (pending)")
                    logger.info("步骤 4/5: 计算收益... (pending)")
            else:
                # T+1 数据不可用，只保存信号
                logger.info(f"[daily] T+1 ({trade_date}) 数据尚未更新，保存信号等待")
                logger.info("步骤 3/5: 执行虚拟交易... (pending)")
                logger.info("步骤 4/5: 计算收益... (pending)")

                # 获取当前预测数据
                predictions = self.state.get('predictions', pd.Series())
                if hasattr(predictions, 'to_dict'):
                    pred_dict = predictions.to_dict()
                else:
                    pred_dict = predictions if isinstance(predictions, dict) else {}

                # 标记为 pending
                self.state['last_update_date'] = date
                # 保存 pending trade 信息（包含 pred_df）
                pending_trade = {
                    'pred_date': date,
                    'trade_date': trade_date,
                    'pred_df': pred_dict,
                }
                self.state.setdefault('pending_trades', []).append(pending_trade)

                # 删除 predictions（不需要持久化）
                if 'predictions' in self.state:
                    del self.state['predictions']
                if 'pred_date' in self.state:
                    del self.state['pred_date']
        else:
            # date 是最后一天，保存为 pending
            logger.info("[daily] 保存为 pending，等待下一个交易日...")
            logger.info("步骤 3/5: 执行虚拟交易... (pending)")
            logger.info("步骤 4/5: 计算收益... (pending)")
            self.state['last_update_date'] = date

            # 删除 predictions（不需要持久化）
            if 'predictions' in self.state:
                del self.state['predictions']
            if 'pred_date' in self.state:
                del self.state['pred_date']

        # 5. 生成图表（如果有收益数据的话）
        logger.info("步骤 5/5: 生成图表...")
        self.generate_plots(date)

        # 保存状态
        self._save_state(self.state)

        logger.info("=" * 60)
        logger.info(f"[{date}] 每日回测完成")
        logger.info("=" * 60)

    def simulate(self, start_date: str, end_date: str = None):
        """
        历史模拟（用于测试）

        Args:
            start_date: 开始日期
            end_date: 结束日期，默认为最新交易日
        """
        from qlib.data import D

        # 获取交易日历
        if end_date is None:
            end_date = self._get_latest_trading_day()

        trading_days = D.calendar(start_time=start_date, end_time=end_date, freq='day')

        if len(trading_days) == 0:
            logger.error(f"未找到交易日历: {start_date} ~ {end_date}")
            return

        logger.info(f"开始历史模拟: {start_date} ~ {end_date}")
        logger.info(f"共 {len(trading_days)} 个交易日")

        # 重置状态
        logger.info("重置状态...")
        backtest_config = self.config.get('backtest', {})
        initial_cash = backtest_config.get('account', 100000000)

        self.state = {
            'last_update_date': None,
            'last_data_update_date': end_date,  # 模拟时使用 end_date
            'last_train_date': None,
            'account': {
                'cash': initial_cash,
                'total_value': initial_cash,
                'positions': {}
            },
            'history': {
                'dates': [],
                'returns': [],
                'total_values': []
            },
            'model_path': None,
            'pending_trades': [],  # 待计算收益的交易列表
        }

        # 首次训练
        logger.info("首次训练...")
        self.train_model()

        # 逐日模拟
        success_count = 0
        fail_count = 0

        for i, day in enumerate(trading_days):
            date_str = day.strftime('%Y-%m-%d')
            logger.info(f"\n[{i+1}/{len(trading_days)}] {date_str}")

            try:
                self.daily(date_str, state=self.state)
                success_count += 1
            except Exception as e:
                logger.error(f"失败: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1

        logger.info("\n" + "=" * 60)
        logger.info(f"历史模拟完成")
        logger.info(f"成功: {success_count}, 失败: {fail_count}")
        logger.info("=" * 60)

    def should_retrain(self, current_date: str) -> bool:
        """
        判断是否需要重新训练

        Args:
            current_date: 当前日期

        Returns:
            是否需要重新训练
        """
        from qlib.data import D

        live_config = self.config.get('live_backtest', {})
        frequency = live_config.get('retrain_frequency', 'weekly')

        # 检查上次训练日期
        last_train = self.state.get('last_train_date')

        # 如果没有训练过，需要训练
        if last_train is None:
            return True

        # 规范化日期并转换为日期类型
        try:
            curr_date = self._normalize_date(current_date)
            last_date = self._normalize_date(last_train)

            if curr_date is None or last_date is None:
                return True

            curr_ts = pd.Timestamp(curr_date)
            last_ts = pd.Timestamp(last_date)
        except Exception:
            return True

        if frequency == 'never':
            return False
        elif frequency == 'daily':
            # 每天都重新训练（只要不是同一天）
            return curr_ts.date() != last_ts.date()
        elif frequency == 'weekly':
            # 每周一重新训练
            if curr_ts.dayofweek != 0:
                return False
            # 确保距离上次训练至少7天（跨周）
            return (curr_ts - last_ts).days >= 7
        elif frequency == 'monthly':
            # 每月1号重新训练
            if curr_ts.day != 1:
                return False
            # 确保是不同的月份
            return curr_ts.month != last_ts.month or curr_ts.year != last_ts.year
        else:
            return False

    def train_model(self):
        """训练新模型"""
        from qlib.utils import init_instance_by_config
        import pickle

        train_config = self.config.get('train', {})

        # 构建配置
        model_config = self._build_model_config(train_config)
        handler_config = self._build_handler_config()
        dataset_config = self._build_dataset_config(handler_config, train_config)

        # 初始化模型和数据集
        logger.info("初始化模型...")
        model = init_instance_by_config(model_config)

        logger.info("初始化数据集...")
        dataset = init_instance_by_config(dataset_config)

        # 训练
        logger.info("开始训练模型...")
        model.fit(dataset)

        # 保存模型（使用 pickle，qlib 模型支持）
        model_dir = self.base_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"model_{timestamp}.pkl"

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            # 尝试使用 dill
            try:
                import dill
                with open(model_path, 'wb') as f:
                    dill.dump(model, f)
                logger.info(f"模型已保存到（使用dill）: {model_path}")
            except Exception as e2:
                logger.error(f"使用dill保存也失败: {e2}")
                model_path = None

        # 更新模型路径和训练日期
        model_path_str = str(model_path) if model_path else None

        # 如果 self.state 存在则更新，否则返回路径和日期
        if hasattr(self, 'state') and self.state is not None:
            self.state['model_path'] = model_path_str
            self.state['last_train_date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            # 返回训练结果供 init 使用
            return model_path_str, datetime.now().strftime('%Y-%m-%d')

        return None

    def _get_latest_data_date_from_qlib(self) -> str:
        """
        从 qlib 数据库获取最新的数据日期

        Returns:
            str: 最新数据日期（YYYY-MM-DD 格式），如果没有数据返回 None
        """
        try:
            from qlib.data import D
            calendar = D.calendar(freq='day')
            if len(calendar) == 0:
                logger.warning("qlib 日历为空，数据可能未正确加载")
                return None
            latest_date = calendar[-1].strftime('%Y-%m-%d')
            logger.debug(f"从 qlib 获取到最新数据日期: {latest_date}")
            return latest_date
        except Exception as e:
            logger.warning(f"从 qlib 获取日期失败: {e}")
            return None

    def _get_latest_data_date(self) -> str:
        """
        获取最新的数据日期（依次尝试多种方式）

        Returns:
            str: 最新数据日期（YYYY-MM-DD 格式）
        """
        # 1. 先尝试从 qlib 获取（最可靠）
        qlib_date = self._get_latest_data_date_from_qlib()
        if qlib_date:
            return qlib_date

        # 2. 从本地数据目录获取
        local_date = self._get_local_data_latest_date()
        if local_date:
            return local_date

        # 3. 回退到 config 中的 end_date
        data_config = self.config.get('data', {})
        end_date = str(data_config.get('end_date', '20250101'))
        normalized = self._normalize_date(end_date)
        if normalized:
            return normalized

        # 4. 最后手段：返回今天
        return datetime.now().strftime('%Y-%m-%d')

    def predict(self, date: str):
        """
        对指定日期进行预测

        Args:
            date: 预测日期

        Returns:
            预测结果（股票代码 -> 预测分数）
        """
        from qlib.utils import init_instance_by_config
        from qlib.data import D
        import pickle

        if self.state.get('model_path') is None:
            logger.error("未找到训练好的模型")
            return None

        model_path = self.state['model_path']
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            return None

        # 加载模型
        logger.info(f"加载模型: {model_path}")

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            try:
                import dill
                with open(model_path, 'rb') as f:
                    model = dill.load(f)
                logger.info("使用 dill 加载模型成功")
            except Exception as e2:
                logger.error(f"使用 dill 加载也失败: {e2}")
                return None

        logger.info(f"获取 {date} 的特征数据...")

        try:
            # 动态获取最新数据日期（不再依赖 config 中的 end_date）
            latest_data_date = self._get_latest_data_date()
            logger.info(f"使用数据截止日期: {latest_data_date}（动态获取）")

            # 构建数据集配置
            train_config = self.config.get('train', {})

            # 使用动态获取的 latest_data_date 作为 end_date，确保能获取到 date 的数据
            handler_config = self._build_handler_config(end_date=latest_data_date)
            dataset_config = self._build_dataset_config(handler_config, train_config)

            # 修改测试集为指定日期
            dataset_config['kwargs']['segments']['test'] = [date, date]

            logger.info(f"数据集配置: end_time={handler_config['end_time']}, test=[{date}, {date}]")
            logger.info("准备数据集...")
            dataset = init_instance_by_config(dataset_config)

            # 获取测试数据（包含股票代码和特征）
            logger.info("获取测试数据...")
            test_data = dataset.prepare('test')

            # 验证数据是否包含请求的日期
            if test_data.empty:
                logger.error(f"错误: 没有获取到 {date} 的数据！")
                logger.error(f"可能原因: 1) 数据尚未更新到 {date}; 2) 日期不在交易日历中")
                logger.error(f"可用数据日期范围请检查 qlib 数据库")
                return None

            # 检查 test_data 的日期索引
            if hasattr(test_data.index, 'get_level_values'):
                data_dates = test_data.index.get_level_values('datetime').unique()
                data_dates_str = [d.strftime('%Y-%m-%d') for d in data_dates]
            else:
                data_dates_str = []

            if date not in data_dates_str and data_dates_str:
                logger.warning(f"警告: 请求的日期 {date} 不在获取的数据中")
                logger.warning(f"获取到的数据日期: {data_dates_str[:5]}... (共 {len(data_dates_str)} 个)")
                # 不返回 None，允许使用已有数据继续

            # test_data 是 DataFrame，索引是 (datetime, instrument) MultiIndex
            # 提取股票代码
            if hasattr(test_data.index, 'get_level_values'):
                stock_codes = test_data.index.get_level_values('instrument').unique().tolist()
            else:
                stock_codes = test_data.index.unique().tolist()

            logger.info(f"获取到 {len(stock_codes)} 只股票")

            # 使用模型预测（传入 dataset 对象，LGBModel.predict 内部会调用 dataset.prepare()）
            logger.info("生成预测...")
            predictions = model.predict(dataset)

            # predictions 是 Series，索引是股票代码
            # 确保 predictions 有正确的索引
            if isinstance(predictions, np.ndarray):
                predictions = pd.Series(predictions.flatten(), index=stock_codes, name='score')

            # 过滤掉 NaN 和无效值
            if hasattr(predictions, 'dropna'):
                predictions = predictions.dropna()
            valid_mask = ~(predictions.isna() | np.isinf(predictions))
            predictions = predictions[valid_mask]

            logger.info(f"有效预测数量: {len(predictions)}")

            return predictions

        except Exception as e:
            logger.error(f"预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_qlib_backtest(self, date: str):
        """
        使用 qlib 内置回测系统执行交易

        backtest_daily + TopkDropoutStrategy 的逻辑：
        - 使用 T-1 日的预测信号
        - 在 T 日执行交易
        - 用 T+1 的价格计算收益

        Args:
            date: 交易日期

        Returns:
            dict: 回测结果，包含 trades 和 portfolio info
        """
        from qlib.contrib.evaluate import backtest_daily
        from qlib.contrib.strategy import TopkDropoutStrategy
        from qlib.backtest.position import Position

        # 从状态中获取预测数据和预测日期
        predictions = self.state.get('predictions')
        pred_date = self.state.get('pred_date')

        if predictions is None or pred_date is None:
            logger.error("[qlib回测] 未找到预测数据，请先执行预测")
            return None

        logger.info(f"[qlib回测] 开始回测")
        logger.info(f"[qlib回测] 预测日期(T): {pred_date}, 交易日期(T+1): {date}")

        # 2. 获取交易日历并检查是否是最后一天
        from qlib.data import D
        calendar = D.calendar(freq='day')
        calendar_dates = [d.strftime('%Y-%m-%d') for d in calendar]

        if date not in calendar_dates:
            logger.warning(f"[qlib回测] 日期 {date} 不在交易日历中，跳过")
            return None

        date_idx = calendar_dates.index(date)
        is_last_day = (date_idx >= len(calendar_dates) - 1)

        # 3. 检查 predictions 的索引格式
        # qlib TopkDropoutStrategy 需要 MultiIndex 格式 (datetime, instrument)
        if hasattr(predictions.index, 'get_level_values'):
            # MultiIndex 格式 (datetime, instrument)
            pred_dates = predictions.index.get_level_values('datetime').unique()
            pred_dates_str = [d.strftime('%Y-%m-%d') for d in pred_dates]

            if pred_date in pred_dates_str:
                # 保持 MultiIndex 格式，只保留 pred_date 的数据
                pred_df = predictions.xs(pd.Timestamp(pred_date), level='datetime')
                # 将单层索引转换回 MultiIndex（datetime=pred_date, instrument=stock）
                pred_df = pd.DataFrame({'score': pred_df})
                pred_df.index = pd.MultiIndex.from_product(
                    [[pd.Timestamp(pred_date)], pred_df.index],
                    names=['datetime', 'instrument']
                )
                pred_df = pred_df['score']  # 转回 Series
            else:
                logger.warning(f"[qlib回测] 预测中没有 {pred_date} 的数据")
                # 使用最新日期的预测
                latest_pred_date = pred_dates[-1]
                pred_df = predictions.xs(latest_pred_date, level='datetime')
                # 转换回 MultiIndex
                pred_df = pd.DataFrame({'score': pred_df})
                pred_df.index = pd.MultiIndex.from_product(
                    [[latest_pred_date], pred_df.index],
                    names=['datetime', 'instrument']
                )
                pred_df = pred_df['score']
                pred_date = latest_pred_date.strftime('%Y-%m-%d')
                logger.info(f"[qlib回测] 使用 {pred_date} 的预测")
        else:
            # 已经是单层索引，转换为 MultiIndex
            logger.info(f"[qlib回测] predictions 是单层索引，转换为 MultiIndex")
            pred_df = pd.DataFrame({'score': predictions})
            pred_df.index = pd.MultiIndex.from_product(
                [[pd.Timestamp(pred_date)], predictions.index],
                names=['datetime', 'instrument']
            )
            pred_df = pred_df['score']

        logger.info(f"[qlib回测] 使用 {len(pred_df)} 只股票的预测信号")

        # 4. 最后一天处理：只保存信号，不计算收益
        if is_last_day:
            logger.info(f"[qlib回测] {date} 是交易日历的最后一天，无法计算收益")
            logger.info(f"[qlib回测] 保存交易信号，明天获取 T+1 数据后再计算收益")

            # 保存 pending trade 信息到状态文件
            pending_trade = {
                'pred_date': pred_date,  # 预测数据日期
                'trade_date': date,      # 交易日期
                'pred_df': pred_df.to_dict() if hasattr(pred_df, 'to_dict') else {},
            }
            self.state.setdefault('pending_trades', []).append(pending_trade)

            # 删除 predictions（不需要持久化，pending_trades 中已有 pred_df）
            if 'predictions' in self.state:
                del self.state['predictions']
            if 'pred_date' in self.state:
                del self.state['pred_date']

            logger.info(f"[qlib回测] pending_trades 已保存: {len(self.state['pending_trades'])} 个待计算收益")
            self._save_state(self.state)

            result = {
                'date': date,
                'trades': [],
                'pending': True,  # 标记为待计算
                'portfolio': {
                    'cash': self.state['account'].get('cash', 0),
                    'total_value': self.state['account'].get('total_value', 0),
                    'positions': self.state['account'].get('positions', {}),
                }
            }
            return result

        # 5. 获取当前持仓信息
        positions = self.state['account'].get('positions', {})
        current_holdings = {s: p for s, p in positions.items() if isinstance(p, dict) and p.get('shares', 0) > 0}

        # 6. 初始化 TopkDropoutStrategy
        backtest_config = self.config.get('backtest', {})
        live_config = self.config.get('live_backtest', {})

        STRATEGY_CONFIG = {
            "topk": live_config.get('topk', 10),
            "n_drop": live_config.get('n_drop', 5),
            "signal": pred_df,
        }

        strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)

        # 7. 初始化账户
        initial_cash = backtest_config.get('account', 100000000)
        account_value = self.state['account'].get('total_value', initial_cash)

        # 如果没有持仓，用初始现金创建 Position
        if not current_holdings:
            account = account_value
        else:
            # 有持仓时需要用 Position 对象
            account = Position()
            account.reset_stock()
            for stock, pos_info in current_holdings.items():
                shares = pos_info.get('shares', 0)
                if shares > 0:
                    account.update_pos(stock, shares)

        # 8. 运行回测（只回测一天）
        exchange_kwargs = backtest_config.get('exchange_kwargs', {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        })

        logger.info(f"[qlib回测] 运行单日回测: {date}, 账户={account_value:,.0f}")
        report_df, positions_df = backtest_daily(
            start_time=date,
            end_time=date,
            strategy=strategy_obj,
            account=account,
            benchmark=backtest_config.get('benchmark', 'SH000300'),
            exchange_kwargs=exchange_kwargs,
        )

        # 9. 提取结果
        result = {
            'date': date,
            'trades': [],
            'pending': False,
            'portfolio': {
                'cash': self.state['account'].get('cash', 0),
                'total_value': account_value,
                'positions': current_holdings,
            }
        }

        try:
            if report_df is not None and not report_df.empty:
                logger.debug(f"[qlib回测] 报告列: {report_df.columns.tolist()}")
                logger.debug(f"[qlib回测] 报告摘要:\n{report_df.iloc[:3]}")

                # 从 report_df 提取账户信息
                if 'total_value' in report_df.columns:
                    final_value = float(report_df['total_value'].iloc[-1])
                    result['portfolio']['total_value'] = final_value
                    logger.info(f"[qlib回测] 账户总价值: {final_value:,.0f}")

                if 'cash' in report_df.columns:
                    result['portfolio']['cash'] = float(report_df['cash'].iloc[-1])

            if positions_df is not None and not positions_df.empty:
                logger.debug(f"[qlib回测] 持仓日期: {list(positions_df.keys())[:3]}...")
                # 更新持仓信息
                latest_date = list(positions_df.keys())[-1]
                if latest_date in positions_df:
                    new_positions = positions_df[latest_date]
                    if hasattr(new_positions, 'get_stock_list'):
                        stock_list = new_positions.get_stock_list()
                        new_holdings = {}
                        for stock in stock_list:
                            shares = new_positions.get_stock_amount(stock)
                            if shares > 0:
                                new_holdings[stock] = {'shares': shares}
                        result['portfolio']['positions'] = new_holdings
                        logger.info(f"[qlib回测] 持仓股票数: {len(new_holdings)}")

        except Exception as e:
            logger.warning(f"[qlib回测] 提取结果失败: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"[qlib回测] 完成: 现金={result['portfolio']['cash']:,.0f}, "
                   f"总资产={result['portfolio']['total_value']:,.0f}")

        # 删除 predictions（不需要持久化）
        if 'predictions' in self.state:
            del self.state['predictions']
        if 'pred_date' in self.state:
            del self.state['pred_date']

        return result

    def execute_trades(self, predictions: pd.Series, date: str) -> dict:
        """
        执行虚拟交易（使用 qlib 内置回测系统）

        Args:
            predictions: 预测结果
            date: 交易日期

        Returns:
            dict: 回测结果
        """
        logger.info(f"使用 qlib 回测系统执行 {date} 的交易")

        # 预测已在 daily() 中保存，这里直接运行 qlib 回测
        # predictions 已保存到 self.state['predictions']
        # pred_date 已保存到 self.state['pred_date']

        # 运行 qlib 回测
        backtest_result = self._run_qlib_backtest(date)

        # 更新状态
        if backtest_result and backtest_result.get('portfolio'):
            portfolio = backtest_result['portfolio']
            self.state['account']['cash'] = portfolio.get('cash', self.state['account']['cash'])
            self.state['account']['total_value'] = portfolio.get('total_value', self.state['account']['total_value'])
            self.state['account']['positions'] = portfolio.get('positions', self.state['account']['positions'])
            logger.info(f"账户已更新: 现金={self.state['account']['cash']:,.0f}, "
                       f"总资产={self.state['account']['total_value']:,.0f}")

        return backtest_result

    def calculate_return(self, date: str, backtest_result: dict = None) -> float:
        """
        计算当日收益

        Args:
            date: 交易日期
            backtest_result: 可选的回测结果，用于检查是否是 pending 状态

        Returns:
            float: 当日收益率，如果未计算则返回 None
        """
        # 检查是否是 pending 状态（最后一天的情况）
        if backtest_result and backtest_result.get('pending'):
            logger.info(f"[{date}] 是最后交易日，收益待明天 T+1 数据更新后计算")
            # 只更新时间，不记录收益
            self.state['last_update_date'] = date
            return None

        history = self.state['history']
        prev_total = history['total_values'][-1] if history['total_values'] else self.config['backtest']['account']
        current_total = self.state['account']['total_value']

        daily_return = (current_total - prev_total) / prev_total

        # 更新历史
        history['dates'].append(date)
        history['returns'].append(daily_return)
        history['total_values'].append(current_total)
        self.state['history'] = history
        self.state['last_update_date'] = date

        return daily_return

    def generate_plots(self, date: str):
        """
        生成每日图表
        """
        history = self.state['history']
        if len(history['dates']) == 0:
            logger.warning("没有历史数据，跳过图表生成")
            return

        # 创建 DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(history['dates']),
            'return': history['returns'],
            'total_value': history['total_values']
        })
        df.set_index('date', inplace=True)

        # 创建输出目录
        plots_dir = self.base_dir / "plots" / date / "live"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"生成图表到: {plots_dir}")

        # 1. 累计收益曲线
        self._plot_cumulative_returns(df, plots_dir)

        # 2. 每日收益柱状图
        self._plot_daily_returns(df, plots_dir)

        # 3. 回撤图
        self._plot_drawdown(df, plots_dir)

        # 4. 收益统计
        self._plot_statistics(df, plots_dir)

        # 5. 保存 CSV
        csv_path = plots_dir / "daily_returns.csv"
        df.to_csv(csv_path)
        logger.info(f"收益数据已保存到: {csv_path}")

        logger.info(f"图表生成完成")

    def _plot_cumulative_returns(self, df, output_dir):
        """绘制累计收益曲线"""
        import matplotlib.pyplot as plt

        cum_return = (1 + df['return']).cumprod() - 1

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(cum_return.index, cum_return.values * 100, label='Strategy', linewidth=2, color='#2E86AB')

        ax.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / '01_cumulative_returns.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_daily_returns(self, df, output_dir):
        """绘制每日收益柱状图"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#27AE60' if r >= 0 else '#E74C3C' for r in df['return']]
        ax.bar(df.index, df['return'] * 100, color=colors, alpha=0.7, width=1)

        ax.set_title('Daily Returns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Daily Return (%)', fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / '02_daily_returns.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_drawdown(self, df, output_dir):
        """绘制回撤图"""
        import matplotlib.pyplot as plt

        cum_return = (1 + df['return']).cumprod()
        running_max = cum_return.cummax()
        drawdown = (cum_return - running_max) / running_max

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.4, color='#E74C3C')
        ax.plot(drawdown.index, drawdown.values * 100, linewidth=1, color='#C0392B')

        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / '03_drawdown.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_statistics(self, df, output_dir):
        """绘制收益统计"""
        import matplotlib.pyplot as plt

        returns = df['return'].dropna()
        if len(returns) == 0:
            return

        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cum_return = (1 + returns).cumprod()
        running_max = cum_return.cummax()
        drawdown = (cum_return - running_max) / running_max
        max_drawdown = drawdown.min()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. 关键指标
        ax1 = axes[0, 0]
        metrics = {
            'Total Return': f'{total_return*100:.2f}%',
            'Annual Return': f'{annual_return*100:.2f}%',
            'Sharpe': f'{sharpe:.3f}',
            'Max Drawdown': f'{max_drawdown*100:.2f}%',
        }
        ax1.axis('off')
        ax1.text(0.1, 0.9, 'Performance Summary', fontsize=14, fontweight='bold', transform=ax1.transAxes)
        for i, (k, v) in enumerate(metrics.items()):
            ax1.text(0.1, 0.7 - i*0.15, f'{k}: {v}', fontsize=12, transform=ax1.transAxes)

        # 2. 月度收益
        ax2 = axes[0, 1]
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        ax2.bar(range(len(monthly)), monthly.values * 100,
                color=['#27AE60' if x >= 0 else '#E74C3C' for x in monthly.values])
        ax2.set_xticks(range(len(monthly)))
        ax2.set_xticklabels([d.strftime('%Y-%m') for d in monthly.index], rotation=45, ha='right', fontsize=8)
        ax2.set_title('Monthly Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 收益分布
        ax3 = axes[1, 0]
        ax3.hist(returns * 100, bins=30, color='#3498DB', alpha=0.7, edgecolor='white', density=True)
        ax3.axvline(x=returns.mean() * 100, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {returns.mean()*100:.3f}%')
        ax3.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 账户价值曲线
        ax4 = axes[1, 1]
        ax4.plot(df.index, df['total_value'] / 1e8 * 100, linewidth=2, color='#2E86AB')
        ax4.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Initial')
        ax4.set_title('Account Value', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Value (% of Initial)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / '04_statistics.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 保存统计到文件
        stats_file = output_dir / "statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("策略表现统计\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"总收益率: {total_return*100:.2f}%\n")
            f.write(f"年化收益率: {annual_return*100:.2f}%\n")
            f.write(f"年化波动率: {annual_vol*100:.2f}%\n")
            f.write(f"夏普比率: {sharpe:.3f}\n")
            f.write(f"最大回撤: {max_drawdown*100:.2f}%\n")
            f.write(f"交易天数: {len(returns)}\n")
            f.write(f"正收益天数: {(returns > 0).sum()}\n")
            f.write(f"胜率: {(returns > 0).mean()*100:.2f}%\n")
        logger.info(f"统计信息已保存到: {stats_file}")

    def check_pending(self):
        """
        手动检查并计算 pending 收益

        用于测试或在数据更新后手动触发 pending 计算
        """
        logger.info("=" * 60)
        logger.info("手动检查 pending 交易")
        logger.info("=" * 60)

        # 加载状态
        self.state = self._load_state()
        if self.state is None:
            logger.error("未找到状态文件，请先运行 init 初始化")
            return

        pending_trades = self.state.get('pending_trades', [])
        if not pending_trades:
            logger.info("没有待计算的 pending 交易")
            return

        logger.info(f"发现 {len(pending_trades)} 个 pending 交易")

        # 获取当前 qlib 数据最后日期
        latest_qlib_date = self._get_latest_data_date_from_qlib()
        logger.info(f"当前 qlib 数据最后日期: {latest_qlib_date}")

        # 手动触发 pending 计算
        calculated = self._calculate_pending_returns(latest_qlib_date)

        if calculated > 0:
            logger.info(f"成功计算 {calculated} 个 pending 交易的收益")
            self._save_state(self.state)

            # 重新生成图表
            if self.state.get('history', {}).get('dates'):
                last_date = self.state['history']['dates'][-1]
                self.generate_plots(last_date)
        else:
            logger.info("没有可以计算的 pending 交易（T+1 数据尚未更新）")

        logger.info("=" * 60)


def main():
    """主函数"""
    backtest = LiveBacktest()
    fire.Fire({
        'init': backtest.init,
        'daily': backtest.daily,
        'simulate': backtest.simulate,
        'check_pending': backtest.check_pending,
    })


if __name__ == "__main__":
    main()
