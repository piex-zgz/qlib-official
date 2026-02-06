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
        """将状态中的字符串转换回数字类型"""
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

    def _save_state(self, state: dict):
        """保存状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"状态已保存到: {self.state_file}")

    def _get_latest_trading_day(self) -> str:
        """获取最新的交易日"""
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

    def init(self):
        """首次初始化"""
        logger.info("=" * 60)
        logger.info("开始初始化实盘模拟系统")
        logger.info("=" * 60)

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

        state = {
            'last_update_date': None,
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
            'model_path': model_path
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
        # 确定运行日期
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

        # 加载状态
        if state is not None:
            # 使用外部传入的状态
            self.state = state
        else:
            self.state = self._load_state()

        if self.state is None:
            logger.error("未找到状态文件，请先运行 init 初始化")
            return

        logger.info(f"当前账户: 现金={self.state['account']['cash']:,.0f}, "
                   f"总资产={self.state['account']['total_value']:,.0f}")

        # 1. 检查是否需要重新训练
        logger.info("步骤 1/5: 检查是否需要重新训练...")
        if self.should_retrain(date):
            logger.info("需要重新训练，开始训练...")
            self.train_model()
        else:
            logger.info("使用现有模型")

        # 2. 获取前一日的预测信号（T日预测T+1日）
        # 找到 date 的前一个交易日
        logger.info("步骤 2/5: 获取前一日预测信号（T日预测T+1日）...")
        from qlib.data import D
        calendar = D.calendar(freq='day')
        try:
            date_idx = list(calendar).index(pd.Timestamp(date))
            if date_idx == 0:
                logger.warning("这是第一个交易日，没有前一日数据，跳过")
                return
            prev_date = calendar[date_idx - 1].strftime('%Y-%m-%d')
            logger.info(f"使用 {prev_date} 的预测数据指导 {date} 的交易")
        except (ValueError, IndexError) as e:
            logger.error(f"无法找到 {date} 的前一交易日: {e}")
            return

        predictions = self.predict(prev_date)

        if predictions is None or predictions.empty:
            logger.warning("预测结果为空，跳过今日交易")
            return

        logger.info(f"预测股票数量: {len(predictions)}")
        topk = self.config.get('live_backtest', {}).get('topk', 50)
        logger.info(f"选择 Top-{topk} 股票作为交易信号")

        # 3. 执行虚拟交易（在 date 这天执行）
        logger.info("步骤 3/5: 执行虚拟交易...")
        self.execute_trades(predictions, date)

        # 4. 计算收益
        logger.info("步骤 4/5: 计算收益...")
        daily_return = self.calculate_return(date)

        if daily_return is not None:
            logger.info(f"当日收益率: {daily_return*100:.4f}%")
            total_return = (self.state['account']['total_value'] / self.config['backtest']['account'] - 1) * 100
            logger.info(f"累计收益率: {total_return:.2f}%")

        # 5. 生成图表
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
            'model_path': None
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
            # 构建数据集配置
            train_config = self.config.get('train', {})
            handler_config = self._build_handler_config(end_date=date)
            dataset_config = self._build_dataset_config(handler_config, train_config)

            # 修改测试集为指定日期
            dataset_config['kwargs']['segments']['test'] = [date, date]

            logger.info("准备数据集...")
            dataset = init_instance_by_config(dataset_config)

            # 获取测试数据（包含股票代码和特征）
            logger.info("获取测试数据...")
            test_data = dataset.prepare('test')

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

    def _get_alpha_feature_names(self) -> list:
        """获取 Alpha158 的特征字段列表"""
        # Alpha158 使用的特征字段（158个因子）
        # 这里列出主要的特征字段

        # 基础价格和成交量特征 (10个)
        basic_features = [
            '$close', '$open', '$high', '$low', '$volume', '$amount',
            '$turn', '$pct_chg', '$adj_close', '$adj_factor',
        ]

        # 技术指标特征 - 动量 (约30个)
        momentum_features = [
            '$resi', '$delta', '$chg', '$pct_chg',
            '$indmom', '$stock_returns', '$stock_wgt',
        ]

        # 时序特征 (约40个)
        tail_features = [
            '$tail', '$max', '$min', '$zscore', '$subind',
            '$tsh', '$sratio', '$sharpe', '$sort',
        ]

        # 成交额特征 (约20个)
        volume_features = [
            '$vm', '$vma', '$vstd', '$vr', '$alpha',
        ]

        # 布林带和波动率特征 (约15个)
        bollinger_features = [
            '$bbands', '$bb_up', '$bb_low', '$bb_range', '$bb_width',
        ]

        # 其他常见特征 (约40个)
        other_features = [
            '$return_1', '$return_3', '$return_5', '$return_10', '$return_20',
            '$volume_1', '$volume_3', '$volume_5', '$volume_10', '$volume_20',
            '$amount_1', '$amount_3', '$amount_5', '$amount_10', '$amount_20',
            '$close_1', '$close_3', '$close_5', '$close_10', '$close_20',
            '$high_1', '$high_3', '$high_5', '$high_10', '$high_20',
            '$low_1', '$low_3', '$low_5', '$low_10', '$low_20',
        ]

        # 组合所有特征
        all_features = (basic_features + momentum_features + tail_features +
                       volume_features + bollinger_features + other_features)

        return all_features

    def execute_trades(self, predictions: pd.Series, date: str):
        """
        执行虚拟交易

        Args:
            predictions: 预测结果
            date: 交易日期
        """
        from qlib.data import D

        live_config = self.config.get('live_backtest', {})
        topk = live_config.get('topk', 50)

        # predictions.index 可能是 MultiIndex (datetime, instrument)
        # 需要正确提取股票代码
        if hasattr(predictions.index, 'get_level_values'):
            # MultiIndex 情况
            stock_codes_all = predictions.index.get_level_values('instrument').unique().tolist()
        else:
            stock_codes_all = predictions.index.tolist()

        # 获取目标股票列表（Top-K）
        top_predictions = predictions.nlargest(topk)

        # 从 Top-K 预测中提取股票代码
        if hasattr(top_predictions.index, 'get_level_values'):
            top_stocks = top_predictions.index.get_level_values('instrument').tolist()
        else:
            top_stocks = top_predictions.index.tolist()

        # 过滤掉不在预测结果中的股票
        top_stocks = [s for s in top_stocks if s in stock_codes_all]

        if not top_stocks:
            logger.warning("没有有效的目标股票")
            return

        logger.info(f"目标股票数量: {len(top_stocks)}")

        # 获取当前持仓股票列表
        positions = self.state['account']['positions']
        current_holdings = [s for s, pos in positions.items() if pos.get('shares', 0) > 0]

        # 需要获取价格的股票 = 当前持仓 + 目标股票
        stocks_to_price = list(set(current_holdings + top_stocks))

        # 获取当前价格
        try:
            # 获取收盘价 - 使用正确的 D.features API
            fields = ['$close']
            price_data = D.features(
                instruments=stocks_to_price,
                fields=fields,
                start_time=date,
                end_time=date,
            )

            if price_data.empty:
                logger.warning(f"无法获取 {date} 的价格数据")
                return

            # 提取收盘价 - price_data 格式为 MultiIndex (datetime, instrument)
            # 使用 .unstack() 将其转换为 (instrument, datetime) 格式
            prices_df = price_data.unstack(level='instrument')

            if prices_df.empty:
                logger.warning("价格数据为空")
                return

            # 获取收盘价列
            if '$close' in prices_df.columns:
                prices = prices_df['$close'].iloc[0]
            else:
                # 尝试其他可能的名字
                prices = prices_df.iloc[:, 0]

            # 确保 prices 是 Series，索引是股票代码
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            prices.name = None
            prices.index.name = None

        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 卖出逻辑
        cash = self.state['account']['cash']

        sell_logs = []
        for stock in list(positions.keys()):
            shares = positions[stock].get('shares', 0)
            if shares > 0 and stock not in top_stocks:
                # 卖出不在目标列表中的股票
                if stock in prices.index:
                    price = prices.get(stock)
                    if pd.notna(price) and price > 0:
                        sell_value = shares * price
                        positions[stock]['shares'] = 0
                        cash += sell_value
                        sell_logs.append(f"  卖出 {stock}: {shares} 股 @ {price:.2f} = {sell_value:,.0f}")
                else:
                    logger.warning(f"  无法获取 {stock} 的价格，跳过卖出")

        for log in sell_logs:
            logger.info(log)

        # 买入逻辑
        available_cash = cash * 0.95
        buy_count = 0
        for s in top_stocks:
            current_shares = positions.get(s, {}).get('shares', 0)
            if current_shares == 0 and s in prices.index:
                price = prices.get(s)
                if pd.notna(price) and price > 0:
                    buy_count += 1

        cash_per_stock = available_cash / buy_count if buy_count > 0 else 0

        buy_logs = []
        for stock in top_stocks:
            current_shares = positions.get(stock, {}).get('shares', 0)
            if current_shares == 0 and stock in prices.index:
                price = prices.get(stock)
                if pd.notna(price) and price > 0:
                    shares = int(cash_per_stock / price)
                    if shares > 0:
                        cost = shares * price
                        positions[stock] = {'shares': shares, 'avg_cost': price}
                        cash -= cost
                        buy_logs.append(f"  买入 {stock}: {shares} 股 @ {price:.2f} = {cost:,.0f}")

        for log in buy_logs:
            logger.info(log)

        # 更新状态
        self.state['account']['positions'] = positions
        self.state['account']['cash'] = cash

        # 计算持仓市值
        position_value = 0
        for s in positions:
            if s in prices.index:
                price = prices.get(s)
                if pd.notna(price):
                    position_value += positions[s]['shares'] * price

        self.state['account']['total_value'] = cash + position_value
        logger.info(f"当前现金: {cash:,.0f}, 持仓市值: {position_value:,.0f}, 总资产: {self.state['account']['total_value']:,.0f}")

    def calculate_return(self, date: str) -> float:
        """
        计算当日收益
        """
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


def main():
    """主函数"""
    backtest = LiveBacktest()
    fire.Fire({
        'init': backtest.init,
        'daily': backtest.daily,
        'simulate': backtest.simulate,
    })


if __name__ == "__main__":
    main()
