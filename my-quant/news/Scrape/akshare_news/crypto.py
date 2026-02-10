# -*- coding: utf-8 -*-
"""
财联社 sign 加密模块

根据逆向分析，cls.cn 使用 SHA1 + MD5 双层加密生成 sign 参数:
1. 先对请求参数做 SHA1 哈希 (40位)
2. 再对 SHA1 结果做 MD5 哈希 (32位)
"""

import hashlib
import time
import json
from typing import Dict, Any, Optional
from urllib.parse import urlencode, parse_qs


def generate_timestamp() -> str:
    """生成13位时间戳（毫秒）"""
    return str(int(time.time() * 1000))


def sha1_hash(data: str) -> str:
    """SHA1 哈希"""
    return hashlib.sha1(data.encode('utf-8')).hexdigest()


def md5_hash(data: str) -> str:
    """MD5 哈希"""
    return hashlib.md5(data.encode('utf-8')).hexdigest()


def generate_sign(params: Dict[str, Any], secret: str = "") -> str:
    """
    生成 sign 参数（SHA1 + MD5 双层加密）

    Args:
        params: 请求参数字典
        secret: 密钥（如果有）

    Returns:
        加密后的 sign 字符串
    """
    # 1. 将参数排序并拼接成字符串
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    param_str = "&".join([f"{k}={v}" for k, v in sorted_params])

    # 2. 第一层 SHA1 哈希
    sha1_result = sha1_hash(param_str)

    # 3. 第二层 MD5 哈希（如果有 secret，则拼接 secret）
    if secret:
        md5_input = sha1_result + secret
    else:
        md5_input = sha1_result

    return md5_hash(md5_input)


class CLSCrypto:
    """财联社加密工具类"""

    BASE_URL = "https://www.cls.cn"

    # 常见的 API 端点
    TELEGRAPH_API = "/api/telegraph/list"
    LATEST_API = "/api/telegraph/latest"

    def __init__(self, app_key: str = "", app_secret: str = ""):
        """
        初始化

        Args:
            app_key: 应用密钥（如果有）
            app_secret: 应用密钥（如果有）
        """
        self.app_key = app_key
        self.app_secret = app_secret

    def get_headers(self) -> Dict[str, str]:
        """
        生成请求头

        Returns:
            请求头字典
        """
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": f"{self.BASE_URL}/telegraph",
            "Origin": self.BASE_URL,
        }

    def prepare_telegraph_params(
        self,
        page: int = 1,
        page_size: int = 50,
        timestamp: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        准备电报列表请求参数

        Args:
            page: 页码
            page_size: 每页数量
            timestamp: 时间戳
            date: 日期筛选（YYYY-MM-DD 格式）

        Returns:
            参数字典
        """
        ts = timestamp or generate_timestamp()

        params = {
            "app_key": self.app_key if self.app_key else "",
            "timestamp": ts,
            "page": page,
            "page_size": page_size,
            "type": "telegraph",
            "os": "web",
            "ver": "1.0.0",
        }

        if date:
            params["date"] = date

        # 生成 sign
        params["sign"] = generate_sign(params, self.app_secret)

        return params

    def parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析响应数据

        Args:
            response_data: 原始响应数据

        Returns:
            解析后的数据
        """
        # 检查响应状态
        if response_data.get("code") != 0:
            raise ValueError(f"API 返回错误: {response_data.get('msg')}")

        return response_data.get("data", {})


if __name__ == "__main__":
    # 测试
    crypto = CLSCrypto()

    params = crypto.prepare_telegraph_params(page=1, page_size=10)
    print("生成的参数:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    print(f"\nSign: {params.get('sign')}")
