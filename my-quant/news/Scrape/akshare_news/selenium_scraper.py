# -*- coding: utf-8 -*-
"""
Selenium 爬虫方案

通过浏览器自动化滚动页面获取更多电报数据
"""

import time
from datetime import date, datetime
from typing import List, Dict, Any, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False


class SeleniumScraper:
    """Selenium 方式爬取电报数据"""

    BASE_URL = "https://www.cls.cn/telegraph"
    TELEGRAPH_URL = "https://www.cls.cn/telegraph"

    def __init__(
        self,
        headless: bool = True,
        window_size: tuple = (1400, 900),
        timeout: int = 30,
    ):
        """
        初始化

        Args:
            headless: 是否使用 headless 模式
            window_size: 浏览器窗口大小
            timeout: 超时时间（秒）
        """
        self.headless = headless
        self.window_size = window_size
        self.timeout = timeout
        self.driver = None

    def _setup_driver(self):
        """配置并启动浏览器"""
        options = Options()

        if self.headless:
            options.add_argument("--headless=new")

        # 反检测设置
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")

        # 用户代理
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # 窗口大小
        options.add_argument(f"--window-size={self.window_size[0]},{self.window_size[1]}")

        # 启动浏览器
        if WEBDRIVER_MANAGER_AVAILABLE:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)

        self.driver.set_page_load_timeout(self.timeout)

    def _ensure_driver(self):
        """确保浏览器已启动"""
        if self.driver is None:
            self._setup_driver()

    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def __enter__(self):
        """上下文管理器入口"""
        self._ensure_driver()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()

    def _scroll_and_load(
        self,
        target_count: int = 1000,
        scroll_pause: float = 1.5,
        max_scrolls: int = 100,
    ) -> int:
        """
        滚动页面加载更多数据

        Args:
            target_count: 目标数量
            scroll_pause: 滚动暂停时间
            max_scrolls: 最大滚动次数

        Returns:
            实际滚动的次数
        """
        scroll_count = 0
        last_count = 0

        while scroll_count < max_scrolls:
            # 滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause)

            # 检查当前已加载的数量
            current_count = len(self.driver.find_elements(By.CLASS_NAME, "telegraph-time-box"))

            # 如果达到目标数量，停止
            if current_count >= target_count:
                print(f"[Selenium] 已达到目标数量: {current_count}")
                break

            # 如果连续两次检查数量没有变化，说明已加载完所有数据
            if current_count == last_count and scroll_count > 5:
                print(f"[Selenium] 页面已无更多数据，当前: {current_count}")
                break

            last_count = current_count
            scroll_count += 1

            if scroll_count % 10 == 0:
                print(f"[Selenium] 滚动 {scroll_count} 次，当前 {current_count} 条...")

        return scroll_count

    def fetch_telegraph(
        self,
        target_count: int = 1000,
        scroll_pause: float = 1.5,
        max_scrolls: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取电报数据（滚动页面加载更多）

        Args:
            target_count: 目标数量
            scroll_pause: 滚动暂停时间
            max_scrolls: 最大滚动次数

        Returns:
            电报数据列表
        """
        self._ensure_driver()

        print(f"[Selenium] 打开页面: {self.TELEGRAPH_URL}")
        self.driver.get(self.TELEGRAPH_URL)
        time.sleep(3)  # 等待页面初始加载

        # 滚动加载更多数据
        print(f"[Selenium] 开始滚动加载数据...")
        self._scroll_and_load(
            target_count=target_count,
            scroll_pause=scroll_pause,
            max_scrolls=max_scrolls,
        )

        # 提取数据
        items = self._extract_items()
        unique_items = self._deduplicate(items)

        print(f"[Selenium] 共获取 {len(unique_items)} 条电报")
        return unique_items

    def fetch_daily_telegraph(
        self,
        dt: date,
        target_count: int = 1000,
        scroll_pause: float = 1.5,
        max_scrolls: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取单日所有电报数据

        Args:
            dt: 日期
            target_count: 目标数量
            scroll_pause: 滚动暂停时间
            max_scrolls: 最大滚动次数

        Returns:
            电报数据列表
        """
        # 打开页面
        self._ensure_driver()
        print(f"[Selenium] 打开页面: {self.TELEGRAPH_URL}")
        self.driver.get(self.TELEGRAPH_URL)
        time.sleep(3)

        # 尝试选择日期
        self._select_date(dt)

        # 滚动加载更多数据
        print(f"[Selenium] 开始滚动加载 {dt} 的数据...")
        self._scroll_and_load(
            target_count=target_count,
            scroll_pause=scroll_pause,
            max_scrolls=max_scrolls,
        )

        # 提取并过滤数据
        all_items = self._extract_items()
        target_date_str = dt.strftime("%Y-%m-%d")

        # 过滤出目标日期的数据
        filtered_items = [
            item for item in all_items
            if item["datetime"].startswith(target_date_str)
        ]

        unique_items = self._deduplicate(filtered_items)
        print(f"[Selenium] {dt} 共获取 {len(unique_items)} 条电报")
        return unique_items

    def _select_date(self, dt: date):
        """
        在日历中选择日期

        Args:
            dt: 要选择的日期
        """
        try:
            # 点击日历按钮
            calendar_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//*[text()='日历']"))
            )
            calendar_btn.click()
            time.sleep(1)

            # 等待日历面板出现
            panel = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//div[contains(@class, 'picker-panel')]")
                )
            )

            # 切换到目标月份
            self._switch_to_month(dt)

            # 点击目标日期
            day_xpath = (
                f"//td[not(contains(@class, 'last')) and not(contains(@class, 'next'))]"
                f"//div[text()='{dt.day}']"
            )
            day_cell = self.driver.find_element(By.XPATH, day_xpath)
            day_cell.click()

            print(f"[Selenium] 已选择日期: {dt}")
            time.sleep(3)  # 等待数据加载

        except TimeoutException:
            print("[Selenium] 无法找到日历元素，跳过日期选择")

    def _switch_to_month(self, dt: date):
        """
        切换日历到目标月份

        Args:
            dt: 目标日期
        """
        try:
            # 获取当前面板显示的月份
            header = self.driver.find_element(
                By.CSS_SELECTOR, ".ant-picker-header"
            ).text
            current_month = int(header.split("年")[1].replace("月", ""))

            target_month = dt.month

            # 计算需要切换的次数
            diff = (dt.year - datetime.now().year) * 12 + (target_month - current_month)

            # 往前翻（历史数据）
            for _ in range(abs(diff)):
                prev_btn = self.driver.find_element(
                    By.XPATH, "//button[@title='上个月']"
                )
                prev_btn.click()
                time.sleep(0.3)

        except NoSuchElementException:
            print("[Selenium] 无法切换月份")

    def _extract_items(self) -> List[Dict[str, Any]]:
        """
        从当前页面提取电报数据

        Returns:
            电报数据列表
        """
        items = []

        # 查找所有时间元素
        time_elements = self.driver.find_elements(By.CLASS_NAME, "telegraph-time-box")

        for time_elem in time_elements:
            time_text = time_elem.text.strip()
            if not time_text:
                continue

            # 找相邻的内容元素
            parent = time_elem.parent
            try:
                content_elem = parent.find_element(By.CLASS_NAME, "c-34304b")
                content_text = content_elem.text.strip()
            except NoSuchElementException:
                continue

            if not content_text:
                continue

            # 尝试解析时间
            try:
                parts = time_text.split(':')
                hour = int(parts[0])
                minute = int(parts[1])
                second = int(parts[2])
                # 日期取今天
                dt = datetime.now()
                dt_str = dt.strftime("%Y-%m-%d") + f" {hour:02d}:{minute:02d}:{second:02d}"
            except (ValueError, IndexError):
                dt_str = time_text

            item = {
                "datetime": dt_str,
                "content": content_text,
                "category": self._detect_category(content_text),
                "source": "财联社",
                "url": self.TELEGRAPH_URL,
            }
            items.append(item)

        return items

    def _detect_category(self, content: str) -> str:
        """检测电报分类"""
        import re
        if content.startswith('【'):
            match = re.search(r'【([^【】]+)】', content)
            if match:
                return match.group(1)
        return "快讯"

    def _deduplicate(
        self, items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        去重

        Args:
            items: 数据列表

        Returns:
            去重后的数据列表
        """
        seen = set()
        unique = []

        for item in items:
            key = item["content"][:100] if len(item["content"]) > 100 else item["content"]
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique


if __name__ == "__main__":
    # 测试
    with SeleniumScraper(headless=False) as scraper:
        # 获取今天数据
        print("=== 获取今天数据 ===")
        items = scraper.fetch_telegraph(target_count=500, max_scrolls=20)
        print(f"\n获取到 {len(items)} 条电报")
