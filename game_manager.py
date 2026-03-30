"""
游戏管理器 - 管理东方永夜抄游戏进程、截图和自动化
"""

import subprocess
import time
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import psutil


class GameManager:
    """东方永夜抄游戏管理器

    负责游戏进程的启动、监控、截图和自动化操作
    """

    # 游戏窗口名称（可能的不同名称）
    WINDOW_NAMES = [
        "东方永夜抄",
        "東方永夜抄",
        "Touhou",
        "th08",
    ]

    # 游戏分辨率
    GAME_WIDTH = 960
    GAME_HEIGHT = 720

    def __init__(self, game_path: Optional[str] = None):
        """初始化游戏管理器

        Args:
            game_path: 游戏exe路径，默认为项目目录下的游戏
        """
        if game_path is None:
            # 默认路径：项目目录下的游戏文件夹
            project_root = Path(__file__).parent.parent
            game_path = project_root / "[th08] 东方永夜抄 (汉化版)" / "th08.exe"

        self.game_path = Path(game_path)
        if not self.game_path.exists():
            raise FileNotFoundError(f"游戏文件不存在: {self.game_path}")

        self.process: Optional[subprocess.Popen] = None
        self.keyboard = None

        # 对话画面的颜色特征（需要根据实际游戏画面调整）
        self.dialogue_color_threshold = 200

    def launch_game(self) -> bool:
        """启动游戏

        Returns:
            bool: 启动是否成功
        """
        if self.is_running():
            print("游戏已在运行中")
            return True

        try:
            self.process = subprocess.Popen(
                str(self.game_path),
                cwd=str(self.game_path.parent)
            )
            print(f"游戏进程已启动，PID: {self.process.pid}")

            # 等待游戏窗口加载
            time.sleep(3)

            # 尝试将游戏窗口移动到左上角
            self._move_window_to_topleft()

            # 确保窗口置顶（使用修复后的 bring_to_front）
            self.bring_to_front()

            return self.is_running()

        except Exception as e:
            print(f"启动游戏失败: {e}")
            return False

    def _move_window_to_topleft(self):
        """将游戏窗口移动到屏幕左上角并设置焦点"""
        try:
            import win32gui
            import win32con

            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    for name in self.WINDOW_NAMES:
                        if name in title:
                            windows.append(hwnd)
                            break
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                hwnd = windows[0]
                # 设置窗口焦点
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.2)
                # 移动窗口到左上角 (0, 0)
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, self.GAME_WIDTH, self.GAME_HEIGHT, 0)
                print(f"已将游戏窗口移动到左上角 (0, 0) 并设置焦点")
                time.sleep(0.5)

        except Exception as e:
            print(f"移动窗口失败: {e}")

    def terminate_game(self) -> bool:
        """强制关闭游戏

        Returns:
            bool: 关闭是否成功
        """
        if self.process is None:
            return True

        try:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
            print("游戏已关闭")
            return True

        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process = None
            print("游戏已强制关闭")
            return True

        except Exception as e:
            print(f"关闭游戏失败: {e}")
            return False

    def is_running(self) -> bool:
        """检查游戏是否正在运行

        Returns:
            bool: 游戏是否运行中
        """
        if self.process is None:
            return False

        # 检查进程是否存在
        if self.process.poll() is not None:
            self.process = None
            return False

        return True

    def is_process_alive(self) -> bool:
        """检查游戏进程是否存活（包括检查进程名）

        Returns:
            bool: 游戏进程是否存活
        """
        for proc in psutil.process_iter(['name', 'exe']):
            try:
                if proc.info['name'] and 'th08.exe' in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def capture_screen(self) -> Optional[np.ndarray]:
        """截取游戏窗口画面

        Returns:
            np.ndarray: 截图图像，失败返回None
        """
        try:
            import pyautogui

            # 获取游戏窗口位置
            window = self._get_window_rect()
            if window is None:
                # 使用固定位置（游戏以640x480运行在屏幕左上角）
                # 截取固定区域 (0, 0, 640, 480)
                screenshot = pyautogui.screenshot(region=(0, 0, self.GAME_WIDTH, self.GAME_HEIGHT))
            else:
                x, y, w, h = window
                # 使用窗口实际尺寸截图
                screenshot = pyautogui.screenshot(region=(x, y, w, h))

            # 转换为OpenCV格式
            img = np.array(screenshot)
            # RGB转BGR
            img = img[:, :, ::-1].copy()

            return img

        except ImportError:
            print("需要安装pyautogui: pip install pyautogui")
            return None

        except Exception as e:
            print(f"截图失败: {e}")
            return None

    def _get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """获取游戏窗口位置

        Returns:
            Tuple[int, int, int, int]: (x, y, width, height) 或 None
        """
        try:
            import win32gui
            import win32con

            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    # 检查是否匹配任何已知的窗口名称
                    for name in self.WINDOW_NAMES:
                        if name in title:
                            windows.append(hwnd)
                            break
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                hwnd = windows[0]
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                # 转换为 (x, y, width, height) 格式
                width = right - left
                height = bottom - top
                return (left, top, width, height)

            return None

        except ImportError:
            print("需要安装pywin32: pip install pywin32")
            return None

        except Exception as e:
            print(f"获取窗口位置失败: {e}")
            return None

    def is_dialogue_screen(self, img: np.ndarray) -> bool:
        """检测是否为对话画面

        Args:
            img: 截图图像

        Returns:
            bool: 是否为对话画面
        """
        if img is None:
            return False

        # 简单的对话画面检测方法
        # 1. 检查画面底部是否有对话框（通常对话框有特定的颜色特征）
        # 2. 可以使用模板匹配或颜色分析

        # 这里使用简单的颜色分析：
        # 检查画面底部区域是否有大量白色/浅色（对话框背景）
        height, width = img.shape[:2]
        bottom_region = img[int(height * 0.7):, :]

        # 计算底部区域的平均亮度
        gray = np.mean(bottom_region, axis=2)

        # 如果平均亮度超过阈值，可能是对话框
        avg_brightness = np.mean(gray)

        # 根据实际游戏画面调整这个阈值
        return avg_brightness > self.dialogue_color_threshold

    def skip_dialogue(self) -> None:
        """跳过对话（按Z键）"""
        if self.keyboard is None:
            from .keyboard_controller import KeyboardController
            self.keyboard = KeyboardController()

        self.keyboard.press_z()
        time.sleep(0.1)

    def auto_restart(self, check_interval: float = 2.0) -> None:
        """自动重启游戏

        检测到游戏结束后自动重启

        Args:
            check_interval: 检查间隔（秒）
        """
        while True:
            if not self.is_running() and not self.is_process_alive():
                print("检测到游戏已结束，准备重启...")
                time.sleep(1)
                self.launch_game()
                time.sleep(3)  # 等待游戏启动

            time.sleep(check_interval)

    def get_game_window_handle(self):
        """获取游戏窗口句柄

        Returns:
            窗口句柄或None
        """
        try:
            import win32gui

            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    for name in self.WINDOW_NAMES:
                        if name in title:
                            windows.append(hwnd)
                            break
                return True

            windows = []
            win32gui.EnumWindows(callback, windows)

            if windows:
                return windows[0]

            return None

        except ImportError:
            return None

    def bring_to_front(self) -> bool:
        """将游戏窗口置顶

        Returns:
            bool: 是否成功
        """
        try:
            import win32gui
            import win32con

            hwnd = self.get_game_window_handle()
            if hwnd:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return True

            return False

        except ImportError:
            return False

    def __del__(self):
        """析构时关闭游戏"""
        self.terminate_game()


# 测试代码
if __name__ == "__main__":
    print("游戏管理器测试")
    print("开始自动测试...")

    manager = GameManager()

    print("测试1: 启动游戏")
    if manager.launch_game():
        print("游戏启动成功")

        time.sleep(2)

        print("测试2: 截图")
        img = manager.capture_screen()
        if img is not None:
            print(f"截图成功，尺寸: {img.shape}")
            # 保存截图
            import cv2
            cv2.imwrite("test_screenshot.png", img)
            print("截图已保存到 test_screenshot.png")

        print("测试3: 检查是否对话画面")
        if img is not None:
            is_dialogue = manager.is_dialogue_screen(img)
            print(f"是否对话画面: {is_dialogue}")

        print("等待5秒后关闭游戏...")
        time.sleep(5)

        manager.terminate_game()
    else:
        print("游戏启动失败")
