"""
键盘控制器 - 用于控制东方永夜抄游戏
使用PostMessage直接向游戏窗口发送消息，完全避免输入法干扰
"""

import time
import ctypes
from ctypes import wintypes
from typing import Optional
from threading import Lock

# Windows API
user32 = ctypes.windll.user32

# 虚拟键码
VK_Z = 0x5A
VK_X = 0x58
VK_UP = 0x26
VK_DOWN = 0x28
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_SHIFT = 0x10
VK_ESCAPE = 0x1B
VK_CONTROL = 0x11
VK_SPACE = 0x20

# Windows消息
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102

# 扫描码
SCAN_Z = 0x2C
SCAN_X = 0x2D
SCAN_UP = 0x48
SCAN_DOWN = 0x50
SCAN_LEFT = 0x4B
SCAN_RIGHT = 0x4D
SCAN_SHIFT = 0x2A
SCAN_ESCAPE = 0x01
SCAN_CONTROL = 0x1D
SCAN_SPACE = 0x39

# keybd_event标志
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002


class KeyboardController:
    """东方永夜抄键盘控制器

    使用keybd_event发送按键，并通过扫描码确保不受输入法影响
    """

    KEYS = {
        'up': (VK_UP, SCAN_UP),
        'down': (VK_DOWN, SCAN_DOWN),
        'left': (VK_LEFT, SCAN_LEFT),
        'right': (VK_RIGHT, SCAN_RIGHT),
        'z': (VK_Z, SCAN_Z),
        'x': (VK_X, SCAN_X),
        'shift': (VK_SHIFT, SCAN_SHIFT),
        'esc': (VK_ESCAPE, SCAN_ESCAPE)
    }

    def __init__(self):
        self._lock = Lock()
        self._pressed_keys = set()

    def _keybd_event(self, vk_code: int, scan_code: int, key_up: bool = False) -> None:
        """发送键盘事件，使用扫描码

        Args:
            vk_code: 虚拟键码
            scan_code: 扫描码
            key_up: 是否为释放按键
        """
        flags = KEYEVENTF_KEYUP if key_up else KEYEVENTF_KEYDOWN
        # 使用扫描码发送，绕过输入法
        user32.keybd_event(vk_code, scan_code, flags, 0)

    def press_arrow(self, direction: str, duration: float = 0.1) -> None:
        """按下方向键"""
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError(f"Invalid direction: {direction}")

        with self._lock:
            vk, scan = self.KEYS[direction]
            self._keybd_event(vk, scan, False)
            self._pressed_keys.add(direction)
            time.sleep(duration)
            self._keybd_event(vk, scan, True)
            self._pressed_keys.discard(direction)

    def hold_arrow(self, direction: str) -> None:
        """按住方向键"""
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError(f"Invalid direction: {direction}")

        with self._lock:
            if direction not in self._pressed_keys:
                vk, scan = self.KEYS[direction]
                self._keybd_event(vk, scan, False)
                self._pressed_keys.add(direction)

    def release_arrow(self, direction: str) -> None:
        """释放方向键"""
        if direction not in ['up', 'down', 'left', 'right']:
            raise ValueError(f"Invalid direction: {direction}")

        with self._lock:
            if direction in self._pressed_keys:
                vk, scan = self.KEYS[direction]
                self._keybd_event(vk, scan, True)
                self._pressed_keys.discard(direction)

    def press_z(self, duration: float = 0.1) -> None:
        """按下Z键"""
        with self._lock:
            vk, scan = self.KEYS['z']
            self._keybd_event(vk, scan, False)
            time.sleep(duration)
            self._keybd_event(vk, scan, True)

    def hold_z(self) -> None:
        """按住Z键（强制模式：每次都发送keydown确保Z被按住）"""
        with self._lock:
            # 强制发送 keydown，不依赖 _pressed_keys 状态
            # 因为 press_z() 会释放 Z 但不更新 _pressed_keys
            vk, scan = self.KEYS['z']
            self._keybd_event(vk, scan, False)
            self._pressed_keys.add('z')

    def release_z(self) -> None:
        """释放Z键"""
        with self._lock:
            if 'z' in self._pressed_keys:
                vk, scan = self.KEYS['z']
                self._keybd_event(vk, scan, True)
                self._pressed_keys.discard('z')

    def press_x(self, duration: float = 0.1) -> None:
        """按下X键"""
        with self._lock:
            vk, scan = self.KEYS['x']
            self._keybd_event(vk, scan, False)
            time.sleep(duration)
            self._keybd_event(vk, scan, True)

    def press_shift(self, hold: bool = True) -> None:
        """按下或释放Shift键"""
        with self._lock:
            vk, scan = self.KEYS['shift']
            if hold and 'shift' not in self._pressed_keys:
                self._keybd_event(vk, scan, False)
                self._pressed_keys.add('shift')
            elif not hold and 'shift' in self._pressed_keys:
                self._keybd_event(vk, scan, True)
                self._pressed_keys.discard('shift')

    def release_shift(self) -> None:
        """释放Shift键"""
        self.press_shift(hold=False)

    def switch_ime_ctrl_space(self) -> None:
        """发送 Ctrl+Space，用于切换中/英输入法（依赖系统/IME设置）"""
        with self._lock:
            # Ctrl down
            self._keybd_event(VK_CONTROL, SCAN_CONTROL, False)
            time.sleep(0.03)
            # Space down/up
            self._keybd_event(VK_SPACE, SCAN_SPACE, False)
            time.sleep(0.03)
            self._keybd_event(VK_SPACE, SCAN_SPACE, True)
            time.sleep(0.03)
            # Ctrl up
            self._keybd_event(VK_CONTROL, SCAN_CONTROL, True)
            time.sleep(0.08)

    def press_esc(self) -> None:
        """按下ESC键"""
        with self._lock:
            vk, scan = self.KEYS['esc']
            self._keybd_event(vk, scan, False)
            time.sleep(0.1)
            self._keybd_event(vk, scan, True)

    def reset_keys(self) -> None:
        """释放所有按键"""
        with self._lock:
            for key in list(self._pressed_keys):
                if key in self.KEYS:
                    vk, scan = self.KEYS[key]
                    self._keybd_event(vk, scan, True)
            self._pressed_keys.clear()

    def move(self, dx: int, dy: int) -> None:
        """根据移动向量控制方向

        Args:
            dx: X方向移动 (-1, 0, 1)
            dy: Y方向移动 (-1, 0, 1)
        """
        # 先释放所有方向键
        for direction in ['up', 'down', 'left', 'right']:
            self.release_arrow(direction)

        # 根据向量按下对应方向
        if dy < 0:
            self.hold_arrow('up')
        elif dy > 0:
            self.hold_arrow('down')

        if dx < 0:
            self.hold_arrow('left')
        elif dx > 0:
            self.hold_arrow('right')

    def stop_movement(self) -> None:
        """停止所有移动"""
        for direction in ['up', 'down', 'left', 'right']:
            self.release_arrow(direction)

    def move_continuous(self, dx: float, dy: float, base_duration: float = 0.016) -> None:
        """连续移动 - 根据dx, dy值按下对应方向键指定时长

        Args:
            dx: X方向移动量 [-1, 1]，正值向右，负值向左
            dy: Y方向移动量 [-1, 1]，正值向下，负值向上
            base_duration: 基础按压时长（秒）
        """
        # 先释放所有方向键
        self.stop_movement()

        # 计算需要按下的方向和时长
        directions_to_press = []

        if abs(dx) > 0.05:  # 死区阈值，避免微小抖动
            duration_x = abs(dx) * base_duration
            if dx < 0:
                directions_to_press.append(('left', duration_x))
            else:
                directions_to_press.append(('right', duration_x))

        if abs(dy) > 0.05:
            duration_y = abs(dy) * base_duration
            if dy < 0:
                directions_to_press.append(('up', duration_y))
            else:
                directions_to_press.append(('down', duration_y))

        # 按下方向键
        for direction, _ in directions_to_press:
            vk, scan = self.KEYS[direction]
            self._keybd_event(vk, scan, False)
            self._pressed_keys.add(direction)

        # 等待计算的最大时长（支持斜向移动）
        max_duration = max((d for _, d in directions_to_press), default=0)
        if max_duration > 0:
            time.sleep(max_duration)

        # 释放方向键
        for direction, _ in directions_to_press:
            vk, scan = self.KEYS[direction]
            self._keybd_event(vk, scan, True)
            self._pressed_keys.discard(direction)

    def hold_arrows_for_duration(self, dx: float, dy: float, duration: float) -> None:
        """按住方向键指定时长（用于连续动作控制）

        Args:
            dx: X方向 [-1, 1]
            dy: Y方向 [-1, 1]
            duration: 按住时长（秒）
        """
        # 先释放所有方向键
        self.stop_movement()

        # 按下需要的方向
        with self._lock:
            if dx < -0.05:
                vk, scan = self.KEYS['left']
                self._keybd_event(vk, scan, False)
                self._pressed_keys.add('left')
            elif dx > 0.05:
                vk, scan = self.KEYS['right']
                self._keybd_event(vk, scan, False)
                self._pressed_keys.add('right')

            if dy < -0.05:
                vk, scan = self.KEYS['up']
                self._keybd_event(vk, scan, False)
                self._pressed_keys.add('up')
            elif dy > 0.05:
                vk, scan = self.KEYS['down']
                self._keybd_event(vk, scan, False)
                self._pressed_keys.add('down')

        # 等待指定时长
        time.sleep(duration)

        # 释放方向键
        self.stop_movement()

    def __del__(self):
        """析构时确保释放所有按键"""
        self.reset_keys()


# 测试代码
if __name__ == "__main__":
    print("键盘控制器测试")
    print("等待5秒后开始测试...")

    controller = KeyboardController()

    time.sleep(5)

    print("测试1: 按方向键")
    controller.press_arrow('up')
    time.sleep(0.5)
    controller.press_arrow('down')
    time.sleep(0.5)
    controller.press_arrow('left')
    time.sleep(0.5)
    controller.press_arrow('right')

    print("测试2: 按Z键射击")
    controller.press_z()

    print("测试3: 按住Shift低速模式")
    controller.press_shift(True)
    time.sleep(1)
    controller.press_shift(False)

    print("测试完成")
