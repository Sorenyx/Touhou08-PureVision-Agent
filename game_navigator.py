"""
游戏导航器 - 自动导航游戏界面，实现完整的训练循环
"""

import time
import cv2
import numpy as np
from enum import Enum
from typing import Optional, Tuple
from pathlib import Path


class GameState(Enum):
    """游戏状态枚举"""
    TITLE = "title"                    # 标题界面
    CHARACTER_SELECT = "character"      # 角色选择界面
    DIFFICULTY_SELECT = "difficulty"    # 难度选择界面
    DIALOGUE = "dialogue"              # 对话画面
    PLAYING = "playing"                # 游戏进行中
    SCORE_SCREEN = "score"             # 结算画面
    DEAD = "dead"                      # 死亡页面
    BAD_END = "badend"                 # Bad End页面（11次重开后）
    REPLAY_SAVE = "replay"             # 保存录像页面
    PAUSE_MENU = "pause"               # 暂停菜单
    QUIT_MENU = "quit"                 # 退出菜单
    UNKNOWN = "unknown"                # 未知状态


class GameNavigator:
    """游戏界面导航器

    自动识别游戏界面状态，并执行导航操作
    """

    def __init__(self, game_manager, keyboard_controller, yolo_processor=None):
        """初始化导航器

        Args:
            game_manager: 游戏管理器实例
            keyboard_controller: 键盘控制器实例
            yolo_processor: YOLO处理器实例（可选，用于状态检测）
        """
        self.game_manager = game_manager
        self.keyboard = keyboard_controller
        self.yolo_processor = yolo_processor

        # 模板图像缓存
        self.templates = {}

        # 输入法切换标志（只在首次进入游戏时切换）
        self._ime_switched = False

        # 难度选择标志（只在首次选择时按下键到Lunatic，后续保持上一次难度）
        self._difficulty_selected_once = False

        # 导航锁（防止重入和重复按键）
        self._navigating = False
        self._character_selected = False  # 本轮导航是否已选择人物

        # 防抖：记录上次导航时间
        self._last_navigate_time = 0
        self._min_navigate_interval = 2.0  # 最小导航间隔（秒）

        # playing 检测锁：只有进入过人物选择页后才能检测为 playing
        self._entered_character_once = False

        # 界面颜色特征（用于快速识别）
        self.state_colors = {
            GameState.TITLE: {
                'bottom_avg': 80,      # 标题界面底部较暗
                'center_avg': 150      # 中心有标题Logo
            },
            GameState.CHARACTER_SELECT: {
                'bottom_avg': 180,     # 角色选择界面底部较亮
                'has_portrait': True   # 有角色肖像
            },
            GameState.DIFFICULTY_SELECT: {
                'bottom_avg': 160,     # 难度选择界面
                'text_density': 'high'
            },
            GameState.PLAYING: {
                'bottom_avg': 60,      # 游戏界面底部有弹幕
                'has_player': True     # 检测到玩家
            },
            GameState.SCORE_SCREEN: {
                'bottom_avg': 200,     # 结算画面很亮
                'has_large_text': True # 有大字分数
            },
            GameState.QUIT_MENU: {
                'bottom_avg': 140      # 菜单界面
            }
        }

        # 模板图像目录
        self.template_dir = Path(__file__).parent / "templates"

    def detect_state(self, screen: Optional[np.ndarray] = None,
                    max_retries: int = 3) -> GameState:
        """检测当前游戏状态

        Args:
            screen: 截图，如果为None则自动截图
            max_retries: 最大重试次数

        Returns:
            GameState: 当前游戏状态
        """
        for _ in range(max_retries):
            if screen is None:
                screen = self.game_manager.capture_screen()

            if screen is None:
                time.sleep(0.5)
                continue

            # 优先使用YOLO检测
            if self.yolo_processor is not None:
                state = self._detect_by_yolo(screen)
                if state != GameState.UNKNOWN:
                    return state

            # 回退到颜色检测
            state = self._detect_by_color(screen)

            if state != GameState.UNKNOWN:
                return state

            time.sleep(0.2)

        return GameState.UNKNOWN

    def _detect_by_yolo(self, screen: np.ndarray) -> GameState:
        """通过YOLO检测状态

        Args:
            screen: 截图（BGR格式）

        Returns:
            GameState: 检测到的状态
        """
        if self.yolo_processor is None:
            return GameState.UNKNOWN

        state_name, conf = self.yolo_processor.detect_state(screen)

        # 映射YOLO检测结果到GameState
        state_mapping = {
            'start': GameState.TITLE,
            'title': GameState.TITLE,
            'character': GameState.CHARACTER_SELECT,
            'difficulty': GameState.DIFFICULTY_SELECT,
            'dialogue': GameState.DIALOGUE,
            'playing': GameState.PLAYING,
            'score': GameState.SCORE_SCREEN,
            'dead': GameState.DEAD,
            'badend': GameState.BAD_END,
            'replay': GameState.REPLAY_SAVE,
            'stage2': GameState.PLAYING,
        }

        return state_mapping.get(state_name, GameState.UNKNOWN)

    def _detect_by_color(self, screen: np.ndarray) -> GameState:
        """通过颜色特征检测状态（使用HSV色彩空间）

        Args:
            screen: 截图（BGR格式）

        Returns:
            GameState: 检测到的状态
        """
        h, w = screen.shape[:2]

        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)

        # 分析画面不同区域的特征
        # 1. 底部区域（通常有菜单或弹幕）
        bottom_region = screen[int(h * 0.8):, :]
        bottom_avg = np.mean(bottom_region)

        # 2. 中心区域
        center_region = screen[int(h * 0.3):int(h * 0.7), :]
        center_avg = np.mean(center_region)

        # 3. 顶部区域
        top_region = screen[:int(h * 0.2), :]
        top_avg = np.mean(top_region)

        # ========== HSV颜色检测 ==========

        # 检测深红色（难度/人物选择背景）
        # 红色在HSV中有两个范围：0-10 和 170-180
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (15, 255, 180))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 180))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / (h * w)

        # 检测紫色（对话/结算背景）- 根据实际数据H在100-145
        purple_mask = cv2.inRange(hsv, (100, 100, 30), (150, 255, 150))
        purple_ratio = np.sum(purple_mask > 0) / (h * w)

        # 检测亮红色（结算分数）
        bright_red_mask = cv2.inRange(hsv, (0, 100, 150), (10, 255, 255))
        bright_red_ratio = np.sum(bright_red_mask > 0) / (h * w)

        # 检测白色（选择框）
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        white_ratio = np.sum(white_mask > 0) / (h * w)

        # 检测底部紫色（对话特征）
        bottom_hsv = hsv[int(h * 0.83):, int(w * 0.1):int(w * 0.9)]
        bottom_purple_mask = cv2.inRange(bottom_hsv, (100, 100, 30), (150, 255, 150))
        bottom_purple_ratio = np.sum(bottom_purple_mask > 0) / bottom_purple_mask.size

        # ========== 状态判断（顺序很重要！）==========

        # 计算整体平均亮度
        overall_avg = (bottom_avg + center_avg + top_avg) / 3

        # 1. 检查死亡页面（整体极暗）
        if overall_avg < 30:
            return GameState.DEAD

        # 2. 检查Bad End页面（白色比例高，亮度高）
        if white_ratio > 0.50 and overall_avg > 150:
            return GameState.BAD_END

        # 3. 紫色系界面（对话/分数/二面/回复/保存录像）
        # 对话/游戏中：紫色通常很高（实时数据50%+，静态35%+）
        if purple_ratio > 0.35:
            # 对话画面通常还有一定红色（人物立绘），而游戏中红色极低
            if red_ratio > 0.03:
                return GameState.DIALOGUE
            # 分数结算：亮度较低，白色较少
            if center_avg < 100:
                return GameState.SCORE_SCREEN
            # 否则是游戏中（二面等）
            return GameState.PLAYING

        # 保存录像：禁用此检测，不再需要
        # if 0.10 < purple_ratio <= 0.35:
        #     return GameState.REPLAY_SAVE

        # 4. 红色系界面（标题/人物/难度）
        if red_ratio > 0.20:
            # 难度选择：整体更暗（底部+中心显著低于人物选择）
            if bottom_avg < 70 and center_avg < 80:
                return GameState.DIFFICULTY_SELECT

            # 标题：中心和底部都偏亮，同时白色不要太夸张（避免被人物选择误伤）
            if center_avg > 100 and bottom_avg > 80 and white_ratio < 0.25:
                return GameState.TITLE

            # 其他红色界面默认人物选择
            return GameState.CHARACTER_SELECT

        # 5. 兜底：优先判断标题页（避免被误判为 playing）
        # 标题页特征：有一定红色 + 中心偏亮 或 顶部亮
        if red_ratio > 0.05:
            if center_avg > 80 or top_avg > 80:
                return GameState.TITLE

        # 6. 游戏进行中（必须先进入过人物选择页才能判定）
        # 这样可以避免标题页被误判为 playing
        if self._entered_character_once:
            if bottom_avg < 80 and center_avg > 80 and red_ratio < 0.05:
                return GameState.PLAYING

        # 7. 兜底：标题（中心暗 + 顶部亮）
        if center_avg < 80 and top_avg > 80:
            return GameState.TITLE

        return GameState.UNKNOWN

    def navigate_to_game_start(self) -> bool:
        """从当前界面导航到游戏开始

        智能检测当前状态，从正确的步骤开始导航。

        流程：
        1. 检测当前状态
        2. 根据当前状态从正确步骤开始
        3. 标题 -> 难度选择 -> 人物选择 -> 开始游戏

        Returns:
            bool: 是否成功
        """
        # 防止重入
        if self._navigating:
            print("导航正在进行中，跳过重复调用")
            return True

        # 防抖：检查最小导航间隔
        current_time = time.time()
        if current_time - self._last_navigate_time < self._min_navigate_interval:
            print("导航间隔过短，跳过")
            return True

        self._navigating = True
        self._character_selected = False  # 重置人物选择标志
        self._last_navigate_time = current_time

        try:
            return self._do_navigate()
        finally:
            self._navigating = False

    def _do_navigate(self) -> bool:
        """实际执行导航的内部方法"""
        print("开始导航到游戏...")

        # 确保游戏窗口在前台
        if not self.game_manager.bring_to_front():
            print("警告：无法将游戏窗口置顶")

        # 只在首次进入游戏时切换输入法（避免重复切换）
        if not self._ime_switched:
            print("切换输入法到英文模式...")
            self.keyboard.switch_ime_ctrl_space()
            self._ime_switched = True
            time.sleep(0.2)

        # 智能检测当前状态
        current_state = self.detect_state()
        current_state_name = current_state.value if hasattr(current_state, 'value') else str(current_state)
        print(f"当前检测状态: {current_state_name}")

        # 根据当前状态决定从哪一步开始
        if current_state_name == 'playing':
            print("已在游戏中，无需导航")
            return True
        elif current_state_name == 'dialogue':
            print("在对话中，跳过对话...")
            self.skip_dialogue()
            return True
        elif current_state_name == 'character':
            # 在人物选择界面，直接按Z选择
            print("在人物选择界面，选择结界组...")
            self._entered_character_once = True
            time.sleep(0.5)
            self.keyboard.press_z()
            self._character_selected = True  # 标记已选择人物
            time.sleep(1.0)
            # 直接跳到等待游戏开始
            return self._wait_and_confirm_game_start()
        elif current_state_name == 'difficulty':
            # 在难度选择界面
            print("在难度选择界面...")
            time.sleep(0.8)
            # 简单模式训练：不移动光标，直接确认当前默认难度
            self._difficulty_selected_once = True
            print("确认难度...")
            self.keyboard.press_z()
            time.sleep(1.0)
            # 继续等待人物选择界面
        elif current_state_name == 'title':
            # 在标题界面，正常流程
            print("在标题界面...")
            time.sleep(0.5)
            # 按Z进入难度选择
            print("进入难度选择...")
            self.keyboard.press_z()
            time.sleep(1.0)

            # 等待难度选择界面
            if not self._wait_for_state(GameState.DIFFICULTY_SELECT, timeout=8):
                print("未检测到难度选择界面，尝试继续...")
            time.sleep(0.8)

            # 选择难度（简单模式训练：不移动光标，直接确认默认难度）
            self._difficulty_selected_once = True

            # 确认难度
            print("确认难度...")
            self.keyboard.press_z()
            time.sleep(1.0)
        else:
            # 未知状态或其他状态，等待一下重新检测
            print(f"未知状态: {current_state_name}，等待重新检测...")
            time.sleep(1.0)

            # 重新检测状态
            current_state = self.detect_state()
            current_state_name = current_state.value if hasattr(current_state, 'value') else str(current_state)
            print(f"重新检测状态: {current_state_name}")

            # 如果还是未知，尝试按Z进入下一个界面（假设在标题页）
            if current_state_name in ['unknown', 'title']:
                print("假设在标题界面，按Z继续...")
                self.keyboard.press_z()
                time.sleep(1.5)
            else:
                # 递归调用内部方法
                return self._do_navigate()

        # 等待角色选择界面
        print("等待角色选择界面...")
        if self._wait_for_state(GameState.CHARACTER_SELECT, timeout=8):
            self._entered_character_once = True  # 进入过人物选择，之后才能检测为 playing
        else:
            print("未检测到角色选择界面，尝试继续...")
        time.sleep(0.5)

        # 防止在同一轮导航中重复选择人物（导致连续按右/循环）
        if not self._character_selected:
            # 按Z选择结界组
            print("选择结界组...")
            self.keyboard.press_z()
            self._character_selected = True
            time.sleep(1.0)
        else:
            print("人物已选择，跳过重复按键")

        return self._wait_and_confirm_game_start()

    def reset_to_title(self) -> bool:
        """返回标题界面

        流程：
        1. 按ESC打开菜单
        2. 按方向键选择Quit
        3. 按Z确认
        4. 等待返回标题

        Returns:
            bool: 是否成功
        """
        print("开始返回标题界面...")

        # 步骤1: 按ESC打开菜单
        print("步骤1: 打开菜单...")
        self.keyboard.press_esc()
        time.sleep(0.5)

        # 步骤2: 按方向键向下选择Quit（通常在第2个位置）
        print("步骤2: 选择Quit...")
        self.keyboard.press_arrow('down')  # 按1次下，选中第二个选项
        time.sleep(0.3)

        # 步骤3: 按Z确认
        print("步骤3: 确认退出...")
        self.keyboard.press_z()
        time.sleep(0.5)

        # 步骤4: 等待返回标题
        print("步骤4: 等待返回标题...")
        if not self._wait_for_state(GameState.TITLE, timeout=10):
            print("未返回标题界面")
            return False

        print("已返回标题界面")
        return True

    def _wait_and_confirm_game_start(self) -> bool:
        """等待并确认游戏已开始

        Returns:
            bool: 是否成功进入游戏
        """
        # 等待游戏开始
        print("等待游戏开始...")
        time.sleep(1.0)

        # 检测是否进入对话或游戏
        screen = self.game_manager.capture_screen()
        if screen is not None:
            if self.yolo_processor is not None:
                state = self._detect_by_yolo(screen)
            else:
                state = self._detect_by_color(screen)

            if state == GameState.DIALOGUE:
                print("检测到对话，开始跳过...")
                if not self.skip_dialogue():
                    print("对话跳过可能未完成")
            elif state == GameState.PLAYING:
                print("直接进入游戏")

        # 最终确认
        if not self._wait_for_state(GameState.PLAYING, timeout=5):
            print("警告: 未检测到游戏界面，但继续执行")

        print("导航完成！游戏已开始")
        return True

    def _wait_for_state(self, target_state: GameState,
                       timeout: float = 10.0) -> bool:
        """等待特定状态

        Args:
            target_state: 目标状态
            timeout: 超时时间（秒）

        Returns:
            bool: 是否检测到目标状态
        """
        start_time = time.time()
        check_interval = 0.2

        while time.time() - start_time < timeout:
            current_state = self.detect_state()

            if current_state == target_state:
                return True

            time.sleep(check_interval)

        return False

    def is_score_screen(self, screen: Optional[np.ndarray] = None) -> bool:
        """检查是否是结算画面

        Args:
            screen: 截图，如果为None则自动截图

        Returns:
            bool: 是否是结算画面
        """
        state = self.detect_state(screen)
        return state == GameState.SCORE_SCREEN

    def is_dialogue(self, screen: Optional[np.ndarray] = None) -> bool:
        """检查是否是对话画面

        Args:
            screen: 截图，如果为None则自动截图

        Returns:
            bool: 是否是对话画面
        """
        state = self.detect_state(screen)
        return state == GameState.DIALOGUE

    def skip_dialogue(self, max_skips: int = 15, skip_delay: float = 0.3) -> bool:
        """跳过对话画面

        在检测到对话界面时持续按Z键跳过，直到进入游戏

        Args:
            max_skips: 最大跳过次数
            skip_delay: 每次跳过的间隔时间（秒）

        Returns:
            bool: 是否成功跳过所有对话
        """
        print("跳过对话...")
        consecutive_non_dialogue = 0

        for i in range(max_skips):
            screen = self.game_manager.capture_screen()
            if screen is None:
                time.sleep(skip_delay)
                continue

            state = self._detect_by_color(screen)

            if state == GameState.DIALOGUE:
                # 仍然是对话，按Z继续
                self.keyboard.press_z()
                consecutive_non_dialogue = 0
                print(f"  跳过对话 {i+1}/{max_skips}")
            elif state == GameState.PLAYING:
                # 进入游戏了
                consecutive_non_dialogue += 1
                if consecutive_non_dialogue >= 2:
                    print("对话跳过完成，已进入游戏")
                    return True
            else:
                # 其他状态（可能在加载中）
                consecutive_non_dialogue += 1

            time.sleep(skip_delay)

        print(f"跳过对话结束（达到最大次数 {max_skips}）")
        return False

    def is_playing(self, screen: Optional[np.ndarray] = None) -> bool:
        """检查是否正在游戏

        Args:
            screen: 截图，如果为None则自动截图

        Returns:
            bool: 是否正在游戏
        """
        state = self.detect_state(screen)
        return state == GameState.PLAYING

    def is_dead(self, screen: Optional[np.ndarray] = None) -> bool:
        """检查是否死亡

        Args:
            screen: 截图，如果为None则自动截图

        Returns:
            bool: 是否死亡
        """
        state = self.detect_state(screen)
        return state == GameState.DEAD

    def is_bad_end(self, screen: Optional[np.ndarray] = None) -> bool:
        """检查是否是Bad End

        Args:
            screen: 截图，如果为None则自动截图

        Returns:
            bool: 是否是Bad End
        """
        state = self.detect_state(screen)
        return state == GameState.BAD_END

    def handle_death(self, retry_count: int = 0) -> str:
        """处理死亡后的流程

        流程：
        1. 死亡后按Z进入人物志/分数榜页面
        2. 再按Z进入保存replay询问页面
        3. 按方向键右选择"算了"再按Z回到标题页

        Args:
            retry_count: 当前重开次数

        Returns:
            str: 返回结果 ("title", "badend", "unknown")
        """
        print(f"处理死亡... (重开次数: {retry_count})")

        # 确保游戏窗口在前台
        self.game_manager.bring_to_front()
        time.sleep(0.3)

        # RL 环境里 Z 可能一直处于按住状态；先释放，保证后续 press_z 有“按下沿”
        self.keyboard.stop_movement()
        self.keyboard.release_z()
        time.sleep(0.1)

        # 等待死亡动画结束
        time.sleep(3.0)

        # 步骤1: 在死亡画面按Z（进入下一个页面）
        print("步骤1: 按Z继续...")
        time.sleep(3.0)
        self.keyboard.press_z(duration=0.2)
        time.sleep(0.8)
        # 如果还停留在死亡页面，重试一次（有时第一下会被动画/焦点吞掉）
        state_after = self.detect_state()
        state_after_name = state_after.value if hasattr(state_after, 'value') else str(state_after)
        if state_after_name == 'dead':
            print("步骤1: 仍在死亡画面，重试按Z...")
            self.keyboard.press_z(duration=0.2)
        time.sleep(2.0)

        # 步骤2: 再按Z进入保存replay询问页面
        print("步骤2: 按Z进入保存replay询问...")
        time.sleep(0.8)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤3: 按右选择"算了"再按Z返回标题
        print("步骤3: 按右选择算了，再按Z返回标题...")
        time.sleep(0.8)
        self.keyboard.press_arrow('right')
        time.sleep(0.5)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤4: 等待返回标题
        print("步骤4: 等待返回标题...")
        if self._wait_for_state(GameState.TITLE, timeout=10):
            print("已返回标题")
            return "title"

        # 检查是否是Bad End
        screen = self.game_manager.capture_screen()
        if screen is not None:
            state = self._detect_by_color(screen)
            if state == GameState.BAD_END:
                print("检测到Bad End")
                return "badend"

        return "unknown"

    def handle_bad_end(self) -> bool:
        """处理Bad End流程

        流程（与死亡后类似）：
        1. 在Bad End页面按Z进入人物志画面
        2. 再按Z进入分数榜页面
        3. 再按Z进入保存replay询问页面
        4. 按方向键右选择"算了"再按Z回到标题页

        Returns:
            bool: 是否成功返回标题
        """
        print("处理Bad End流程...")

        # 步骤1: 在Bad End页面按Z进入人物志画面
        print("步骤1: 按Z进入人物志...")
        time.sleep(1.0)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤2: 在人物志画面按Z进入分数榜页面
        print("步骤2: 按Z进入分数榜...")
        time.sleep(1.0)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤3: 在分数榜页面按Z进入保存replay询问页面
        print("步骤3: 按Z进入保存replay询问...")
        time.sleep(1.0)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤4: 在保存录像页面按右选择"算了"再按Z
        print("步骤4: 按右选择算了，再按Z返回标题...")
        time.sleep(1.0)
        self.keyboard.press_arrow('right')
        time.sleep(0.5)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤5: 等待返回标题
        print("步骤5: 等待返回标题...")
        if self._wait_for_state(GameState.TITLE, timeout=10):
            print("已返回标题")
            return True
        else:
            print("未能返回标题")
            return False

    def handle_score_screen(self, continue_to_next: bool = False) -> str:
        """处理结算画面

        流程：
        1. 按Z进入人物志画面
        2. 再按Z进入分数榜页面
        3. 再按Z进入保存replay询问页面
        4. 按方向键右选择"算了"再按Z回到标题页

        Args:
            continue_to_next: 是否继续到下一关（暂不支持）

        Returns:
            str: 返回结果 ("title", "next_stage", "unknown")
        """
        print("处理结算画面...")

        if continue_to_next:
            # 按Z进入下一关（暂不支持，按正常流程返回标题）
            pass

        # 步骤1: 在结算画面按Z进入人物志画面
        print("步骤1: 按Z进入人物志...")
        time.sleep(1.0)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤2: 在人物志画面按Z进入分数榜页面
        print("步骤2: 按Z进入分数榜...")
        time.sleep(1.0)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤3: 在分数榜页面按Z进入保存replay询问页面
        print("步骤3: 按Z进入保存replay询问...")
        time.sleep(1.0)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 步骤4: 在保存录像页面按右选择"算了"再按Z
        print("步骤4: 按右选择算了，再按Z返回标题...")
        time.sleep(1.0)
        self.keyboard.press_arrow('right')
        time.sleep(0.5)
        self.keyboard.press_z()
        time.sleep(2.0)

        # 等待返回标题
        if self._wait_for_state(GameState.TITLE, timeout=10):
            return "title"

        return "unknown"

    def get_current_state(self) -> GameState:
        """获取当前状态

        Returns:
            GameState: 当前状态
        """
        return self.detect_state()


# 测试代码
if __name__ == "__main__":
    print("游戏导航器测试")

    from .game_manager import GameManager
    from .keyboard_controller import KeyboardController

    # 创建管理器和键盘控制器
    manager = GameManager()
    keyboard = KeyboardController()

    # 创建导航器
    navigator = GameNavigator(manager, keyboard)

    try:
        # 启动游戏
        print("启动游戏...")
        if not manager.launch_game():
            print("游戏启动失败")
            exit(1)

        time.sleep(3)

        # 测试状态检测
        print("\n测试状态检测...")
        for i in range(5):
            state = navigator.detect_state()
            print(f"当前状态: {state.value}")
            time.sleep(1)

        # 测试导航到游戏开始
        print("\n测试导航到游戏开始...")
        if navigator.navigate_to_game_start():
            print("导航成功！")

            time.sleep(3)

            # 测试返回标题
            print("\n测试返回标题...")
            if navigator.reset_to_title():
                print("返回成功！")
            else:
                print("返回失败")
        else:
            print("导航失败")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        manager.terminate_game()