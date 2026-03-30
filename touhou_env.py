"""
东方永夜抄自定义Gymnasium环境
集成键盘控制、游戏管理、YOLO检测，实现PPO训练环境
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Optional, Dict
import time
from pathlib import Path
from collections import deque


class TouhouEnv(gym.Env):
    """东方永夜抄强化学习环境"""

    # 游戏状态
    STATE_DIALOGUE = 'dialogue'
    STATE_PLAYING = 'playing'
    STATE_GAME_OVER = 'game_over'

    # 移动速度（px/帧）
    MOVE_SPEED_NORMAL = 3.2
    MOVE_SPEED_SLOW = 1.6

    # Shift滞回控制（基于势场斥力Fmag）
    SHIFT_ON_THRESHOLD = 1.0
    SHIFT_OFF_THRESHOLD = 0.2

    # 屏幕参数
    SCREEN_WIDTH = 960
    SCREEN_HEIGHT = 720
    PLAYER_HITBOX_RADIUS = 5
    DEAD_ZONE = 0.05

    def __init__(self,
                 game_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 max_bullets: int = 50,
                 render_mode: Optional[str] = None,
                 auto_restart: bool = False,
                 yolo_infer_interval: int = 4):
        super().__init__()

        if game_path is None:
            project_root = Path(__file__).parent.parent
            game_path = project_root / "[th08] 东方永夜抄 (汉化版)" / "th08.exe"

        if model_path is None:
            model_path = project_root / "models" / "best.onnx"

        self.game_path = game_path
        self.model_path = model_path
        self.render_mode = render_mode
        self.auto_restart = auto_restart

        from .keyboard_controller import KeyboardController
        from .game_manager import GameManager
        from .yolo_processor import YOLOProcessor
        from .game_navigator import GameNavigator

        self.keyboard = KeyboardController()
        self.game_manager = GameManager(str(game_path))
        self.yolo_processor = YOLOProcessor(
            str(model_path),
            max_bullets=max_bullets,
            conf_threshold=0.08,
            img_width=960,
            img_height=720
        )
        self.navigator = GameNavigator(self.game_manager, self.keyboard, yolo_processor=self.yolo_processor)

        # 动作空间：连续动作 [dx, dy]，范围 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # 观测空间
        obs_dim = self.yolo_processor.obs_dim
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 环境状态
        self.current_state = self.STATE_PLAYING
        self.step_count = 0
        self.total_reward = 0.0
        self.max_episode_steps = 10000
        self._last_state_name = 'unknown'
        self._terminal_state_name = None
        self._terminal_state_count = 0

        # 玩家位置估计（用于边界约束）
        self._player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)

        # APF + Shift状态
        self._shift_pressed = False
        self._last_apf_force = 0.0
        self._force_danger = False

        # 预判弹幕缓存（P1-P5）
        self._predicted_bullets_p1 = []
        self._predicted_bullets_p2 = []
        self._predicted_bullets_p3 = []
        self._predicted_bullets_p4 = []
        self._predicted_bullets_p5 = []
        # 斜向预判（已禁用）
        self._predicted_bullets_pl1 = []
        self._predicted_bullets_pl2 = []
        self._predicted_bullets_pl3 = []
        self._predicted_bullets_pr1 = []
        self._predicted_bullets_pr2 = []
        self._predicted_bullets_pr3 = []

        # 空旷点引导状态
        self._empty_point = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.7], dtype=np.float32)
        self._stay_frames = 0

        # 移动历史
        self._move_history = deque(maxlen=3)
        self._last_player_pos = None

        # YOLO推理间隔
        self.yolo_infer_interval = yolo_infer_interval
        self._cached_obs = None
        self._cached_detections = None
        self._cached_state_name = 'unknown'
        self._steps_since_yolo = 0

        # 死亡检测与生命计数
        self.lives = 3
        self._player_history = deque(maxlen=4)
        self._death_reward_given = False
        self._final_death_reward_given = False
        self._pending_life_loss_reward = 0.0

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        if not self.game_manager.is_running():
            print("游戏未运行，正在启动...")
            if not self.game_manager.launch_game():
                raise RuntimeError("游戏启动失败")
            time.sleep(3)

        last_state = getattr(self, '_last_state_name', 'unknown')
        time.sleep(0.5)
        current_nav_state = self.navigator.detect_state()
        current_state_name = current_nav_state.value if hasattr(current_nav_state, 'value') else str(current_nav_state)
        print(f"当前检测状态: {current_state_name}")

        terminal_states = ['score', 'badend', 'dead']
        if current_state_name in terminal_states:
            if current_state_name == 'score':
                self.navigator.handle_score_screen(continue_to_next=False)
                time.sleep(1.5)
            elif current_state_name == 'badend':
                self.navigator.handle_bad_end()
                time.sleep(1.5)
            elif current_state_name == 'dead':
                self.navigator.handle_death()
                time.sleep(1.5)

        print("导航到游戏开始...")
        if not self.navigator.navigate_to_game_start():
            print("导航失败，等待后重试...")
            time.sleep(2)
            if not self.navigator.navigate_to_game_start():
                print("导航再次失败，继续尝试...")

        self.current_state = self.STATE_PLAYING
        self.step_count = 0
        self.total_reward = 0.0
        self._last_state_name = 'unknown'
        self._terminal_state_name = None
        self._terminal_state_count = 0
        self.navigator._entered_character_once = False

        self._cached_obs = None
        self._cached_detections = None
        self._cached_state_name = 'unknown'
        self._steps_since_yolo = 0

        self.lives = 3
        self._player_history.clear()
        self._death_reward_given = False
        self._final_death_reward_given = False
        self._pending_life_loss_reward = 0.0

        self._shift_pressed = False
        self._last_apf_force = 0.0
        self._force_danger = False
        self._predicted_bullets_p1 = []
        self._predicted_bullets_p2 = []
        self._predicted_bullets_p3 = []
        self._predicted_bullets_p4 = []
        self._predicted_bullets_p5 = []
        self._empty_point = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.7], dtype=np.float32)
        self._stay_frames = 0
        self._move_history.clear()
        self._last_player_pos = None
        self.keyboard.release_shift()
        self.keyboard.stop_movement()
        self.keyboard.hold_z()
        time.sleep(0.5)

        obs = self._get_observation()

        if self.render_mode == 'human':
            self.game_manager.bring_to_front()

        return obs, {'state': self.current_state, 'step': self.step_count, 'total_reward': self.total_reward}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        self.keyboard.hold_z()
        self._execute_action(action)
        time.sleep(0.016)

        obs = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_episode_steps

        survival_bonus = 0.0
        if terminated or truncated:
            if self.step_count >= 1400:
                survival_bonus = 100.0
            elif self.step_count >= 1200:
                survival_bonus = 20.0
            elif self.step_count >= 1000:
                survival_bonus = 10.0
            elif self.step_count >= 800:
                survival_bonus = 5.0

            if survival_bonus > 0:
                reward += survival_bonus

        self.total_reward += reward

        info = {
            'state': self.current_state,
            'step': self.step_count,
            'total_reward': self.total_reward,
            'lives': self.lives,
            'survival_bonus': survival_bonus
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: np.ndarray) -> None:
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)

        dx, dy = float(action[0]), float(action[1])
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)

        if self.current_state != self.STATE_PLAYING:
            self.keyboard.stop_movement()
            self.keyboard.hold_z()
            return

        # APF斥力计算
        apf_vector = np.array([0.0, 0.0])
        apf_magnitude = 0.0

        if self._cached_detections is not None and self._cached_detections['player'] is not None:
            # 只使用预判第3帧、第5帧弹幕
            predicted_for_apf = (self._predicted_bullets_p3 or []) + (self._predicted_bullets_p5 or [])
            apf_vector, apf_magnitude = self.yolo_processor.compute_apf_repulsion(
                self._cached_detections,
                predicted_bullets=predicted_for_apf
            )
            self._last_apf_force = apf_magnitude

        # 自动Shift切换
        if apf_magnitude > self.SHIFT_ON_THRESHOLD:
            if not self._shift_pressed:
                self._shift_pressed = True
                self.keyboard.press_shift(True)
        elif apf_magnitude < self.SHIFT_OFF_THRESHOLD:
            if self._shift_pressed:
                self._shift_pressed = False
                self.keyboard.press_shift(False)

        current_speed = self.MOVE_SPEED_SLOW if self._shift_pressed else self.MOVE_SPEED_NORMAL

        if abs(dx) < self.DEAD_ZONE:
            dx = 0.0
        if abs(dy) < self.DEAD_ZONE:
            dy = 0.0

        # 边界约束
        margin = self.PLAYER_HITBOX_RADIUS + 5
        estimated_new_x = self._player_pos[0] + dx * current_speed
        estimated_new_y = self._player_pos[1] + dy * current_speed

        if estimated_new_x < margin and dx < 0:
            dx = 0.0
        elif estimated_new_x > self.SCREEN_WIDTH - margin and dx > 0:
            dx = 0.0
        if estimated_new_y < margin and dy < 0:
            dy = 0.0
        elif estimated_new_y > self.SCREEN_HEIGHT - margin and dy > 0:
            dy = 0.0

        move_magnitude = np.sqrt(dx * dx + dy * dy)
        if move_magnitude > self.DEAD_ZONE:
            press_duration = move_magnitude * 0.016
            self.keyboard.hold_arrows_for_duration(dx, dy, press_duration)
        else:
            self.keyboard.stop_movement()

        self._move_history.append((dx, dy))

        actual_dx = dx * current_speed
        actual_dy = dy * current_speed
        self._player_pos[0] = np.clip(self._player_pos[0] + actual_dx, margin, self.SCREEN_WIDTH - margin)
        self._player_pos[1] = np.clip(self._player_pos[1] + actual_dy, margin, self.SCREEN_HEIGHT - margin)

        self.keyboard.hold_z()

    def _get_observation(self) -> np.ndarray:
        self._steps_since_yolo += 1

        need_yolo = (self._steps_since_yolo >= self.yolo_infer_interval or
                     self._cached_obs is None)

        if need_yolo:
            self._steps_since_yolo = 0
            img = self.game_manager.capture_screen()

            if img is None:
                if self._cached_obs is not None:
                    return self._cached_obs
                return np.zeros(self.observation_space.shape, dtype=np.float32)

            detections = self.yolo_processor.detect(img)
            state_name = detections['state']

            if state_name == 'dialogue':
                self.current_state = self.STATE_DIALOGUE
                self.keyboard.press_z()
                self.keyboard.hold_z()
            elif state_name == 'dead':
                self.current_state = self.STATE_GAME_OVER
            elif state_name in ['playing', 'stage2']:
                self.current_state = self.STATE_PLAYING
            elif state_name in ['score', 'badend']:
                self.current_state = self.STATE_GAME_OVER
            else:
                self.current_state = self.STATE_PLAYING

            self._last_state_name = state_name
            self._cached_state_name = state_name

            terminal_states = ['dead', 'score', 'badend']
            if state_name in terminal_states:
                if self._terminal_state_name == state_name:
                    self._terminal_state_count += 1
                else:
                    self._terminal_state_name = state_name
                    self._terminal_state_count = 1
            else:
                self._terminal_state_name = None
                self._terminal_state_count = 0

            obs = self.yolo_processor.to_observation(detections)
            self._cached_obs = obs
            self._cached_detections = detections

            # 弹幕跟踪与预判
            p1, p2, p3, p4, p5 = self.yolo_processor.update_bullet_tracks(detections)
            self._predicted_bullets_p1 = p1
            self._predicted_bullets_p2 = p2
            self._predicted_bullets_p3 = p3
            self._predicted_bullets_p4 = p4
            self._predicted_bullets_p5 = p5

            # 生命损失检测
            player_detected = detections['player'] is not None
            self._player_history.append(player_detected)

            collision_detected = False
            if player_detected and detections['bullets']:
                player_cx, player_cy = detections['player']
                for bullet in detections['bullets']:
                    real_dist = self.yolo_processor.compute_real_distance(
                        player_cx, player_cy, bullet['cx'], bullet['cy'], bullet['hitbox']
                    )
                    if real_dist < 0:
                        collision_detected = True
                        break

            if player_detected:
                self._player_pos[0] = float(detections['player'][0])
                self._player_pos[1] = float(detections['player'][1])

            # 死亡判定：前2帧有自机 + 后2帧消失 + 当前帧无自机
            if len(self._player_history) >= 4:
                history = list(self._player_history)
                first_two_have_player = history[-4] and history[-3]
                last_two_missing = (not history[-2]) or (not history[-1])
                current_missing = not history[-1]

                death_condition = first_two_have_player and last_two_missing and current_missing

                if death_condition and self.lives > 0 and not self._death_reward_given:
                    self.lives -= 1
                    self._pending_life_loss_reward = -100.0
                    self._death_reward_given = True
                    print(f"[生命损失] 剩余生命数: {self.lives}")

                if player_detected and self._death_reward_given:
                    self._death_reward_given = False

            return obs
        else:
            if self._cached_state_name in ['dead', 'score', 'badend']:
                if self._terminal_state_name == self._cached_state_name:
                    self._terminal_state_count += 1
                else:
                    self._terminal_state_name = self._cached_state_name
                    self._terminal_state_count = 1
            return self._cached_obs

    def _calculate_reward(self) -> float:
        reward = 0.0

        if self._pending_life_loss_reward != 0:
            reward += self._pending_life_loss_reward
            self._pending_life_loss_reward = 0.0

        if self._cached_detections is not None:
            predicted_for_apf = (self._predicted_bullets_p3 or []) + (self._predicted_bullets_p5 or [])
            reward += self.yolo_processor.compute_reward(
                self._cached_detections,
                predicted_bullets=predicted_for_apf
            )

        return reward

    def _check_terminated(self) -> bool:
        if not self.game_manager.is_running():
            return True

        terminal_state = self._terminal_state_name
        terminal_count = self._terminal_state_count

        if terminal_count < 2:
            return False

        if terminal_state == 'dead':
            if not self._final_death_reward_given:
                self._pending_life_loss_reward -= 20.0
                self._final_death_reward_given = True
            return True

        if terminal_state == 'score':
            return True

        if terminal_state == 'badend':
            return True

        if self.current_state == self.STATE_GAME_OVER and not self.auto_restart:
            return True

        return False

    def close(self) -> None:
        if self.keyboard:
            self.keyboard.stop_movement()
            self.keyboard.release_z()
            self.keyboard.reset_keys()

    def render(self) -> None:
        pass

    def seed(self, seed: int) -> None:
        np.random.seed(seed)


def make_touhou_env(**kwargs) -> gym.Env:
    return TouhouEnv(**kwargs)


if __name__ == "__main__":
    env = TouhouEnv(render_mode='human', auto_restart=False)
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
