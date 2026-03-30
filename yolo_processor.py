"""
YOLO数据处理器 - 东方永夜抄真实判定版本
"""

import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2


class YOLOProcessor:
    """YOLO检测结果处理器"""

    # 判定半径（像素）
    HITBOX = {
        'player': 1.65,
        'bullet_small': 5.0,
        'bullet_mid': 9.0,
        'bullet_heavy': 24.0,
    }

    # APF参数
    APF_RADIUS = 240.0       # d0: 势场有效半径
    APF_GAIN = 2000000.0     # k: 斥力放大系数

    # 类别定义
    CLASSES = {
        0: 'player',
        1: 'enemy',
        2: 'bullet_small',
        3: 'bullet_mid',
        4: 'bullet_heavy',
        5: 'start',
        6: 'dialogue',
        7: 'difficulty',
        8: 'character',
        9: 'replay',
        10: 'score',
        11: 'stage2',
        12: 'badend',
        13: 'dead',
    }

    BULLET_HITBOX = {
        'bullet_small': HITBOX['bullet_small'],
        'bullet_mid': HITBOX['bullet_mid'],
        'bullet_heavy': HITBOX['bullet_heavy'],
    }

    STATE_CLASSES = {
        'title': 5, 'dialogue': 6, 'difficulty': 7, 'character': 8,
        'replay': 9, 'score': 10, 'stage2': 11, 'badend': 12, 'dead': 13,
    }

    # 观测向量布局
    OBS_PLAYER_START = 0
    OBS_PLAYER_DIM = 2
    OBS_BULLET_STRIDE = 4

    def __init__(self, model_path: str,
                 max_bullets: int = 50,
                 conf_threshold: float = 0.15,
                 img_width: int = 960,
                 img_height: int = 720):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO模型不存在: {self.model_path}")

        self.max_bullets = max_bullets
        self.conf_threshold = conf_threshold
        self.img_width = img_width
        self.img_height = img_height

        print(f"正在加载YOLO模型: {self.model_path}")
        self.model = YOLO(str(self.model_path), task="detect")
        print("YOLO模型加载完成")

        self.obs_dim = self._calculate_obs_dim()

    def _calculate_obs_dim(self) -> int:
        return self.OBS_PLAYER_DIM + self.max_bullets * self.OBS_BULLET_STRIDE

    def get_bullet_start_index(self) -> int:
        return self.OBS_PLAYER_START + self.OBS_PLAYER_DIM

    def detect(self, image: np.ndarray) -> Dict:
        """执行YOLO检测"""
        results = self.model(image, verbose=False)

        detections = {
            'player': None,
            'bullets': [],
            'state': 'unknown',
            'state_conf': 0.0,
            'all_detections': []
        }

        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        sorted_indices = np.argsort(confidences)[::-1]
        state_detections = {}

        for idx in sorted_indices:
            conf = confidences[idx]
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = xyxy[idx]
            cls = classes[idx]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if cls == 0:  # player
                if detections['player'] is None:
                    detections['player'] = (center_x, center_y)

            elif cls in [2, 3, 4]:  # bullets
                bullet_type = self.CLASSES[cls]
                hitbox = self.BULLET_HITBOX[bullet_type]
                detections['bullets'].append({
                    'cx': center_x, 'cy': center_y,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'type': bullet_type, 'hitbox': hitbox
                })

            elif cls >= 5:  # 界面状态
                state_name = self.CLASSES.get(cls, 'unknown')
                if state_name not in state_detections or conf > state_detections[state_name]:
                    state_detections[state_name] = conf

            detections['all_detections'].append({
                'class': self.CLASSES.get(cls, f'cls_{cls}'),
                'class_id': int(cls),
                'cx': center_x, 'cy': center_y,
                'conf': float(conf)
            })

        if state_detections:
            best_state = max(state_detections.keys(), key=lambda k: state_detections[k])
            detections['state'] = best_state
            detections['state_conf'] = state_detections[best_state]
        elif detections['player'] is not None or len(detections['bullets']) > 0:
            detections['state'] = 'playing'
            detections['state_conf'] = 0.5

        return detections

    def compute_real_distance(self, player_cx: float, player_cy: float,
                              bullet_cx: float, bullet_cy: float,
                              bullet_hitbox: float) -> float:
        """计算真实距离（考虑判定半径）"""
        dist_center = np.sqrt((player_cx - bullet_cx)**2 + (player_cy - bullet_cy)**2)
        return dist_center - self.HITBOX['player'] - bullet_hitbox

    def compute_apf_repulsion(self, detections: Dict, predicted_bullets: List = None) -> Tuple[np.ndarray, float]:
        """计算人工势场(APF)总斥力矢量

        公式: F = k * (1/d - 1/d0) / d²
        """
        if detections['player'] is None:
            return np.array([0.0, 0.0]), 0.0

        player_cx, player_cy = detections['player']
        fx_total, fy_total = 0.0, 0.0

        # 合并当前帧小弹和预判弹幕
        all_obstacles = []
        for bullet in detections['bullets']:
            if bullet['type'] == 'bullet_small':
                all_obstacles.append({'cx': bullet['cx'], 'cy': bullet['cy']})

        if predicted_bullets:
            for pred in predicted_bullets:
                all_obstacles.append({'cx': pred['cx'], 'cy': pred['cy']})

        d0 = self.APF_RADIUS
        k = self.APF_GAIN

        for obs in all_obstacles:
            dx = player_cx - obs['cx']
            dy = player_cy - obs['cy']
            d = np.sqrt(dx*dx + dy*dy)

            if d <= 0 or d >= d0:
                continue

            ux, uy = dx / d, dy / d
            repulsion = k * (1.0/d - 1.0/d0) / (d * d)
            fx_total += ux * repulsion
            fy_total += uy * repulsion

        force_magnitude = np.sqrt(fx_total**2 + fy_total**2)
        return np.array([fx_total, fy_total], dtype=np.float32), float(force_magnitude)

    def compute_voronoi_empty_point(self, detections: Dict, predicted_bullets_p2: List[Dict] = None,
                                     grid_size: int = 40) -> Tuple[np.ndarray, float]:
        """计算游戏下半区域的最空旷点"""
        GAME_X_MIN, GAME_X_MAX = 74, 934
        GAME_Y_MIN, GAME_Y_MAX = 70, 1040
        GAME_HEIGHT = GAME_Y_MAX - GAME_Y_MIN

        margin = 40
        y_mid = GAME_Y_MIN + GAME_HEIGHT * 0.5
        x_min, x_max = GAME_X_MIN + margin, GAME_X_MAX - margin
        y_min, y_max = y_mid, GAME_Y_MAX - margin

        if detections['player'] is None:
            return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2]), 100.0

        player_cx, player_cy = detections['player']
        current_bullets = [b for b in detections['bullets'] if b['type'] == 'bullet_small']
        all_bullets = current_bullets + (predicted_bullets_p2 or [])

        if not all_bullets:
            return np.array([player_cx, player_cy]), 100.0

        min_real_dist = min(
            np.sqrt((player_cx - b['cx'])**2 + (player_cy - b['cy'])**2)
            for b in all_bullets
        )

        best_point = np.array([player_cx, player_cy])
        best_safety = min_real_dist

        # 极坐标搜索
        for angle in np.linspace(0, 2 * np.pi, num=8, endpoint=False):
            for r in [30.0, 60.0, 100.0, 150.0]:
                test_x = player_cx + r * np.cos(angle)
                test_y = player_cy + r * np.sin(angle)
                if not (x_min <= test_x <= x_max and y_min <= test_y <= y_max):
                    continue
                min_d = min(np.sqrt((test_x - b['cx'])**2 + (test_y - b['cy'])**2) for b in all_bullets)
                if min_d > best_safety:
                    best_safety = min_d
                    best_point = np.array([test_x, test_y])

        if best_safety < 30.0:
            return np.array([player_cx, player_cy]), min_real_dist

        return best_point, best_safety

    def compute_empty_point_distance_reward(self, detections: Dict,
                                             player_pos: Tuple[float, float],
                                             empty_point: np.ndarray) -> float:
        """计算到空旷点的距离引导奖励

        公式: V = 0.00025 * sqrt(810000 - L²)
        """
        if detections['player'] is None:
            return 0.0

        player_x, player_y = player_pos
        empty_x, empty_y = empty_point
        L = np.sqrt((player_x - empty_x)**2 + (player_y - empty_y)**2)

        if L > 900:
            L = 900
        return 0.00025 * np.sqrt(810000 - L * L)

    def to_observation(self, detections: Dict, player_pos: Tuple[float, float] = None) -> np.ndarray:
        """将检测结果转换为观测向量"""
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        if detections['player'] is not None:
            player_cx, player_cy = detections['player']
        elif player_pos is not None:
            player_cx, player_cy = player_pos
        else:
            player_cx, player_cy = self.img_width / 2, self.img_height * 0.8

        obs[self.OBS_PLAYER_START] = player_cx / self.img_width
        obs[self.OBS_PLAYER_START + 1] = player_cy / self.img_height

        bullet_with_dist = []
        for bullet in detections['bullets']:
            real_dist = self.compute_real_distance(
                player_cx, player_cy, bullet['cx'], bullet['cy'], bullet['hitbox']
            )
            bullet_with_dist.append({**bullet, 'real_dist': real_dist})

        bullet_with_dist.sort(key=lambda b: b['real_dist'])
        bullet_start = self.get_bullet_start_index()

        for i in range(self.max_bullets):
            idx = bullet_start + i * self.OBS_BULLET_STRIDE
            if i < len(bullet_with_dist):
                bullet = bullet_with_dist[i]
                obs[idx] = bullet['cx'] / self.img_width
                obs[idx + 1] = bullet['cy'] / self.img_height
                type_map = {'bullet_small': 0, 'bullet_mid': 1, 'bullet_heavy': 2}
                obs[idx + 2] = type_map.get(bullet['type'], 0) / 2.0
                obs[idx + 3] = np.tanh(bullet['real_dist'] / 100.0)
            else:
                obs[idx] = -1.0
                obs[idx + 1] = -1.0
                obs[idx + 2] = 0.0
                obs[idx + 3] = 1.0

        return obs

    def compute_reward(self, detections: Dict, step_count: int = 0, predicted_bullets: List[Dict] = None) -> float:
        """计算奖励

        奖励项：
        1. 生存奖励：+0.1/帧
        2. 势场惩罚：-Fmag/帧
        3. 敌怪距离惩罚：-500000/de²/帧
        4. 死亡惩罚：-20
        """
        reward = 0.1

        if detections['state'] == 'dead':
            return reward - 20.0

        if detections['player'] is None:
            return reward

        player_cx, player_cy = detections['player']
        _, force_magnitude = self.compute_apf_repulsion(detections, predicted_bullets)
        reward -= force_magnitude

        if detections.get('enemies'):
            min_enemy_dist = float('inf')
            for enemy in detections['enemies']:
                dist = np.sqrt((player_cx - enemy['cx'])**2 + (player_cy - enemy['cy'])**2)
                if dist < min_enemy_dist:
                    min_enemy_dist = dist
            if min_enemy_dist < float('inf'):
                reward += -500000.0 / (min_enemy_dist ** 2)

        return reward

    def visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """可视化检测结果"""
        img = image.copy()
        colors = {
            'player': (0, 255, 0),
            'bullet_small': (255, 255, 0),
            'bullet_mid': (255, 0, 255),
            'bullet_heavy': (0, 255, 255),
        }

        if detections['player'] is not None:
            px, py = int(detections['player'][0]), int(detections['player'][1])
            cv2.circle(img, (px, py), int(self.HITBOX['player'] * 3), colors['player'], 2)
            cv2.putText(img, 'Player', (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['player'], 1)

        for bullet in detections['bullets']:
            bx, by = int(bullet['cx']), int(bullet['cy'])
            hitbox = int(bullet['hitbox'])
            color = colors.get(bullet['type'], (255, 255, 255))
            cv2.circle(img, (bx, by), hitbox, color, 1)
            cv2.circle(img, (bx, by), 2, color, -1)

        return img

    def compute_iou(self, box1: Dict, box2: Dict) -> float:
        """计算两个检测框的IOU"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union_area = area1 + area2 - inter_area

        if union_area < 1e-6:
            return 0.0
        return inter_area / union_area

    def update_bullet_tracks(self, detections: Dict) -> Tuple[List[Dict], ...]:
        """更新弹幕跟踪tracks，返回预判位置P1-P5

        通过IOU匹配跟踪弹幕，连续3帧稳定匹配后进行预判
        """
        if not hasattr(self, '_bullet_tracks'):
            self._bullet_tracks = []
        if not hasattr(self, '_last_bullet_boxes'):
            self._last_bullet_boxes = []

        player_cx, player_cy = 0, 0
        if detections['player'] is not None:
            player_cx, player_cy = detections['player']

        current_boxes = []
        for b in detections['bullets']:
            if b['type'] != 'bullet_small':
                continue
            w, h = b['x2'] - b['x1'], b['y2'] - b['y1']
            if w < 2 or w > 40 or h < 2 or h > 40:
                continue
            dist = np.sqrt((b['cx'] - player_cx)**2 + (b['cy'] - player_cy)**2)
            if dist > self.APF_RADIUS * 1.5:
                continue
            current_boxes.append(b)

        unmatched_current = list(range(len(current_boxes)))

        for track in self._bullet_tracks:
            best_iou, best_idx = 0.0, -1
            for i in unmatched_current:
                iou = self.compute_iou(current_boxes[i], track['last_box'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= 0.3 and best_idx >= 0:
                track['cx_history'].append((current_boxes[best_idx]['cx'], current_boxes[best_idx]['cy']))
                if len(track['cx_history']) > 3:
                    track['cx_history'].pop(0)
                track['last_box'] = current_boxes[best_idx]
                track['matched_frames'] += 1
                unmatched_current.remove(best_idx)
            else:
                track['matched_frames'] = 0

        for i in unmatched_current:
            self._bullet_tracks.append({
                'cx_history': [(current_boxes[i]['cx'], current_boxes[i]['cy'])],
                'last_box': current_boxes[i],
                'matched_frames': 1
            })

        if len(self._bullet_tracks) > 100:
            self._bullet_tracks.sort(key=lambda t: -t['matched_frames'])
            self._bullet_tracks = self._bullet_tracks[:100]

        self._last_bullet_boxes = current_boxes

        # 生成预判位置P1-P5
        predicted = [[] for _ in range(5)]

        for track in self._bullet_tracks:
            if track['matched_frames'] < 3:
                continue
            hist = track['cx_history']
            dx = hist[-1][0] - hist[-2][0]
            dy = hist[-1][1] - hist[-2][1]
            cx, cy = hist[-1][0], hist[-1][1]

            fx, fy = cx, cy
            for i in range(5):
                fx += dx
                fy += dy
                predicted[i].append({'cx': fx, 'cy': fy, 'is_predicted': True})

        return tuple(predicted)


if __name__ == "__main__":
    model_path = Path(__file__).parent.parent / "models" / "best.onnx"
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        exit(1)

    processor = YOLOProcessor(str(model_path), max_bullets=50, conf_threshold=0.15)
    print(f"观测空间维度: {processor.obs_dim}")
    print(f"APF参数: d0={processor.APF_RADIUS}, k={processor.APF_GAIN}")