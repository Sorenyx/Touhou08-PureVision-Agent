"""
PPO训练脚本 V2 - 训练AI玩东方永夜抄
基于APF势场避障和生存奖励的强化学习训练
"""

import os
import time
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

# 导入环境
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from PPO import TouhouEnv, make_touhou_env


class TrainingCallback(BaseCallback):
    """训练回调函数，用于记录训练过程"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_length = 0

    def _on_step(self) -> bool:
        """每步调用"""
        # 累加当前episode的奖励和步数
        if 'rewards' in self.locals:
            self.current_ep_reward += sum(self.locals['rewards'])
        if 'dones' in self.locals:
            self.current_ep_length += len([d for d in self.locals['dones'] if d])

        # episode结束时的统计
        if self.locals.get('done') or self.locals.get('truncated'):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            # 每10个episode打印一次统计
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                max_reward = max(self.episode_rewards[-10:])
                print(f"[V2] Episodes {len(self.episode_rewards)-10}-{len(self.episode_rewards)}: "
                      f"Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}, Max Reward = {max_reward:.2f}")

            self.current_ep_reward = 0
            self.current_ep_length = 0

        return True

    def _on_rollout_end(self) -> None:
        """每个rollout结束时调用"""
        return True


def train_ppo(total_timesteps: int = 100000,
              model_path: str = "models/touhou_ppo_v2",
              resume: bool = False,
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 64,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_range: float = 0.2,
              ent_coef: float = 0.01,
              vf_coef: float = 0.5,
              max_grad_norm: float = 0.5,
              verbose: int = 1):
    """训练PPO模型 V2（连续动作版本）

    动作空间：2维连续动作 [dx, dy]，范围 [-1, 1]
    - dx: X方向移动，正值向右，负值向左
    - dy: Y方向移动，正值向下，负值向上
    - 基础移动速度：3.0px/帧
    - 支持斜向平滑移动

    奖励函数说明：
    - 失命惩罚：-10
    - 最终死亡惩罚：-20
    - 存活步数奖励（episode结束时）：<800=0, 800~999=+5, 1000~1199=+10, 1200~1399=+20, >=1400=+100
    - 分段生存奖励：0~500=+0.01/步, 501~1000=+0.05/步, 1001+=+0.10/步
    - 小弹擦弹奖励：5~10px=+0.3, 10~30px=+0.15
    - 小弹极近距离惩罚：<5px=-1.0

    Args:
        total_timesteps: 总训练步数
        model_path: 模型保存路径
        resume: 是否从现有模型继续训练
        learning_rate: 学习率
        n_steps: 每次更新收集的步数
        batch_size: 批次大小
        gamma: 折扣因子
        gae_lambda: GAE参数
        clip_range: PPO裁剪范围
        ent_coef: 熵系数（鼓励探索）
        vf_coef: 价值损失系数
        max_grad_norm: 最大梯度范数
        verbose: 日志详细程度
    """
    # 创建模型保存目录
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 创建环境
    print("=" * 60)
    print("PPO训练 V2 - 东方永夜抄（连续动作版）")
    print("=" * 60)
    print("动作空间配置：")
    print("  - 类型: Box 连续动作")
    print("  - 维度: [dx, dy]")
    print("  - 范围: [-1, 1]")
    print("  - 基础速度: 3.0 px/帧")
    print("  - 支持斜向平滑移动")
    print()
    print("奖励函数配置：")
    print("  - 失命惩罚: -10")
    print("  - 最终死亡惩罚: -20")
    print("  - 存活步数奖励: <800=0, 800~999=+5, 1000~1199=+10, 1200~1399=+20, >=1400=+100")
    print("  - 分段生存奖励: 0~500=+0.01, 501~1000=+0.05, 1001+=+0.10")
    print("  - 小弹擦弹: 5~10px=+0.3, 10~30px=+0.15, <5px=-1.0")
    print("  - YOLO FPS: 15 (每4步推理一次)")
    print("=" * 60)
    print()

    env = make_touhou_env(
        render_mode=None,  # 训练时不渲染
        auto_restart=False
    )

    # 包装环境
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=gamma)

    # 创建或加载模型
    if resume and (model_dir / "best_model.zip").exists():
        print(f"从 {model_dir / 'best_model.zip'} 加载模型...")
        model = PPO.load(str(model_dir / "best_model.zip"), env=env)
        print("模型加载完成")
    else:
        print("创建新模型（从头开始训练）...")
        print(f"模型将保存到: {model_dir}")
        print()

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            tensorboard_log=str(model_dir / "tensorboard"),
            seed=42  # 固定种子以保证可复现性
        )
        print("模型创建完成")
        print(f"超参数: lr={learning_rate}, n_steps={n_steps}, batch_size={batch_size}, gamma={gamma}")
        print()

    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="touhou_ppo_v2"
    )

    training_callback = TrainingCallback(verbose=1)

    # 开始训练
    print(f"开始训练，总步数: {total_timesteps:,}")
    print("=" * 60)
    print("按 Ctrl+C 可随时中断训练，模型会自动保存")
    print("-" * 60)

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, training_callback],
            progress_bar=True
        )

    except KeyboardInterrupt:
        print("\n\n训练被用户中断，正在保存模型...")

    finally:
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print()
        print("=" * 60)
        print("训练完成！")
        print(f"耗时: {hours}h {minutes}m {seconds}s")
        print(f"训练步数: {training_callback.num_timesteps:,}")
        print(f"完成回合数: {len(training_callback.episode_rewards)}")

        if training_callback.episode_rewards:
            print(f"平均回合奖励: {np.mean(training_callback.episode_rewards):.2f}")
            print(f"最大回合奖励: {max(training_callback.episode_rewards):.2f}")
            print(f"平均回合长度: {np.mean(training_callback.episode_lengths):.1f}")

        # 保存模型
        print()
        print("保存模型...")
        model.save(str(model_dir / "final_model"))
        model.save(str(model_dir / "best_model"))
        print(f"模型已保存到: {model_dir}")
        print("=" * 60)

    # 关闭环境
    env.close()


def test_model(model_path: str = "models/touhou_ppo_v2/best_model.zip",
               num_episodes: int = 5,
               render: bool = True):
    """测试训练好的模型"""
    print("=" * 60)
    print("测试模式 - 东方永夜抄 V2")
    print("=" * 60)

    env = make_touhou_env(
        render_mode='human' if render else None,
        auto_restart=False
    )

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    print(f"运行 {num_episodes} 个测试回合...")
    print("-" * 60)

    episode_stats = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        deaths = 0
        done = False
        truncated = False

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if render:
                time.sleep(0.016)

        deaths = 3 - info.get('lives', 3)
        episode_stats.append({
            'reward': episode_reward,
            'length': episode_length,
            'deaths': deaths,
            'survival_bonus': info.get('survival_bonus', 0)
        })

        print(f"回合 {episode + 1}: "
              f"奖励={episode_reward:.1f}, "
              f"长度={episode_length}, "
              f"死亡={deaths}, "
              f"存活奖励={info.get('survival_bonus', 0):.0f}")

    print("-" * 60)
    print(f"平均奖励: {np.mean([s['reward'] for s in episode_stats]):.2f}")
    print(f"平均长度: {np.mean([s['length'] for s in episode_stats]):.1f}")
    print(f"平均死亡: {np.mean([s['deaths'] for s in episode_stats]):.1f}")
    print("=" * 60)

    env.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="东方永夜抄PPO训练 V2")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "test"], help="运行模式")
    parser.add_argument("--model-path", type=str, default="models/touhou_ppo_v2",
                       help="模型保存/加载路径")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="训练步数")
    parser.add_argument("--resume", action="store_true",
                       help="从现有模型继续训练")
    parser.add_argument("--episodes", type=int, default=5,
                       help="测试回合数")
    parser.add_argument("--no-render", action="store_true",
                       help="测试时不渲染")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="学习率")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="批次大小")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="折扣因子")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="每次更新收集的步数")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="熵系数")

    args = parser.parse_args()

    if args.mode == "train":
        train_ppo(
            total_timesteps=args.timesteps,
            model_path=args.model_path,
            resume=args.resume,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef
        )

    elif args.mode == "test":
        model_path = args.model_path
        if not model_path.endswith(".zip"):
            model_path = str(Path(model_path) / "best_model.zip")

        test_model(
            model_path=model_path,
            num_episodes=args.episodes,
            render=not args.no_render
        )


if __name__ == "__main__":
    main()
