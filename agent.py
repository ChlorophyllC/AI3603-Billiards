"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class MultiHeadPoolPolicyNetwork(nn.Module):
    """和 train_student.py 对齐的多头策略网络。

    输出 [B, K, 6]：
      [V0_norm, sin(phi), cos(phi), theta_norm, a01, b01]
    """

    def __init__(self, input_dim, num_heads=5, action_dim=6):
        super().__init__()
        self.num_heads = int(num_heads)
        self.action_dim = int(action_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_dim),
            )
            for _ in range(self.num_heads)
        ])

    def forward(self, state):
        feat = self.encoder(state)
        outs = []
        for head in self.heads:
            raw = head(feat)
            v0 = torch.sigmoid(raw[:, 0:1])
            sincos = torch.tanh(raw[:, 1:3])
            norm = torch.sqrt((sincos ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
            sincos = sincos / norm
            theta = torch.sigmoid(raw[:, 3:4])
            ab = torch.sigmoid(raw[:, 4:6])
            outs.append(torch.cat([v0, sincos, theta, ab], dim=1))

        return torch.stack(outs, dim=1)


class NewAgent(Agent):
    """
    Neural network-based agent with safety fallback mechanisms.
    """
    
    def __init__(self, model_path='best_student_imitation.pth'):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        self.TABLE_WIDTH = 1.9812
        self.TABLE_HEIGHT = 0.9906

        self.cushions = {
            'x_pos': self.TABLE_WIDTH / 2, 
            'x_neg': -self.TABLE_WIDTH / 2, 
            'y_pos': self.TABLE_HEIGHT / 2, 
            'y_neg': -self.TABLE_HEIGHT / 2
        }
        
        # 初始化神经网络
        self.input_dim = 2 + 7*3 + 7*3 + 3 + 3 + 2  # 47维
        self.NUM_CANDIDATES = 5
        self.model = MultiHeadPoolPolicyNetwork(self.input_dim, num_heads=self.NUM_CANDIDATES, action_dim=6)
        
        # 加载训练好的权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            print(f"[NeuralAgent] Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"[NeuralAgent] Warning: Failed to load model: {e}")
            self.model = None
        
        # Safety thresholds
        self.SAFE_SCORE_THRESHOLD = 20.0  # 神经网络预测的最低可接受分数
        self.FATAL_RATE_THRESHOLD = 0.15  # 最大可接受的致命失误率
        self.SCORE_THRESHOLD_NORMAL = 45.0
        self.SCORE_THRESHOLD_EIGHT = 60.0
        self.NET_EVAL_TRIALS = 3
        self.LOCAL_PERTURB_STEPS = 6
        self.LOCAL_PERTURB_SCALE = {
            'V0': 0.8,
            'phi': 5.0,
            'theta': 2.0,
            'a': 0.03,
            'b': 0.03,
        }
        self.USE_LOCAL = True
        self.CMA_MAXITER = 3
        self.CMA_POPSIZE = 3
        self.CMA_SIGMA = 0.2
        
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }

        print("[NeuralAgent] Neural network agent with safety fallback initialized.")
    
    # ==================== Utility Functions ====================
    
    def _safe_action(self):
        return {'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0}
    
    def _calc_dist(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    
    def _unit_vector(self, vec):
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        return np.array([1.0, 0.0]) if norm < 1e-6 else vec / norm
    
    def _direction_to_degrees(self, direction_vec):
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360
    
    # ==================== Feature Extraction ====================
    
    def _ball_features_from_pooltool(self, ball):
        """从 pooltool ball 对象提取特征 [x, y, exist] - 和数据收集完全一致"""
        if ball.state.s == 4:  # 球已入袋
            return [0.0, 0.0, 0.0]
        else:
            pos = ball.state.rvw[0]
            return [
                float(pos[0] / (self.TABLE_WIDTH / 2)),
                float(pos[1] / (self.TABLE_HEIGHT / 2)),
                1.0
            ]
    
    def _extract_state_features(self, balls, my_targets):
        """
        提取状态特征 - 和数据收集完全一致
        
        关键：数据收集时是按 1-15 顺序遍历，只选 my_targets 中的球
        """
        features = []
        
        # 1. Cue ball position (2)
        cue_pos = balls['cue'].state.rvw[0]
        features.extend([
            float(cue_pos[0] / (self.TABLE_WIDTH / 2)),
            float(cue_pos[1] / (self.TABLE_HEIGHT / 2))
        ])
        
        # 2. My balls (7 × 3 = 21)
        # 数据收集时：for i in range(1, 16): if ball_id in my_targets and ball_id != '8'
        # 所以要按 1-15 顺序遍历，而不是对 my_targets 排序！
        my_balls_list = []
        for i in range(1, 16):
            ball_id = str(i)
            if ball_id in my_targets and ball_id != '8':
                if ball_id in balls and balls[ball_id].state.s != 4:
                    my_balls_list.append(self._ball_features_from_pooltool(balls[ball_id]))
                # 注意：数据收集时如果球入袋了就不加入列表！
        
        # Padding到7个
        for i in range(7):
            if i < len(my_balls_list):
                features.extend(my_balls_list[i])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # 3. Opponent balls (7 × 3 = 21)
        # 同样按 1-15 顺序遍历
        opponent_balls_list = []
        for i in range(1, 16):
            ball_id = str(i)
            if ball_id not in my_targets and ball_id != '8':
                if ball_id in balls and balls[ball_id].state.s != 4:
                    opponent_balls_list.append(self._ball_features_from_pooltool(balls[ball_id]))
        
        for i in range(7):
            if i < len(opponent_balls_list):
                features.extend(opponent_balls_list[i])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # 4. Eight ball (3)
        if '8' in balls and balls['8'].state.s != 4:
            features.extend(self._ball_features_from_pooltool(balls['8']))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 5. Phase flags (3)
        is_eight_ball_game = float(my_targets == ['8'])
        
        my_balls_remaining = len(my_balls_list)
        opponent_balls_remaining = len(opponent_balls_list)
        
        # 数据收集时的判断逻辑
        is_opening = float(
            my_balls_remaining >= 6 and opponent_balls_remaining >= 6
        )
        is_endgame = float(
            my_balls_remaining <= 2 or opponent_balls_remaining <= 2
        )
        
        features.extend([is_eight_ball_game, is_opening, is_endgame])
        
        # 6. Remaining counts (2)
        features.extend([
            my_balls_remaining / 7.0,
            opponent_balls_remaining / 7.0
        ])
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def _denormalize_action(self, normalized_action):
        """将网络输出转换回环境动作。

        normalized_action 维度为 6：
          [V0_norm, sin(phi), cos(phi), theta_norm, a01, b01]
          
        关键：必须与train_student.py中的_extract_action完全对应！
        训练时：V0_norm = (V0-0.5)/7.5, theta_norm = theta/90, a01 = a+0.5
        推理时：V0 = V0_norm*7.5+0.5, theta = theta_norm*90, a = a01-0.5
        """
        # V0: [0,1] -> [0.5, 8.0]
        V0 = float(normalized_action[0]) * 7.5 + 0.5
        
        # phi: 从sin/cos恢复角度
        sin_phi = float(normalized_action[1])
        cos_phi = float(normalized_action[2])
        phi = float(np.arctan2(sin_phi, cos_phi) * 180.0 / np.pi)
        phi = phi % 360.0  # 确保在[0, 360)
        
        # theta: [0,1] -> [0, 90]
        theta = float(normalized_action[3]) * 90.0
        
        # a, b: [0,1] -> [-0.5, 0.5]
        a = float(normalized_action[4]) - 0.5
        b = float(normalized_action[5]) - 0.5

        return {
            'V0': float(np.clip(V0, 0.5, 8.0)),
            'phi': float(phi),
            'theta': float(np.clip(theta, 0, 90)),
            'a': float(np.clip(a, -0.5, 0.5)),
            'b': float(np.clip(b, -0.5, 0.5))
        }
    
    # ==================== Safety Check ====================
    
    def _improved_reward_function(self, shot, last_state, player_targets, table):
        """评估击球质量"""
        score = analyze_shot_for_reward(shot, last_state, player_targets)
        
        is_targeting_eight_legally = (player_targets == ['8'])
        
        # 惩罚非法移动黑8
        if not is_targeting_eight_legally and '8' in shot.balls and shot.balls['8'].state.s != 4:
            eight_before_dist = self._distance_to_nearest_pocket(
                last_state['8'].state.rvw[0], table
            )
            eight_after_dist = self._distance_to_nearest_pocket(
                shot.balls['8'].state.rvw[0], table
            )
            if eight_after_dist < eight_before_dist:
                score -= (eight_before_dist - eight_after_dist) * 150
        
        # 白球安全性评估
        cue_pocketed = "cue" in [bid for bid, b in shot.balls.items() if b.state.s == 4]
        if not cue_pocketed and 'cue' in shot.balls:
            cue_pos = shot.balls['cue'].state.rvw[0]
            cue_dist = self._distance_to_nearest_pocket(cue_pos, table)
            if cue_dist < 0.1:
                score -= 30 * (0.1 - cue_dist) / 0.1
            elif cue_dist > 0.2:
                score += min(15, cue_dist * 20)
        
        return score
    
    def _distance_to_nearest_pocket(self, ball_pos, table):
        """计算球到最近袋口的距离"""
        min_dist = float('inf')
        for pocket in table.pockets.values():
            dist = self._calc_dist(ball_pos, pocket.center)
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _evaluate_action_safety(self, action, balls, my_targets, table, num_trials=10):
        """评估动作的安全性和预期得分"""
        scores = []
        fatal_count = 0
        is_targeting_eight_legally = (my_targets == ['8'])
        
        for _ in range(num_trials):
            try:
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(table=sim_table, balls=sim_balls, cue=pt.Cue(cue_ball_id="cue"))

                noise = self.noise_std
                V0 = action['V0'] + np.random.normal(0, noise['V0'])
                phi = (action['phi'] + np.random.normal(0, noise['phi'])) % 360
                theta = action['theta'] + np.random.normal(0, noise['theta'])
                a = action['a'] + np.random.normal(0, noise['a'])
                b = action['b'] + np.random.normal(0, noise['b'])
                
                shot.cue.set_state(
                    V0=np.clip(V0, 0.5, 8.0),
                    phi=phi,
                    theta=np.clip(theta, 0, 90),
                    a=np.clip(a, -0.5, 0.5),
                    b=np.clip(b, -0.5, 0.5)
                )
                
                if not simulate_with_timeout(shot, timeout=3):
                    scores.append(-100)
                    continue
                
                score = self._improved_reward_function(shot, balls, my_targets, sim_table)
                scores.append(score)
                
                # 检查致命失误
                new_pocketed = [bid for bid, ball in shot.balls.items() 
                               if ball.state.s == 4 and balls[bid].state.s != 4]
                if "cue" in new_pocketed and "8" in new_pocketed:
                    fatal_count += 1
                elif "8" in new_pocketed and not is_targeting_eight_legally:
                    fatal_count += 1
                    
            except Exception:
                scores.append(-100)
        
        avg_score = float(np.mean(scores)) if scores else -999
        fatal_rate = fatal_count / num_trials
        
        return avg_score, fatal_rate
    
    # ==================== Defensive Strategy (Fallback) ====================
    
    def _get_opponent_targets(self, my_targets):
        """确定对手的目标球"""
        all_targets = set(str(i) for i in range(1, 16))
        my_set = set(my_targets)
        opponent_set = all_targets - my_set - {'8'}
        return list(opponent_set)
    
    def _calc_ghost_ball(self, target_pos, pocket_pos):
        direction = self._unit_vector(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        return np.array(target_pos[:2]) - direction * (2 * self.BALL_RADIUS)
    
    def _geo_shot(self, cue_pos, target_pos, pocket_pos):
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        direction = self._unit_vector(ghost_pos - np.array(cue_pos[:2]))
        phi = self._direction_to_degrees(direction)
        dist = self._calc_dist(cue_pos, ghost_pos)
        V0 = min(2.0 + dist * 1.5 if dist < 0.8 else 4.0 + dist * 0.8, 7.5)
        return {'V0': float(V0), 'phi': float(phi), 'theta': 0.0, 'a': 0.0, 'b': 0.0}

    def _geo_bank_shot(self, cue_pos, target_pos, pocket_pos, cushion_id):
        """通过镜像袋口生成简单的反弹击球。"""
        mirrored = np.array(pocket_pos[:2])
        if 'x' in cushion_id:
            mirrored[0] = 2 * self.cushions[cushion_id] - mirrored[0]
        else:
            mirrored[1] = 2 * self.cushions[cushion_id] - mirrored[1]
        return self._geo_shot(cue_pos, target_pos, mirrored)
    
    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        count = 0
        line_vec = np.array(to_pos[:2]) - np.array(from_pos[:2])
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-6:
            return 0
        line_dir = line_vec / line_length
        
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4:
                continue
            vec_to_ball = ball.state.rvw[0][:2] - np.array(from_pos[:2])
            proj_length = np.dot(vec_to_ball, line_dir)
            if 0 < proj_length < line_length:
                perp_dist = np.linalg.norm(
                    ball.state.rvw[0][:2] - (np.array(from_pos[:2]) + line_dir * proj_length)
                )
                if perp_dist < self.BALL_RADIUS * 2.2:
                    count += 1
        return count
    
    def _choose_top_targets(self, balls, my_targets, table, num_choices=1, is_defense=True):
        """为防守策略选择目标（简化版）"""
        all_choices = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:
                continue
            target_pos = balls[target_id].state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                if (self._count_obstructions(balls, cue_pos, target_pos, ['cue', target_id]) == 0 and
                    self._count_obstructions(balls, target_pos, pocket_pos, ['cue', target_id]) == 0):
                    dist = self._calc_dist(cue_pos, target_pos) + self._calc_dist(target_pos, pocket_pos)
                    score = 100 - dist * 20
                    all_choices.append({
                        'type': 'direct',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'score': score
                    })
        
        all_choices.sort(key=lambda x: x['score'], reverse=True)
        return all_choices[:num_choices]

    def _cma_refine_action(self, action, balls, my_targets, table):
        """使用小规模CMA-ES在局部搜索，步数和种群均受限以降低耗时。"""
        try:
            # 定义以当前动作为中心的边界
            bounds = np.array([
                [max(0.5, action['V0'] - 1.0), min(8.0, action['V0'] + 1.0)],
                [(action['phi'] - 8) % 360, (action['phi'] + 8) % 360],
                [max(0.0, action['theta'] - 3), min(90.0, action['theta'] + 3)],
                [max(-0.5, action['a'] - 0.06), min(0.5, action['a'] + 0.06)],
                [max(-0.5, action['b'] - 0.06), min(0.5, action['b'] + 0.06)],
            ])

            def normalize(x):
                return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

            def denormalize(x):
                return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

            # 先clip到bounds范围再normalize，避免初始点超界
            raw_x = np.array([
                np.clip(action['V0'], bounds[0,0], bounds[0,1]),
                np.clip(action['phi'], bounds[1,0], bounds[1,1]),
                np.clip(action['theta'], bounds[2,0], bounds[2,1]),
                np.clip(action['a'], bounds[3,0], bounds[3,1]),
                np.clip(action['b'], bounds[4,0], bounds[4,1]),
            ])
            x0 = normalize(raw_x)
            x0 = np.clip(x0, 0, 1)

            opts = {
                'bounds': [[0]*5, [1]*5],
                'maxiter': self.CMA_MAXITER,
                'popsize': self.CMA_POPSIZE,
                'verb_disp': 0,
                'verb_log': 0,
            }

            def objective(x_norm):
                x = denormalize(np.clip(x_norm, 0, 1))
                cand = {
                    'V0': float(x[0]),
                    'phi': float(x[1] % 360),
                    'theta': float(x[2]),
                    'a': float(x[3]),
                    'b': float(x[4]),
                }
                score, fatal = self._evaluate_action_safety(
                    cand, balls, my_targets, table, num_trials=self.NET_EVAL_TRIALS
                )
                # 将致命率直接转化为惩罚
                penalty = 300 * fatal
                return -(score - penalty)

            es = cma.CMAEvolutionStrategy(x0, self.CMA_SIGMA, opts)
            es.optimize(objective)

            best_x = denormalize(np.clip(es.result.xbest, 0, 1))
            best = {
                'V0': float(best_x[0]),
                'phi': float(best_x[1] % 360),
                'theta': float(best_x[2]),
                'a': float(best_x[3]),
                'b': float(best_x[4]),
            }
            best_score, best_fatal = self._evaluate_action_safety(
                best, balls, my_targets, table, num_trials=self.NET_EVAL_TRIALS
            )
            return best, best_score, best_fatal
        except Exception:
            return action, -1e9, 1.0
    
    def _local_optimize_action(self, action, balls, my_targets, table):
            """在网络输出附近做极小范围扰动微调，减少额外决策时间。"""
            candidates = [action]

            # 生成对称扰动
            for _ in range(self.LOCAL_PERTURB_STEPS):
                perturbed = dict(action)
                for k, scale in self.LOCAL_PERTURB_SCALE.items():
                    noise = np.random.randn() * scale
                    if k == 'phi':
                        perturbed[k] = (perturbed[k] + noise) % 360
                    else:
                        perturbed[k] = perturbed[k] + noise

                # 边界裁剪
                perturbed['V0'] = float(np.clip(perturbed['V0'], 0.5, 8.0))
                perturbed['theta'] = float(np.clip(perturbed['theta'], 0, 90))
                perturbed['a'] = float(np.clip(perturbed['a'], -0.5, 0.5))
                perturbed['b'] = float(np.clip(perturbed['b'], -0.5, 0.5))
                candidates.append(perturbed)

            best = action
            best_score, best_fatal = self._evaluate_action_safety(
                best, balls, my_targets, table, num_trials=1
            )

            for cand in candidates[1:]:
                score, fatal = self._evaluate_action_safety(
                    cand, balls, my_targets, table, num_trials=1
                )
                if fatal <= self.FATAL_RATE_THRESHOLD and (fatal < best_fatal or score > best_score):
                    best, best_score, best_fatal = cand, score, fatal

            return best, best_score, best_fatal
    
    def _find_best_safety_shot(self, balls, my_targets, table):
        """生成防守性击球"""
        print("[NeuralAgent] Attempting defensive safety shot...")
        
        cue_pos = balls['cue'].state.rvw[0]
        candidate_safeties = []
        
        # Strategy 1: 轻推白球到安全位置
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = angle * np.pi / 180
            direction = np.array([np.cos(rad), np.sin(rad)])
            phi = self._direction_to_degrees(direction)
            action = {'V0': 0.8, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
            candidate_safeties.append(action)
        
        # Strategy 2: 推自己的球到边缘
        for my_ball_id in my_targets:
            if my_ball_id == '8' or balls[my_ball_id].state.s == 4:
                continue
            
            my_ball_pos = balls[my_ball_id].state.rvw[0]
            
            # 找最近的cushion
            for cushion_name, cushion_val in self.cushions.items():
                if 'x' in cushion_name:
                    target_pos = [cushion_val, my_ball_pos[1]]
                else:
                    target_pos = [my_ball_pos[0], cushion_val]
                
                action = self._geo_shot(cue_pos, my_ball_pos, target_pos)
                action['V0'] = 1.2  # 低力度
                candidate_safeties.append(action)
        
        if not candidate_safeties:
            return self._safe_action()
        
        # 简单评估：选择第一个不会犯规的
        for action in candidate_safeties:
            avg_score, fatal_rate = self._evaluate_action_safety(
                action, balls, my_targets, table, num_trials=5
            )
            if avg_score > -50 and fatal_rate < 0.3:
                print(f"[NeuralAgent] Using safety shot with score={avg_score:.1f}")
                return action
        
        print("[NeuralAgent] No good safety shot found, using safe action")
        return self._safe_action()
    
    # ==================== Main Decision Logic ====================
    
    def decision(self, balls=None, my_targets=None, table=None):
        if not all([balls, my_targets, table]):
            return self._safe_action()
        
        try:
            # 检查是否需要打黑8
            remaining = [bid for bid in my_targets if bid != '8' and balls[bid].state.s != 4]
            if not remaining and "8" in balls and balls["8"].state.s != 4:
                my_targets = ['8']
                print("[NeuralAgent] Switching to black eight")
            
            # 如果模型没有加载成功，返回安全动作
            if self.model is None:
                print("[NeuralAgent] Model not loaded")
                return self._safe_action()
            
            # 1. 使用神经网络生成多个候选动作，然后用轻量仿真筛选
            is_black_eight = (my_targets == ['8'])
            score_threshold = self.SCORE_THRESHOLD_EIGHT if is_black_eight else self.SCORE_THRESHOLD_NORMAL

            with torch.no_grad():
                state_features = self._extract_state_features(balls, my_targets)
                normalized_outputs = self.model(state_features)  # [1, K, 6]

            candidates = []
            for k in range(normalized_outputs.shape[1]):
                cand_norm = normalized_outputs[0, k].cpu().numpy()
                cand_action = self._denormalize_action(cand_norm)
                # 只评估一次分数和致命率
                score, fatal = self._evaluate_action_safety(
                    cand_action, balls, my_targets, table, num_trials=1
                )
                candidates.append((cand_action, score, fatal, f"net-{k}"))

            # 选分数最高且致命率合格的
            best_action = None
            best_score = -1e9
            best_fatal = 1.0
            for cand_action, score, fatal, name in candidates:
                if fatal <= self.FATAL_RATE_THRESHOLD and score > best_score:
                    best_action = cand_action
                    best_score = score
                    best_fatal = fatal

                    if best_score >= score_threshold:
                     print(f"[NeuralAgent] ✓ Selected action from network: score={best_score:.2f}, fatal_rate={best_fatal:.1%}")
                     return best_action

            # 如果没有任何合格动作，退而求其次，选分数最高的
            if best_action is None and candidates:
                best_action, best_score, best_fatal, _ = max(candidates, key=lambda x: x[1])

            # 3a. 局部扰动微调（替代CMA-ES）
            if self.USE_LOCAL:
                print("[NeuralAgent] Performing local optimization...")
                opt_action, opt_score, opt_fatal = self._local_optimize_action(
                    best_action, balls, my_targets, table
                )
                
                opt_score, opt_fatal = self._evaluate_action_safety(
                    opt_action, balls, my_targets, table, num_trials=5)
                
                if opt_fatal <= self.FATAL_RATE_THRESHOLD and opt_score > 0:
                    print(f"[NeuralAgent] ✓ Local optimization improved action:\n    V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, theta={best_action['theta']:.2f}, a={best_action['a']:.2f}, b={best_action['b']:.2f}\n ->V0={opt_action['V0']:.2f}, phi={opt_action['phi']:.2f}, theta={opt_action['theta']:.2f}, a={opt_action['a']:.2f}, b={opt_action['b']:.2f}\n score={opt_score:.2f}, fatal_rate={opt_fatal:.1%}")
                    return opt_action
            
            print("[NeuralAgent] No acceptable action found, reduced to safe strategy.")
            return self._safe_action()
        except Exception as e:
            print(f"[NeuralAgent] Error: {e}")
            import traceback
            traceback.print_exc()
            return self._safe_action()

