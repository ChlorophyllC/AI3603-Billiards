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

import cma
class NewAgent(Agent):
    """
    Optimized two-layer decision architecture agent for billiards.
    Layer 1: Strategy - select best target ball
    Layer 2: Tactics - geometric calculation or optimization search
    Layer 3: Safety Check
    """
    
    def __init__(self):
        super().__init__()
        self.BALL_RADIUS = 0.028575
        
        # Optimizer configuration

        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("[NewAgent] Optimized agent initialized with improved reward function")
    
    # ==================== Utility Functions ====================
    def _safe_action(self):
        """Return a neutral action (no movement)"""
        return {'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0}

    def _calc_dist(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    
    def _unit_vector(self, vec):
        """Convert vector to unit direction"""
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        return np.array([1.0, 0.0]) if norm < 1e-6 else vec / norm
    
    def _direction_to_degrees(self, direction_vec):
        """Convert direction vector to angle (0-360 degrees)"""
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360
    
    # ==================== Reward Functions ====================
    
    def _improved_reward_function(self, shot, last_state, player_targets, table):
        """
        Enhanced reward function with dense reward signals.
        Evaluates: ball pocketing, fouls, and proximity to pockets.
        """
        # Detect newly pocketed balls
        new_pocketed = [bid for bid, b in shot.balls.items() 
                        if b.state.s == 4 and last_state[bid].state.s != 4]
        
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
        enemy_pocketed = [bid for bid in new_pocketed 
                          if bid not in player_targets and bid not in ["cue", "8"]]
        
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        is_targeting_eight_legally = (len(player_targets) == 1 and player_targets[0] == "8")

        # Analyze first contact ball
        first_contact_ball_id = None
        foul_first_hit = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue']
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        
        # Check for foul: no first contact
        if first_contact_ball_id is None:
            foul_first_hit = True
        else:
            if is_targeting_eight_legally:
                # Check for illegal first contact
                if first_contact_ball_id != '8':
                    foul_first_hit = True
            else:
                remaining_own = [bid for bid in player_targets if last_state[bid].state.s != 4]
                opponent_plus_eight = [bid for bid in last_state.keys() 
                                    if bid not in player_targets and bid != 'cue']
                if '8' not in opponent_plus_eight:
                    opponent_plus_eight.append('8')
                
                if remaining_own and first_contact_ball_id in opponent_plus_eight:
                    foul_first_hit = True
        
        # Analyze cushion contact
        cue_hit_cushion = False
        target_hit_cushion = False
        foul_no_rail = False
        
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                cue_hit_cushion = cue_hit_cushion or ('cue' in ids)
                target_hit_cushion = (target_hit_cushion or 
                                     (first_contact_ball_id and first_contact_ball_id in ids))

        # Check for foul: no rail contact
        if (len(new_pocketed) == 0 and not cue_hit_cushion and not target_hit_cushion):
            foul_no_rail = True
        
        # Calculate base score
        score = 0
        
        # Cue and eight ball penalties
        
        if cue_pocketed and eight_pocketed:
            score = -500
        elif cue_pocketed:
            score -= 30  # Minor penalty, game continues
        elif eight_pocketed:
            score += 200 if is_targeting_eight_legally else -500
        
        # Foul penalties
        score -= 30 if foul_first_hit else 0
        score -= 30 if foul_no_rail else 0
        
        # Pocketing rewards
        score += len(own_pocketed) * 50
        score -= len(enemy_pocketed) * 20
        
        # Default reward for no-event shots
        if (not cue_pocketed and not eight_pocketed and 
            not foul_first_hit and not foul_no_rail and score == 0):
            score = 5 
        
        # ============ Dense Reward Signals ============
        
        # Distance penalties for eight ball (avoid accidental pocketing)
        if (not is_targeting_eight_legally and '8' in shot.balls and 
            shot.balls['8'].state.s != 4):
            eight_before_dist = self._distance_to_nearest_pocket(last_state['8'].state.rvw[0], table)
            eight_after_dist = self._distance_to_nearest_pocket(shot.balls['8'].state.rvw[0], table)
            
            # If eight ball is closer to pocket, give penalty
            if eight_after_dist < eight_before_dist:
                distance_decrease = eight_before_dist - eight_after_dist
                penalty = distance_decrease * 150 
                score -= penalty
        
        # Distance penalties for cue ball (avoid scratching)
        if not cue_pocketed and 'cue' in shot.balls:
            cue_pos = shot.balls['cue'].state.rvw[0]
            cue_dist = self._distance_to_nearest_pocket(cue_pos, table)
            
            if cue_dist < 0.1:
                score -= 30 * (0.1 - cue_dist) / 0.1
            elif cue_dist > 0.2:
                score += min(15, cue_dist * 20)
        
        # Combined risk: eight ball and cue ball near same pocket
        if (not is_targeting_eight_legally and not cue_pocketed and '8' in shot.balls and 
            shot.balls['8'].state.s != 4):
            cue_pos = shot.balls['cue'].state.rvw[0]
            eight_pos = shot.balls['8'].state.rvw[0]
            
            for pocket in table.pockets.values():
                pocket_pos = pocket.center
                cue_to_pocket = self._calc_dist(cue_pos, pocket_pos)
                eight_to_pocket = self._calc_dist(eight_pos, pocket_pos)
                
                if cue_to_pocket < 0.2 and eight_to_pocket < 0.2:
                    score -= 50
                    break
        
        return score
    
    def _distance_to_nearest_pocket(self, ball_pos, table):
        """Calculate distance from ball to nearest pocket"""
        min_dist = float('inf')
        for pocket in table.pockets.values():
            pocket_pos = pocket.center
            dist = np.linalg.norm(np.array(ball_pos[:2]) - np.array(pocket_pos[:2]))
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _check_fatal_failure(self, action, balls, my_targets, table, num_trials=10, fatal_threshold=0.1):
        """
        Check if an action leads to fatal failures (game-losing situations).
        
        Parameters:
            action: dict with keys ['V0', 'phi', 'theta', 'a', 'b']
            
        Returns:
            fatal_rate: Probability of fatal failure (0.0 to 1.0)
            fatal_count: Number of fatal failures in trials
            score: Newly calculated score based on more trials
        """
        is_targeting_eight_legally = (my_targets == ['8'])
        fatal_count = 0
        error_count = 0
        scores = []
        for trial in range(num_trials):
            try:
                curr_fatal_rate = fatal_count / (trial + 1)
                if curr_fatal_rate > fatal_threshold:
                    return curr_fatal_rate, fatal_count, -999
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                noise = self.noise_std
                V0 = np.clip(action['V0'] + np.random.normal(0, noise['V0']), 0.5, 8.0)
                phi = (action['phi'] + np.random.normal(0, noise['phi'])) % 360
                theta = np.clip(action['theta'] + np.random.normal(0, noise['theta']), 0, 90)
                a = np.clip(action['a'] + np.random.normal(0, noise['a']), -0.5, 0.5)
                b = np.clip(action['b'] + np.random.normal(0, noise['b']), -0.5, 0.5)
                cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                
                pt.simulate(shot, inplace=True)
                trial_score = self._improved_reward_function(
                    shot,
                    {bid: copy.deepcopy(ball) for bid, ball in balls.items()},
                    my_targets,
                    sim_table
                )
                scores.append(trial_score)
                new_pocketed = [bid for bid in sim_balls.keys()
                            if sim_balls[bid].state.s == 4 and balls[bid].state.s != 4]
                
                cue_pocketed = "cue" in new_pocketed
                eight_pocketed = "8" in new_pocketed
                
                # Debug: print first trial
                if trial == 0:
                    print(f"[Fatal Check] Sample: cue={cue_pocketed}, eight={eight_pocketed}, pocketed={new_pocketed}")
                
                # Fatal condition 1: Cue and eight both pocketed
                if cue_pocketed and eight_pocketed:
                    fatal_count += 1
                    continue
                
                # Fatal condition 2: Eight pocketed illegally (when not targeting it)
                if eight_pocketed and not is_targeting_eight_legally:
                    fatal_count += 1
                    continue
                    
            except Exception as e:
                if trial == 0:  
                    print(f"[Fatal Check] Simulation error: {e}")
                # Note: simulation errors are not considered fatal
                error_count += 1
                continue
        
        # Count successful simulations
        success_count = num_trials - error_count
    
        # If successful simulations are zero, our action is guaranteed to fail
        if success_count == 0:
            print(f"[Fatal Check] WARNING: All {num_trials} trials failed to simulate")
            return 1.0, num_trials, -999
        
        if error_count > 0:
            print(f"[Fatal Check] {error_count}/{num_trials} trials had errors")
        
        # Calculate fatal rate according to successful simulations
        fatal_rate = fatal_count / success_count
        return fatal_rate, fatal_count, float(np.mean(scores))

    # ==================== Geometric Calculation ====================
    
    def _calc_ghost_ball(self, target_pos, pocket_pos):
        """Calculate ghost ball position for aiming"""
        direction = self._unit_vector(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        ghost_pos = np.array(target_pos[:2]) - direction * (2 * self.BALL_RADIUS)
        return ghost_pos
    
    def _geo_shot(self, cue_pos, target_pos, pocket_pos):
        """Calculate shot parameters using geometry"""
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        cue_to_ghost = ghost_pos - np.array(cue_pos[:2])
        direction = self._unit_vector(cue_to_ghost)
        phi = self._direction_to_degrees(direction)
        
        dist = self._calc_dist(cue_pos, ghost_pos)
        if dist < 0.3:
            V0 = 2.0
        elif dist < 0.8:
            V0 = 2.0 + dist * 1.5
        else:
            V0 = 4.0 + dist * 0.8
        V0 = min(V0, 7.5)
        
        return {
            'V0': float(V0),
            'phi': float(phi),
            'theta': 0.0,
            'a': 0.0,
            'b': 0.0
        }
    
    def _calculate_cut_angle(self, cue_pos, target_pos, pocket_pos):
        """Calculate the cut angle between cue and target"""
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        vec1 = self._unit_vector(np.array(ghost_pos) - np.array(cue_pos[:2]))
        vec2 = self._unit_vector(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle = np.arccos(dot) * 180 / np.pi
        return angle
    
    # ==================== Target Ball Selection ====================
    
    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        """Count balls blocking the path between two positions"""
        count = 0
        line_vec = np.array(to_pos[:2]) - np.array(from_pos[:2])
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-6:
            return 0
        
        line_dir = line_vec / line_length
        
        for bid, ball in balls.items():
            if bid in exclude_ids or ball.state.s == 4:
                continue
            
            ball_pos = ball.state.rvw[0][:2]
            vec_to_ball = ball_pos - np.array(from_pos[:2])
            proj_length = np.dot(vec_to_ball, line_dir)
            
            if proj_length < 0 or proj_length > line_length:
                continue
            
            proj_point = np.array(from_pos[:2]) + line_dir * proj_length
            dist_to_line = np.linalg.norm(ball_pos - proj_point)
            
            if dist_to_line < self.BALL_RADIUS * 2.5:
                count += 1
        
        return count
    
    def _evaluate_pocket_angle(self, target_pos, pocket_pos):
        """Score pocket alignment (closer = better)"""
        dist = self._calc_dist(target_pos, pocket_pos)
        return 1.0 / (1.0 + dist)
    
    def _choose_top_targets(self, balls, my_targets, table, num_choices=3):
        """
        Select top N target-pocket combinations.
        For black eight: select top 5 choices
        For regular balls: select top 3 choices
        """
        all_choices = []
        cue_pos = balls['cue'].state.rvw[0]
        black_8_pos = balls['8'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                score = 0
                
                # Distance factor
                dist_cue_to_target = self._calc_dist(cue_pos, target_pos)
                score += 50 / (1 + dist_cue_to_target)
                
                # Pocket angle quality
                angle_quality = self._evaluate_pocket_angle(target_pos, pocket_pos)
                score += angle_quality * 60
                
                # Cut angle (closer to 0 is better)
                cut_angle = self._calculate_cut_angle(cue_pos, target_pos, pocket_pos)
                score += (90 - cut_angle) / 90 * 40
                
                # Obstruction penalties
                obstruction1 = self._count_obstructions(balls, cue_pos, target_pos, 
                                                        exclude_ids=['cue', target_id])
                score -= obstruction1 * 25
                
                obstruction2 = self._count_obstructions(balls, target_pos, pocket_pos, 
                                                        exclude_ids=['cue', target_id])
                score -= obstruction2 * 30
                
                # Black eight safety distance
                if target_id != '8':
                    dist_to_black_8 = self._calc_dist(target_pos, black_8_pos)
                    min_safe_distance = 0.3
                    if dist_to_black_8 < min_safe_distance:
                        proximity_penalty = ((min_safe_distance - dist_to_black_8) / 
                                           min_safe_distance) ** 2 * 150
                        score -= proximity_penalty
                
                all_choices.append((target_id, pocket_id, score))
        
        all_choices.sort(key=lambda x: x[2], reverse=True)
        
        # For black eight: select top 5, otherwise top 3
        if my_targets == ['8']:
            return all_choices[:5]
        
        return all_choices[:num_choices]
    
    # ==================== Optimization Search ====================
    def _evaluate_action(self, action, trials, balls, my_targets, table, threshold=20, enable_noise=False):
        """
        Evaluate action with early stopping mechanism.
        
        Parameters:
            threshold: if any single trial score < threshold, return immediately
                      with current mean instead of completing all trials
        """
        scores = []
        try:
            for trial_idx in range(trials):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                if enable_noise:
                    noise = self.noise_std
                    V0 = np.clip(action['V0'] + np.random.normal(0, noise['V0']), 0.5, 8.0)
                    phi = (action['phi'] + np.random.normal(0, noise['phi'])) % 360
                    theta = np.clip(action['theta'] + np.random.normal(0, noise['theta']), 0, 90)
                    a = np.clip(action['a'] + np.random.normal(0, noise['a']), -0.5, 0.5)
                    b = np.clip(action['b'] + np.random.normal(0, noise['b']), -0.5, 0.5)
                    cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                else:
                    cue.set_state(**action)

                pt.simulate(shot, inplace=True)
                trial_score = self._improved_reward_function(
                    shot,
                    {bid: copy.deepcopy(ball) for bid, ball in balls.items()},
                    my_targets,
                    sim_table
                )
                if trials == 1:
                    return trial_score
                
                scores.append(trial_score)
                
                # Early stopping: if current trial score below threshold, return immediately
                if trial_score < threshold:
                    result = float(np.mean(scores))
                    return result

            return float(np.mean(scores))

        except Exception as e:
            print(f"[NewAgent] Evaluation error: {e}")
            return -999

    def _cma_es_optimized(self, geo_action, balls, my_targets, table, 
                      is_black_eight=False, is_opening=False):
        """
        CMA-ES优化 - 加速版本
        
        优化策略:
        1. 减少迭代次数和种群大小
        2. 使用自适应 trials (开始用少量，接近收敛时增加)
        3. 早停机制：连续N代无改进则停止
        """
        initial_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 原始边界
        bounds_original = np.array([
            [max(0.5, geo_action['V0'] - 2.0), min(8.0, geo_action['V0'] + 2.0)],
            [geo_action['phi'] - 20, geo_action['phi'] + 20],
            [0, 15],
            [-0.3, 0.3],
            [-0.3, 0.3]
        ])
        
        def normalize(x):
            return (x - bounds_original[:, 0]) / (bounds_original[:, 1] - bounds_original[:, 0])
        
        def denormalize(x_norm):
            return bounds_original[:, 0] + x_norm * (bounds_original[:, 1] - bounds_original[:, 0])
        
        # 归一化的初始点
        x0_norm = normalize(np.array([
            geo_action['V0'],
            geo_action['phi'],
            geo_action['theta'],
            geo_action['a'],
            geo_action['b']
        ]))
        try_times = 1 if not is_black_eight else 2
        # ========== 加速参数 ==========
        # 更激进的参数设置
        if is_opening:
            maxiter = 3  # 开局更快
            popsize = 3
            sigma0 = 0.25
        elif is_black_eight:
            maxiter = 6  # 黑八稍微谨慎
            popsize = 8
            sigma0 = 0.15
        else:
            maxiter = 4  # 普通球快速
            popsize = 6
            sigma0 = 0.3
        
        opts = {
            'bounds': [[0]*5, [1]*5],
            'maxiter': maxiter,
            'popsize': popsize,
            'verb_disp': 0,
            'verb_log': 0,
            'tolfun': 1e-3,  # 提前收敛阈值（分数改进小于此值则停止）
            'tolx': 1e-3     # 参数变化小于此值则停止
        }
        
        # ========== 自适应trials策略 ==========
        eval_count = [0]
        
        def objective(x_norm):
            eval_count[0] += 1

            x = denormalize(np.clip(x_norm, 0, 1))
            
            restored_balls = {bid: copy.deepcopy(ball) 
                            for bid, ball in initial_balls.items()}
            
            action = {
                'V0': float(np.clip(x[0], 0.5, 8.0)),
                'phi': float(x[1] % 360),
                'theta': float(np.clip(x[2], 0, 90)),
                'a': float(np.clip(x[3], -0.5, 0.5)),
                'b': float(np.clip(x[4], -0.5, 0.5))
            }
            
            score = self._evaluate_action(
                action, 
                try_times,
                restored_balls, 
                my_targets, 
                table,
                threshold=10,
                enable_noise=True if is_black_eight else False
            )
            
            return -score
        
        try:
            es = cma.CMAEvolutionStrategy(x0_norm, sigma0, opts)
            es.optimize(objective)
            
            # 反归一化最优解
            best_x = denormalize(np.clip(es.result.xbest, 0, 1))
            best_score = -es.result.fbest
            
            best_action = {
                'V0': float(np.clip(best_x[0], 0.5, 8.0)),
                'phi': float(best_x[1] % 360),
                'theta': float(np.clip(best_x[2], 0, 90)),
                'a': float(np.clip(best_x[3], -0.5, 0.5)),
                'b': float(np.clip(best_x[4], -0.5, 0.5))
            }
            
            print(f"[CMA-ES] Converged: score={best_score:.2f}, evals={es.result.evaluations}")
            return best_action, best_score
            
        except Exception as e:
            print(f"[CMA-ES] Failed: {e}")
            return None, -999
    
    # ==================== Game State Detection ====================
    
    def _detect_opening_state(self, balls):
        """
        Detect if current state is an opening position.
        Opening state: most colored balls still on table (>= 12 balls)
        
        Returns:
            True if opening state detected, False otherwise
        """
        colored_balls = [bid for bid in balls.keys() if bid not in ['cue', '8']]
        colored_on_table = [bid for bid in colored_balls if balls[bid].state.s != 4]
        
        is_opening = len(colored_on_table) >= 12
        if is_opening:
            print(f"[NewAgent] Opening state detected ({len(colored_on_table)} colored balls on table)")
        return is_opening
    
    # ==================== Main Decision Logic ====================
    
    def decision(self, balls=None, my_targets=None, table=None):
        """
        Main decision function with early-stopping safety check.
        
        Strategy:
        1. Evaluate geometric shots, return if good enough
        2. Optimize candidates one-by-one
        3. After each optimization, check safety immediately
        4. Return first action that meets both score and safety thresholds
        """
        if not all([balls, my_targets, table]):
            print("[NewAgent] Incomplete parameters")
            return self._safe_action()
        
        try:
            # Detect opening state
            is_opening = self._detect_opening_state(balls)
            
            # Switch to black eight if all own balls pocketed
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if not remaining:
                my_targets = ['8']
                print("[NewAgent] Switching to black eight")
            
            is_black_eight = (my_targets == ['8'])
            cue_pos = balls['cue'].state.rvw[0]

            # Define thresholds based on ball type
            GEO_THRESHOLD = 220 if is_black_eight else 60
            SCORE_THRESHOLD = 200 if is_black_eight else 50
            SAFE_FATAL_THRESHOLD = 0  # Maximum allowed fatal failure rate
            PRE_TRIALS = 8 if is_black_eight else 5

            # ============ Layer 1: Select Best Targets ============
            num_choices = 5 if is_black_eight else 3
            top_choices = self._choose_top_targets(balls, my_targets, table, num_choices=num_choices)
            
            if not top_choices:
                print("[NewAgent] No valid targets available, using safe action")
                return self._safe_action()

            # ============ Layer 2: Quick Geometric Check ============
            all_initial_candidates = []
            
            for target_id, pocket_id, target_score in top_choices:
                target_pos = balls[target_id].state.rvw[0]
                pocket_pos = table.pockets[pocket_id].center
                geo_action = self._geo_shot(cue_pos, target_pos, pocket_pos)
                geo_score = self._evaluate_action(
                    geo_action, PRE_TRIALS, balls, my_targets, table, 20, True
                )
                
                # If geometric shot is excellent, check safety and return immediately
                if geo_score > GEO_THRESHOLD:
                    fatal_rate, fatal_count, verified_score = self._check_fatal_failure(
                        geo_action, balls, my_targets, table, num_trials=15, 
                        fatal_threshold=SAFE_FATAL_THRESHOLD
                    )
                    print(f"[NewAgent] Excellent geo_action {target_id}→{pocket_id}: "
                        f"score={verified_score:.2f}, fatal_rate={fatal_rate:.1%}")
                    
                    if fatal_rate <= SAFE_FATAL_THRESHOLD and verified_score > 0:
                        print(f"[NewAgent] Using excellent geometric shot (no optimization needed)")
                        return geo_action
                
                all_initial_candidates.append((geo_action, geo_score, f'{target_id}→{pocket_id}'))
            
            # ============ Layer 3: Optimize with Early Stopping ============
            num_to_optimize = 5 if is_black_eight else 3
            all_initial_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = all_initial_candidates[:num_to_optimize]
            
            print(f"[NewAgent] Optimizing top {len(top_candidates)} candidates:")
            for idx, (_, score, shot_type) in enumerate(top_candidates):
                print(f"  {idx+1}. {shot_type}: {score:.1f}")
            
            # Track all evaluated candidates as fallback
            all_evaluated = []
            
            for idx, (base_action, base_score, shot_type) in enumerate(top_candidates):
                print(f"\n[NewAgent] Optimizing {idx+1}/{len(top_candidates)}: {shot_type}")
                
                # Optimize
                opt_action, opt_score = self._cma_es_optimized(
                    base_action, balls, my_targets, table, 
                    is_black_eight=is_black_eight, is_opening=is_opening
                )
                
                # Use optimized if successful, otherwise use base
                if opt_action is not None:
                    action_to_check = opt_action
                    score_to_check = opt_score
                    print(f"[NewAgent] Optimization improved score: {base_score:.2f} → {opt_score:.2f}")
                else:
                    action_to_check = base_action
                    score_to_check = base_score
                    print(f"[NewAgent] Optimization failed, using base action")
                
                # Immediate safety check
                fatal_rate, fatal_count, verified_score = self._check_fatal_failure(
                    action_to_check, balls, my_targets, table, num_trials=15,
                    fatal_threshold=SAFE_FATAL_THRESHOLD
                )
                
                print(f"[NewAgent] {shot_type}: score={verified_score:.2f}, "
                    f"fatal_rate={fatal_rate:.1%} ({fatal_count}/15 fatal)")
                
                all_evaluated.append((action_to_check, verified_score, shot_type, fatal_rate))
                
                # Early stopping: if this action meets both thresholds, use it immediately
                if verified_score >= SCORE_THRESHOLD and fatal_rate <= SAFE_FATAL_THRESHOLD:
                    print(f"[NewAgent] ✓ Found acceptable action: {shot_type} "
                        f"(score={verified_score:.2f}, fatal_rate={fatal_rate:.1%})")
                    return action_to_check
            
            # ============ Layer 4: Fallback Selection ============
            print(f"\n[NewAgent] No action met threshold (score>={SCORE_THRESHOLD}), "
                f"selecting best safe option...")
            
            # Filter safe candidates (fatal_rate <= threshold)
            safe_candidates = [(a, s, st, f) for a, s, st, f in all_evaluated 
                            if f <= SAFE_FATAL_THRESHOLD]
            
            if safe_candidates:
                # Select highest scoring safe action
                best_action, best_score, best_type, fatal_rate = max(safe_candidates, key=lambda x: x[1])
                
                if best_score > 0:
                    print(f"[NewAgent] Using best safe option: {best_type} "
                        f"(score={best_score:.2f}, fatal_rate={fatal_rate:.1%})")
                    return best_action
            
            # Last resort: use safe action
            print(f"[NewAgent] No safe options with positive score, using defensive safe action")
            return self._safe_action()
            
        except Exception as e:
            print(f"[NewAgent] Decision failed: {e}")
            import traceback
            traceback.print_exc()
            return self._safe_action()
