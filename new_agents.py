import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime

from .agent import Agent
import copy
import signal
import random
import optuna
import cma
import logging

# Disable Optuna logs
logging.getLogger('optuna').setLevel(logging.WARNING)
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
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
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
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

import cma
class NewAgent(Agent):
    """
    Advanced pool agent with strategic offense and defense capabilities.
    Uses CMA-ES optimization for shot refinement and evaluates safety plays.
    """
    
    def __init__(self):
        super().__init__()
        
        # Table and ball constants
        self.BALL_RADIUS = 0.028575
        self.TABLE_WIDTH = 1.9812
        self.TABLE_HEIGHT = 0.9906
        
        # Cushion positions for banking calculations
        self.cushions = {
            'x_pos': self.TABLE_WIDTH / 2,
            'x_neg': -self.TABLE_WIDTH / 2,
            'y_pos': self.TABLE_HEIGHT / 2,
            'y_neg': -self.TABLE_HEIGHT / 2
        }
        
        # Noise parameters for robustness testing
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        
        self.last_strategy = None  # Track the strategy used in the last decision
        
        print("[Agent] Strategic pool agent initialized with offense/defense capabilities")
    
    # ============================================================================
    # Basic Utility Functions
    # ============================================================================
    
    def _safe_action(self):
        """Return a no-op action when all else fails."""
        return {'V0': 0, 'phi': 0, 'theta': 0, 'a': 0, 'b': 0}
    
    def _calc_dist(self, pos1, pos2):
        """Calculate 2D Euclidean distance between two positions."""
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))
    
    def _unit_vector(self, vec):
        """Normalize a 2D vector to unit length."""
        vec = np.array(vec[:2])
        norm = np.linalg.norm(vec)
        return np.array([1.0, 0.0]) if norm < 1e-6 else vec / norm
    
    def _direction_to_degrees(self, direction_vec):
        """Convert a direction vector to angle in degrees [0, 360)."""
        phi = np.arctan2(direction_vec[1], direction_vec[0]) * 180 / np.pi
        return phi % 360
    
    def _distance_to_nearest_pocket(self, ball_pos, table):
        """Find the shortest distance from a ball position to any pocket."""
        return min(self._calc_dist(ball_pos, pocket.center) 
                   for pocket in table.pockets.values())
    
    # ============================================================================
    # Reward and Evaluation Functions
    # ============================================================================
    
    def _improved_reward_function(self, shot, last_state, player_targets, table):
        """
        Enhanced reward function that penalizes risky eight-ball positioning
        and rewards safe cue ball placement.
        """
        # 先定义相关变量
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
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
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        if first_contact_ball_id is None:
            if len(last_state) > 2 or player_targets != ['8']:
                foul_first_hit = True
        else:
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

        base_score = analyze_shot_for_reward(shot, last_state, player_targets)

        # Check if we're legally targeting the eight ball
        targeting_eight = (player_targets == ['8'])

        # Penalize moving eight ball closer to pockets when not targeting it
        if not targeting_eight and '8' in shot.balls and shot.balls['8'].state.s != 4:
            eight_before = self._distance_to_nearest_pocket(
                last_state['8'].state.rvw[0], table)
            eight_after = self._distance_to_nearest_pocket(
                shot.balls['8'].state.rvw[0], table)
            if eight_after < eight_before:
                base_score -= (eight_before - eight_after) * 150

        # Reward keeping cue ball away from pockets (unless scratched)
        if not cue_pocketed and 'cue' in shot.balls:
            cue_pos = shot.balls['cue'].state.rvw[0]
            cue_dist = self._distance_to_nearest_pocket(cue_pos, table)
            if cue_dist < 0.1:
                base_score -= 30 * (0.1 - cue_dist) / 0.1
            elif cue_dist > 0.2:
                base_score += min(15, cue_dist * 20)

        # Reward positioning cue ball closer to next target ball (excluding eight ball)
        # 只有本杆进球或完全安全（无进球但无犯规）时才奖励
        if not cue_pocketed and 'cue' in shot.balls and not targeting_eight:
            cue_pos_after = shot.balls['cue'].state.rvw[0][:2]
            cue_pos_before = last_state['cue'].state.rvw[0][:2]
            remaining_targets = [bid for bid in player_targets 
                               if bid != '8' and shot.balls[bid].state.s != 4]
            is_safe = (
                len(new_pocketed) == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail
            )
            if remaining_targets and (own_pocketed or is_safe):
                dist_before = min(self._calc_dist(cue_pos_before, shot.balls[bid].state.rvw[0][:2]) 
                                for bid in remaining_targets)
                dist_after = min(self._calc_dist(cue_pos_after, shot.balls[bid].state.rvw[0][:2]) 
                               for bid in remaining_targets)
                if dist_after < dist_before:
                    base_score += max(min(25, (dist_before - dist_after) * 50), 0)  # Reward up to 25 points

        return base_score
    
    def _evaluate_action(self, action, trials, balls, my_targets, table, 
                        threshold=20, enable_noise=False):
        """
        Simulate an action multiple times and return average reward.
        Early termination if consistently poor.
        """
        scores = []
        
        try:
            for trial_num in range(trials):
                # Deep copy game state for simulation
                sim_balls = {bid: copy.deepcopy(ball) 
                            for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(table=sim_table, balls=sim_balls, 
                               cue=pt.Cue(cue_ball_id="cue"))
                
                # Apply noise if requested for robustness testing
                if enable_noise:
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
                else:
                    shot.cue.set_state(**action)
                
                # Run simulation
                if not simulate_with_timeout(shot, timeout=3):
                    scores.append(-100)
                    continue
                
                trial_score = self._improved_reward_function(
                    shot, balls, my_targets, sim_table)
                scores.append(trial_score)
                
                # Early exit if consistently bad
                if trial_score < threshold and len(scores) > 1:
                    return float(np.mean(scores))
            
            return float(np.mean(scores))
            
        except Exception:
            return -999
    
    def _check_fatal_failure(self, action, balls, my_targets, table, 
                            num_trials=10, fatal_threshold=0.1):
        """
        Check for catastrophic outcomes (scratching eight ball, etc.).
        Returns (fatal_rate, fatal_count, avg_score).
        """
        targeting_eight = (my_targets == ['8'])
        fatal_count = 0
        error_count = 0
        scores = []
        
        for trial in range(num_trials):
            try:
                # Early termination if fatal rate is already too high
                if fatal_count / (trial + 1) > fatal_threshold + 0.1:
                    break
                
                # Simulate with noise
                sim_balls = {bid: copy.deepcopy(ball) 
                            for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(table=sim_table, balls=sim_balls,
                               cue=pt.Cue(cue_ball_id="cue"))
                
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
                    error_count += 1
                    continue
                
                scores.append(self._improved_reward_function(
                    shot, balls, my_targets, sim_table))
                
                # Check for fatal outcomes
                new_pocketed = [bid for bid, ball in shot.balls.items()
                               if ball.state.s == 4 and balls[bid].state.s != 4]
                
                if "cue" in new_pocketed and "8" in new_pocketed:
                    fatal_count += 1
                elif "8" in new_pocketed and not targeting_eight:
                    fatal_count += 1
                    
            except Exception:
                error_count += 1
        
        success_count = num_trials - error_count
        if success_count == 0:
            return 1.0, num_trials, -999
        
        fatal_rate = fatal_count / success_count
        avg_score = float(np.mean(scores)) if scores else -500
        
        return fatal_rate, fatal_count, avg_score
    
    def _check_fatal_failure_early_stop(self, action, balls, my_targets, table, 
                            num_trials=10, fatal_threshold=0.1, score_threshold=20):
        """
        Check for catastrophic outcomes (scratching eight ball, etc.).
        Returns (fatal_rate, fatal_count, avg_score).
        """
        targeting_eight = (my_targets == ['8'])
        fatal_count = 0
        error_count = 0
        scores = []
        
        for trial in range(num_trials):
            try:
                # Early termination if fatal rate is already too high
                if fatal_count / (trial + 1) > fatal_threshold + 0.1:
                    break
                
                # Simulate with noise
                sim_balls = {bid: copy.deepcopy(ball) 
                            for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                shot = pt.System(table=sim_table, balls=sim_balls,
                               cue=pt.Cue(cue_ball_id="cue"))
                
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
                    error_count += 1
                    continue
                
                scores.append(self._improved_reward_function(
                    shot, balls, my_targets, sim_table))
                
                # Check for fatal outcomes
                new_pocketed = [bid for bid, ball in shot.balls.items()
                               if ball.state.s == 4 and balls[bid].state.s != 4]
                
                if "cue" in new_pocketed and "8" in new_pocketed:
                    fatal_count += 1
                elif "8" in new_pocketed and not targeting_eight:
                    fatal_count += 1
                
                if len(scores) >= 1 and np.mean(scores) < score_threshold:
                    return 1.0, fatal_count, float(np.mean(scores)), "Early Stop: Low Score"
            except Exception:
                error_count += 1
        
        success_count = num_trials - error_count
        if success_count == 0:
            return 1.0, num_trials, -999, "No successful trials"
        
        fatal_rate = fatal_count / success_count
        avg_score = float(np.mean(scores)) if scores else -500
        
        return fatal_rate, fatal_count, avg_score, "Completed"
    
    # ============================================================================
    # Geometric Shot Calculations
    # ============================================================================
    
    def _calc_ghost_ball(self, target_pos, pocket_pos):
        """
        Calculate ghost ball position for perfect contact.
        Ghost ball is where cue ball should be at moment of contact.
        """
        direction = self._unit_vector(
            np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        return np.array(target_pos[:2]) - direction * (2 * self.BALL_RADIUS)
    
    def _calculate_cut_angle(self, cue_pos, target_pos, pocket_pos):
        """
        Calculate the signed cut angle in degrees.
        Positive: counterclockwise cut, negative: clockwise.
        """
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        vec1 = self._unit_vector(ghost_pos - np.array(cue_pos[:2]))
        vec2 = self._unit_vector(np.array(pocket_pos[:2]) - np.array(target_pos[:2]))
        
        # Calculate signed angle using atan2
        angle1 = np.arctan2(vec1[1], vec1[0])
        angle2 = np.arctan2(vec2[1], vec2[0])
        angle_diff = (angle2 - angle1) * 180 / np.pi
        
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff <= -180:
            angle_diff += 360
        
        return angle_diff
    
    def _calc_curve_a(self, cut_angle):
        """
        Calculate side spin 'a' for curve shot based on signed cut angle.
        Positive cut_angle: use positive a (counterclockwise spin)
        Negative cut_angle: use negative a (clockwise spin)
        """
        sign = 1 if cut_angle <= 0 else -1
        magnitude = min(0.5, abs(cut_angle) / 180 * 0.5)
        return sign * magnitude
    
    def _geo_shot(self, cue_pos, target_pos, pocket_pos):
        """Generate basic geometric shot parameters."""
        ghost_pos = self._calc_ghost_ball(target_pos, pocket_pos)
        direction = self._unit_vector(ghost_pos - np.array(cue_pos[:2]))
        phi = self._direction_to_degrees(direction)
        
        dist = self._calc_dist(cue_pos, ghost_pos)
        
        # Adaptive power based on distance
        if dist < 0.8:
            V0 = min(2.0 + dist * 1.5, 7.5)
        else:
            V0 = min(4.0 + dist * 0.8, 7.5)
        
        return {
            'V0': float(V0),
            'phi': float(phi),
            'theta': 0.0,
            'a': 0.0,
            'b': 0.0
        }
    
    def _geo_bank_shot(self, cue_pos, target_pos, pocket_pos, cushion_id):
        """
        Generate bank shot by mirroring pocket position across cushion.
        Uses the reflection principle for banking.
        """
        mirrored_pocket = np.array(pocket_pos[:2])
        
        if 'x' in cushion_id:
            mirrored_pocket[0] = 2 * self.cushions[cushion_id] - mirrored_pocket[0]
        else:
            mirrored_pocket[1] = 2 * self.cushions[cushion_id] - mirrored_pocket[1]
        
        return self._geo_shot(cue_pos, target_pos, mirrored_pocket)
    
    # ============================================================================
    # Target Selection and Obstruction Detection
    # ============================================================================
    
    def _count_obstructions(self, balls, from_pos, to_pos, exclude_ids=['cue']):
        """
        Count balls obstructing the line between two positions.
        Uses perpendicular distance to line for detection.
        """
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
            
            # Check if ball is along the line segment
            if 0 < proj_length < line_length:
                perp_dist = np.linalg.norm(
                    ball.state.rvw[0][:2] - 
                    (np.array(from_pos[:2]) + line_dir * proj_length)
                )
                
                if perp_dist < self.BALL_RADIUS * 2.2:
                    count += 1
        
        return count
    
    def _choose_top_targets(self, balls, my_targets, table, 
                           num_choices=3, is_defense=False):
        """
        Rank all possible shots by geometric favorability.
        Returns top N shot configurations.
        """
        all_choices = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            # Evaluate direct shots to each pocket
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # Check for clear path
                cue_to_target_obs = self._count_obstructions(
                    balls, cue_pos, target_pos, ['cue', target_id])
                target_to_pocket_obs = self._count_obstructions(
                    balls, target_pos, pocket_pos, ['cue', target_id])
                
                
                dist = (self._calc_dist(cue_pos, target_pos) + 
                        self._calc_dist(target_pos, pocket_pos))
                cut_angle = self._calculate_cut_angle(
                    cue_pos, target_pos, pocket_pos)
                
                score = 100 - (dist * 20 + abs(cut_angle) * 0.5)
                
                # Penalize shots near the eight ball
                if (target_id != '8' and '8' in balls and 
                    balls['8'].state.s != 4):
                    eight_dist = self._calc_dist(
                        target_pos, balls['8'].state.rvw[0])
                    if eight_dist < 0.3:
                        score -= (0.3 - eight_dist) * 150

                if cue_to_target_obs == 0 and target_to_pocket_obs == 0:
                    all_choices.append({
                        'type': 'direct',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'score': score
                    })

                elif cue_to_target_obs <= 1 and target_to_pocket_obs <= 1:
                    curve_score = score - 20 - abs(cut_angle) * 0.1 # Penalize curve shots slightly
                    all_choices.append({
                        'type': 'curve',
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'cut_angle': cut_angle,
                        'score': curve_score
                    })
                
                # Skip bank shots in defense mode
                if is_defense:
                    continue
                
                # Evaluate bank shots
                for cushion_id in self.cushions.keys():
                    mirrored_pocket = np.array(pocket_pos[:2])
                    idx = 0 if 'x' in cushion_id else 1
                    mirrored_pocket[idx] = (2 * self.cushions[cushion_id] - 
                                           mirrored_pocket[idx])
                    
                    cue_clear = self._count_obstructions(
                        balls, cue_pos, target_pos, ['cue', target_id]) == 0
                    bank_clear = self._count_obstructions(
                        balls, target_pos, mirrored_pocket, 
                        ['cue', target_id]) == 0
                    
                    if cue_clear and bank_clear:
                        dist = (self._calc_dist(cue_pos, target_pos) + 
                               self._calc_dist(target_pos, mirrored_pocket))
                        cut_angle = self._calculate_cut_angle(
                            cue_pos, target_pos, mirrored_pocket)
                        
                        score = 100 - (dist * 25 + abs(cut_angle) * 0.6 + 40)
                        
                        if score > 0:
                            all_choices.append({
                                'type': 'bank',
                                'target_id': target_id,
                                'pocket_id': pocket_id,
                                'cushion_id': cushion_id,
                                'score': score
                            })
        
        all_choices.sort(key=lambda x: x['score'], reverse=True)
        return all_choices[:num_choices]
    
    # ============================================================================
    # CMA-ES Optimization
    # ============================================================================
    
    def _cma_es_optimized(self, geo_action, balls, my_targets, table,
                         is_black_eight=False, is_opening=False):
        """
        Refine geometric shot using CMA-ES evolutionary optimization.
        Searches parameter space around initial geometric solution.
        """
        # Define search bounds around geometric solution
        bounds = np.array([
            [max(0.5, geo_action['V0'] - 1.5), min(8.0, geo_action['V0'] + 1.5)],
            [geo_action['phi'] - 15, geo_action['phi'] + 15],
            [0, 10],
            [max(geo_action['a'] - 0.2, -0.5), min(0.5, geo_action['a'] + 0.2)],
            [max(geo_action['b'] - 0.2, -0.5), min(0.5, geo_action['b'] + 0.2)]
        ])
        
        if is_opening:
            MAXITER = 3
            POPSIZE = 3
        elif is_black_eight:
            MAXITER = 5
            POPSIZE = 8
        else:
            MAXITER = 5
            POPSIZE = 5

        def normalize(x):
            return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        
        def denormalize(x):
            return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])
        
        # Starting point (normalized)
        x0_norm = normalize(np.array([
            geo_action['V0'],
            geo_action['phi'],
            0, geo_action['a'], geo_action['b']
        ]))
        
        # CMA-ES options
        opts = {
            'bounds': [[0]*5, [1]*5],
            'maxiter': MAXITER,
            'popsize': POPSIZE,
            'verb_disp': 0,
            'verb_log': 0
        }
        
        def objective(x_norm):
            """Objective function: negative average reward."""
            x = denormalize(np.clip(x_norm, 0, 1))
            action = {
                'V0': float(x[0]),
                'phi': float(x[1]),
                'theta': float(x[2]),
                'a': float(x[3]),
                'b': float(x[4])
            }
            trials = 3 if is_black_eight else 2
            return -self._evaluate_action(
                action, trials, balls, my_targets, table, 20, True)
        
        try:
            es = cma.CMAEvolutionStrategy(x0_norm, 0.2, opts)
            es.optimize(objective)
            
            best_x = denormalize(np.clip(es.result.xbest, 0, 1))
            best_action = {
                'V0': float(best_x[0]),
                'phi': float(best_x[1]),
                'theta': float(best_x[2]),
                'a': float(best_x[3]),
                'b': float(best_x[4])
            }
            
            return best_action, -es.result.fbest
            
        except Exception:
            return None, -999
    
    # ============================================================================
    # Defensive Strategy
    # ============================================================================
    
    def _get_opponent_targets(self, my_targets):
        """Determine which balls belong to opponent."""
        all_balls = set(str(i) for i in range(1, 16))
        my_set = set(my_targets)
        opponent_set = all_balls - my_set - {'8'}
        return list(opponent_set)
    
    def _evaluate_defensive_action(self, action, balls, my_targets, table):
        """
        Evaluate a defensive action and return a reward score.
        Higher score is better defense.
        """
        # Simulate our defensive shot
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        shot = pt.System(table=sim_table, balls=sim_balls,
                        cue=pt.Cue(cue_ball_id="cue"))
        
        shot.cue.set_state(**action)
        
        if not simulate_with_timeout(shot, timeout=3):
            return 0  # Simulation timeout due to CPU speed, not action quality
        
        # Check if our safety shot causes a foul
        safety_reward = analyze_shot_for_reward(shot, balls, my_targets)
        if safety_reward < 0:
            return -100 + safety_reward  # Fouls are bad
        
        # Evaluate opponent's best shot from resulting position
        opponent_targets = self._get_opponent_targets(my_targets)
        opponent_choices = self._choose_top_targets(
            shot.balls, opponent_targets, shot.table,
            num_choices=5, is_defense=True)
        
        if not opponent_choices:
            return 200  # Perfect defense - opponent has no shot
        
        # Opponent's best score (we want this minimized, so negative)
        opp_best_score = opponent_choices[0]['score']
        
        # Additional reward for positioning cue ball near our balls
        cue_pos = shot.balls['cue'].state.rvw[0][:2]
        near_our_ball_bonus = 0
        for my_ball_id in my_targets:
            if shot.balls[my_ball_id].state.s != 4:
                my_ball_pos = shot.balls[my_ball_id].state.rvw[0][:2]
                dist = self._calc_dist(cue_pos, my_ball_pos)
                if dist < self.BALL_RADIUS * 3:
                    near_our_ball_bonus += 50  # Bonus for being close to our balls
        
        # Reward for hiding cue ball from opponent's targets
        hide_bonus = 0
        for opp_target in opponent_targets:
            if shot.balls[opp_target].state.s != 4:
                opp_target_pos = shot.balls[opp_target].state.rvw[0][:2]
                dist_to_opp = self._calc_dist(cue_pos, opp_target_pos)
                if dist_to_opp > 0.5:  # Cue ball far from opponent's targets
                    hide_bonus += 20
        
        # New: Penalize if opponent can easily foul us back
        # Simulate opponent's possible foul shots
        foul_penalty = 0
        for opp_choice in opponent_choices[:3]:  # Check top 3 opponent shots
            # Simulate opponent's shot
            opp_sim_balls = {bid: copy.deepcopy(ball) for bid, ball in shot.balls.items()}
            opp_sim_table = copy.deepcopy(shot.table)
            opp_shot = pt.System(table=opp_sim_table, balls=opp_sim_balls,
                               cue=pt.Cue(cue_ball_id="cue"))
            
            # Use geometric action for opponent (since BasicAgentPro uses heuristics)
            opp_cue_pos = opp_sim_balls['cue'].state.rvw[0]
            opp_target_pos = opp_sim_balls[opp_choice['target_id']].state.rvw[0]
            opp_pocket_pos = opp_sim_table.pockets[opp_choice['pocket_id']].center
            
            opp_action = self._geo_shot(opp_cue_pos, opp_target_pos, opp_pocket_pos)
            opp_shot.cue.set_state(**opp_action)
            
            if simulate_with_timeout(opp_shot, timeout=2):
                opp_reward = analyze_shot_for_reward(opp_shot, shot.balls, opponent_targets)
                if opp_reward > 0:  # Opponent can score
                    foul_penalty -= 20  # Penalize our defense if opponent can still score
        
        # Total reward: bonus for no opponent shot + penalty for opponent score + positioning bonus + hide bonus + foul penalty
        reward = (200 if not opponent_choices else 0) - opp_best_score + near_our_ball_bonus + hide_bonus + foul_penalty
        
        return reward
    
    def _find_best_safety_shot(self, balls, my_targets, table):
        """
        Generate and evaluate multiple defensive strategies using Optuna optimization.
        Returns the action that maximizes defensive reward.
        """
        print("[Agent] Switching to defensive play - optimizing safety shot with Optuna")
        
        def defensive_objective(trial):
            """Objective function for defensive shot optimization."""
            # Suggest parameters
            V0 = trial.suggest_float('V0', 0.5, 2.0)
            phi = trial.suggest_float('phi', 0, 360)
            theta = trial.suggest_float('theta', 0, 10)
            a = trial.suggest_float('a', -0.2, 0.2)
            b = trial.suggest_float('b', -0.2, 0.2)
            
            action = {
                'V0': V0,
                'phi': phi,
                'theta': theta,
                'a': a,
                'b': b
            }
            
            # Evaluate defensive action
            reward = self._evaluate_defensive_action(action, balls, my_targets, table)
            return reward
        
        # Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(defensive_objective, n_trials=50, timeout=15)  # Increased trials and timeout
        
        # Get best action
        best_params = study.best_params
        best_action = {
            'V0': best_params['V0'],
            'phi': best_params['phi'],
            'theta': best_params['theta'],
            'a': best_params['a'],
            'b': best_params['b']
        }
        
        # Verify fatal rate of the best action
        fatal_rate, _, _ = self._check_fatal_failure(best_action, balls, my_targets, table, num_trials=10, fatal_threshold=0.1)
        if fatal_rate > 0.1:
            print(f"[Agent] Best defensive action has high fatal rate ({fatal_rate:.1%}), falling back to safe action")
            return self._safe_action()
        
        print(f"[Agent] Defense optimized - best reward: {study.best_value:.2f}, fatal_rate: {fatal_rate:.1%}")
        
        return best_action
    
    
    # ============================================================================
    # Main Decision Function
    # ============================================================================
    def _is_opening_state(self, balls):
        """
        判断是否为开球局面：所有15颗球都在桌上，且除了白球，其他球基本聚在中间。
        """
        # 检查所有球都在桌上
        total_on_table = sum(1 for bid in balls if bid != 'cue' and balls[bid].state.s != 4)
        if total_on_table != 15:
            return False
        
        # 计算除了白球的其他球的位置
        positions = []
        for bid, ball in balls.items():
            if bid != 'cue' and ball.state.s != 4:
                positions.append(ball.state.rvw[0][:2])
        
        if len(positions) != 15:
            return False
        
        # 计算质心
        centroid = np.mean(positions, axis=0)
        
        # 计算位置方差（到质心的平均距离平方）
        variance = np.mean([np.linalg.norm(pos - centroid)**2 for pos in positions])
        
        # 如果方差小于阈值，则认为是开球（球聚在一起）
        threshold = 0.5  # 阈值可调整
        return variance < threshold

    def _generate_opening_actions(self, balls, my_targets, table):
        """
        为开球生成候选动作：重点瞄准己方球，增加连杆可能性。
        """
        actions = []
        cue_pos = balls['cue'].state.rvw[0][:2]

        # 只瞄准己方目标球
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]

        for tid in target_ids:
            obj_pos = balls[tid].state.rvw[0][:2]

            # 瞄准每个袋口
            for pocket in table.pockets.values():
                pocket_pos = pocket.center[:2]

                # 计算角度
                vec = np.array(pocket_pos) - np.array(obj_pos)
                dist = np.linalg.norm(vec)
                if dist == 0:
                    continue
                phi = np.degrees(np.arctan2(vec[1], vec[0])) % 360

                # 生成变种：力度从强到弱，增加连杆机会
                for V0 in [3.0, 5.0, 7.0]:
                    for a in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                        actions.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': 0.0,
                            'a': a,
                            'b': 0.0
                        })

        # 如果没有动作，添加随机
        if not actions:
            for _ in range(10):
                actions.append(self._random_action())

        # 限制数量
        random.shuffle(actions)
        return actions[:50]
    
    def _select_best_opening_action(self, candidates, balls, my_targets, table):
        """
        评估候选动作，选择最好的。
        """
        best_action = None
        best_score = 40
        best_candidates = []
        for action in candidates:
            score = self._evaluate_action(action, 1, balls, my_targets, table, threshold=20, enable_noise=True)
            if score > 40:
                best_candidates.append((action, score))
        
        best_candidates.sort(key=lambda x: x[1], reverse=True)
        for action, score in best_candidates[:5]:
            # 进一步验证
            fatal_rate, _, verified_score = self._check_fatal_failure(action, balls, my_targets, table, num_trials=2)
            if fatal_rate < 0.1 and verified_score > best_score:
                best_score = verified_score
                best_action = action

        if best_action:
            print(f"[NewAgent] Opening shot selected with estimated score {best_score:.2f}")
            return best_action
        else:
            return None
        
    def decision(self, balls=None, my_targets=None, table=None):
        if not all([balls, my_targets, table]):
            self.last_strategy = "Safe"
            return self._safe_action()
        
        try:
            print("[NewAgent] Starting decision process...")
            is_opening = self._is_opening_state(balls)
            # 检查是否为开球局面
            if is_opening:
                print("[NewAgent] Detected opening state, using special opening strategy.")
                candidates = self._generate_opening_actions(balls, my_targets, table)
                action = self._select_best_opening_action(candidates, balls, my_targets, table)
                if action:
                    self.last_strategy = "Opening"
                    return action
            
            # Determine game state
            remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
            if not remaining and "8" in balls and balls["8"].state.s != 4:
                my_targets = ['8']
                print("[NewAgent] Switching to black eight")
            is_black_eight = (my_targets == ['8'])

            # Define thresholds
            GEO_THRESHOLD = 120 if is_black_eight else 60
            SCORE_THRESHOLD = 120 if is_black_eight else 55
            SAFE_FATAL_THRESHOLD = 0.05
            PRE_TRIALS = 5

            # Layer 1: Select Best Targets
            if is_opening:
                top_choices = self._choose_top_targets(balls, my_targets, table, num_choices=3)
            
            else:
                top_choices = self._choose_top_targets(balls, my_targets, table, num_choices=30)
            
            if not top_choices:
                print("[NewAgent] No targets found, switching to defense...")
                self.last_strategy = "Safety"
                return self._find_best_safety_shot(balls, my_targets, table)

            # Layer 2 & 3: Evaluate Geometric shots and Optimize
            all_evaluated = []
            cue_pos = balls['cue'].state.rvw[0]
            geo_evaluated = []
            print(f"[NewAgent] Evaluating top {len(top_choices)} geometric shot candidates")
            for (i, choice) in enumerate(top_choices):
                print(f"\r[NewAgent] Processing shot {i+1}/{len(top_choices)}...", end="", flush=True)
                target_pos = balls[choice['target_id']].state.rvw[0]
                pocket_pos = table.pockets[choice['pocket_id']].center
                if choice['type'] == 'bank':
                    base_action = self._geo_bank_shot(cue_pos, target_pos, pocket_pos, choice['cushion_id'])
                elif choice['type'] == 'curve':
                    base_action = self._geo_shot(cue_pos, target_pos, pocket_pos)
                    base_action['a'] = self._calc_curve_a(choice['cut_angle'])
                else:  # direct
                    base_action = self._geo_shot(cue_pos, target_pos, pocket_pos)
                
                geo_score = self._evaluate_action(base_action, 1, balls, my_targets, table, 30, False)
                if geo_score > -100 or (is_black_eight and geo_score > -300):
                    geo_evaluated.append((base_action, geo_score, choice))

            # Sort by geometric score
            geo_evaluated.sort(key=lambda x: x[1], reverse=True)
            print(f"\n[NewAgent] Evaluating top {len(geo_evaluated)} geometric shots, best score: {geo_evaluated[0][1]:.2f}" if geo_evaluated else f"\n[NewAgent] No valid geometric shots found")
            for i, geo_choice in enumerate(geo_evaluated[:8]):
                print(f"\r[NewAgent] Processing shot {i+1}/{len(geo_evaluated[:8])}...", end="", flush=True)
                base_action, geo_score, choice = geo_choice
                action_to_check, score_to_check = base_action, geo_score
                if geo_score > GEO_THRESHOLD:
                    # Directly proceed to safety check
                    fatal_rate, _, verified_score = self._check_fatal_failure(action_to_check, balls, my_targets, table, num_trials=3)
                    
                    shot_type_str = f"{choice['type']} {choice['target_id']}->{choice['pocket_id']}"
                    if verified_score >= GEO_THRESHOLD and fatal_rate <= 0:
                        fatal_rate, _, verified_score, msg = self._check_fatal_failure_early_stop(action_to_check, balls, my_targets, table, num_trials=15)
                        if msg != "Completed":
                            print(f"\r[NewAgent] ! Early stop during fatal check: {msg}")
                        elif fatal_rate <= SAFE_FATAL_THRESHOLD and verified_score >= GEO_THRESHOLD - 10:
                                print(f"\n[NewAgent] ✓ Found acceptable geometric action: {shot_type_str}: score={verified_score:.2f}, fatal_rate={fatal_rate:.1%}")
                                self.last_strategy = "Geo"
                                return action_to_check

                # If geo score is not great, try to optimize
                opt_action, opt_score = self._cma_es_optimized(base_action, balls, my_targets, table, is_black_eight, is_opening)
                if opt_action and opt_score > SCORE_THRESHOLD - 20:
                    action_to_check, score_to_check = opt_action, opt_score
                    fatal_rate, _, verified_score = self._check_fatal_failure(action_to_check, balls, my_targets, table, num_trials=20)
                    shot_type_str = f"{choice['type']} {choice['target_id']}->{choice['pocket_id']}"
                    if verified_score >= SCORE_THRESHOLD and fatal_rate <= SAFE_FATAL_THRESHOLD:
                        print(f"\n[NewAgent] ✓ Found acceptable optimized action: {shot_type_str}: score={verified_score:.2f}, fatal_rate={fatal_rate:.1%}")
                        self.last_strategy = "Opt"
                        return action_to_check
                    elif verified_score > 0 and fatal_rate <= SAFE_FATAL_THRESHOLD:
                        all_evaluated.append((action_to_check, verified_score, shot_type_str, fatal_rate))
                        continue

            print()  # New line after progress
            
            # Layer 4: Fallback Selection (Offensive)
            print(f"[NewAgent] Fallback evaluation among {len(all_evaluated)} safe candidates")
            safe_candidates = [cand for cand in all_evaluated if cand[3] <= SAFE_FATAL_THRESHOLD]
            
            if safe_candidates:
                best_action, best_score, best_type, _ = max(safe_candidates, key=lambda x: x[1])
                
                if best_score > 30: # Only take the shot if it's reasonably good
                    print(f"[NewAgent] Using best fallback option: {best_type} (score={best_score:.2f})")
                    self.last_strategy = "Fallback"
                    return best_action

            print("[NewAgent] Final fallback: Switching to defensive strategy...")
            # Final Fallback: Switch to Defensive Strategy
            self.last_strategy = "Safety"
            return self._find_best_safety_shot(balls, my_targets, table)
            
        except Exception as e:
            print(f"[NewAgent] Decision failed: {e}")
            import traceback
            traceback.print_exc()
            self.last_strategy = "Safe"
            return self._safe_action()
    
    
