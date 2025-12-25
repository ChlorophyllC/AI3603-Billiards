import re
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics
import math


class GameLog:
    """单局游戏日志"""
    def __init__(self):
        self.game_number = None
        self.player_a_agent = None
        self.player_b_agent = None
        self.player_a_target = None
        self.shots = []  # [[shot_number, player, timestamp, pocketed_balls], ...]
        self.winner = None
        self.duration = None
        self.opponent_ball_pocketed = []  # 对方球入袋事件
        self.agent_a_remaining = None
        self.agent_b_remaining = None
        self.strategies = {}  # 策略记录
        
    def get_agent_for_player(self, player):
        """根据Player获取对应的Agent名称"""
        if player == 'A':
            return self.player_a_agent
        elif player == 'B':
            return self.player_b_agent
        return None
    
    def get_game_duration(self):
        """计算本局从第一次击球到最后一次击球的时间"""
        if len(self.shots) < 2:
            return 0
        first_shot_time = self.shots[0][2]
        last_shot_time = self.shots[-1][2]
        return (last_shot_time - first_shot_time).total_seconds()


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self):
        self.agent_a_name = None
        self.agent_b_name = None
        self.games = []
        
    def parse_log_file(self, file_path):
        """解析单个日志文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_game = None
        last_timestamp = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 提取时间戳
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)', line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                if '.' in timestamp_str:
                    last_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    last_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                content = line[len(timestamp_str):].strip(' -')
            else:
                content = line

            # 解析Agent名称（全局信息）- 只在开始评估时解析，避免匹配到战绩统计行
            if '开始对战评估' in content or ('Agent A:' in content and 'Agent B:' in content and '累计战绩' not in content and '局结果' not in content):
                match = re.search(r'Agent A:\s*(\w+),\s*Agent B:\s*(\w+)', content)
                if match and not self.agent_a_name:
                    self.agent_a_name = match.group(1)
                    self.agent_b_name = match.group(2)
                    for game in self.games:
                        if not game.strategies:
                            game.strategies = {self.agent_a_name: [], self.agent_b_name: []}
                    if current_game and not current_game.strategies:
                        current_game.strategies = {self.agent_a_name: [], self.agent_b_name: []}

            # 新局开始
            if '局比赛开始' in content:
                match = re.search(r'第\s*(\d+)\s*局比赛开始', content)
                if match:
                    current_game = GameLog()
                    current_game.game_number = int(match.group(1))
                    if self.agent_a_name and self.agent_b_name:
                        current_game.strategies = {self.agent_a_name: [], self.agent_b_name: []}
                    else:
                        current_game.strategies = {}

            # 本局Player和Agent映射
            if current_game and '本局 Player A:' in content:
                match = re.search(r'Player A:\s*(\w+),\s*目标球型:\s*(\w+)', content)
                if match:
                    current_game.player_a_agent = match.group(1)
                    current_game.player_a_target = match.group(2)
                    if current_game.player_a_agent == self.agent_a_name:
                        current_game.player_b_agent = self.agent_b_name
                    else:
                        current_game.player_b_agent = self.agent_a_name

            # 击球记录
            if current_game and re.match(r'\[第\d+次击球\] player: [AB]', content):
                match = re.match(r'\[第(\d+)次击球\] player: ([AB])', content)
                if match and last_timestamp:
                    shot_number = int(match.group(1))
                    player = match.group(2)
                    current_game.shots.append([shot_number, player, last_timestamp, [], []])  # [shot_number, player, timestamp, my_pocketed, opp_pocketed]

            # 策略记录
            if current_game and 'NewAgent 策略:' in content:
                match = re.search(r'NewAgent 策略:\s*(\w+)', content)
                if match:
                    strategy = match.group(1)
                    if current_game.player_a_agent == 'NewAgent':
                        current_game.strategies.setdefault('NewAgent', []).append(strategy)
                    elif current_game.player_b_agent == 'NewAgent':
                        current_game.strategies.setdefault('NewAgent', []).append(strategy)

            # 进球记录
            if current_game and '本杆进球' in content:
                match = re.search(r'本杆进球 \((\w+)\) - 我方: (\[.*?\]), 对方: (\[.*?\])', content)
                if match:
                    agent = match.group(1)
                    try:
                        my_balls = eval(match.group(2))
                        opp_balls = eval(match.group(3))
                    except Exception:
                        my_balls = []
                        opp_balls = []
                    if current_game.shots:
                        current_game.shots[-1][3] = my_balls
                        current_game.shots[-1][4] = opp_balls

            # 结束时剩余球数
            if current_game and '结束时剩余球数' in content:
                match = re.search(r'Agent A:\s*(\d+),\s*Agent B:\s*(\d+)', content)
                if match:
                    current_game.agent_a_remaining = int(match.group(1))
                    current_game.agent_b_remaining = int(match.group(2))

            # 局结果 - 这里的Agent A/B指的是全局Agent
            if current_game and '局结果' in content:
                match = re.search(r'Agent ([AB])\s*获胜', content)
                if match:
                    winner_label = match.group(1)
                    if winner_label == 'A':
                        current_game.winner = self.agent_a_name
                    else:
                        current_game.winner = self.agent_b_name

            # 本局耗时
            if current_game and '本局耗时' in content:
                match = re.search(r'本局耗时:\s*([\d.]+)s', content)
                if match:
                    current_game.duration = float(match.group(1))
                    self.games.append(current_game)
                    current_game = None
    
    def _analyze_strategy_ratio(self):
        """分析 NewAgent 策略占比"""
        print("\n" + "-"*80)
        print("11. NewAgent 策略占比分析")
        print("-"*80)
        
        strategy_counts = defaultdict(int)
        total_decisions = 0
        
        for game in self.games:
            for agent, strategies in game.strategies.items():
                if agent == 'NewAgent':
                    for strategy in strategies:
                        strategy_counts[strategy] += 1
                        total_decisions += 1
        
        if total_decisions == 0:
            print("未找到 NewAgent 的策略记录")
            return
        
        print(f"NewAgent 总决策次数: {total_decisions}")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            ratio = count / total_decisions * 100
            print(f"  {strategy}: {count}次 ({ratio:.2f}%)")

    def _analyze_mistake_strategies(self):
        """分析NewAgent失误时的策略分布"""
        print("\n" + "-"*80)
        print("12. NewAgent 失误策略分析")
        print("-"*80)
        
        mistake_strategies = defaultdict(int)
        total_mistakes = 0
        
        for game in self.games:
            if not game.shots or not game.winner:
                continue
            
            # 最后一次击球的agent
            last_shot_player = game.shots[-1][1]
            last_shot_agent = game.get_agent_for_player(last_shot_player)
            
            # 如果最后击球的是NewAgent，且NewAgent输了
            if last_shot_agent == 'NewAgent' and game.winner != 'NewAgent':
                # 获取最后一次策略
                if 'NewAgent' in game.strategies and game.strategies['NewAgent']:
                    last_strategy = game.strategies['NewAgent'][-1]
                    mistake_strategies[last_strategy] += 1
                    total_mistakes += 1
        
        if total_mistakes == 0:
            print("未找到NewAgent的失误记录")
            return
        
        print(f"NewAgent 失误总次数: {total_mistakes}")
        for strategy, count in sorted(mistake_strategies.items(), key=lambda x: x[1], reverse=True):
            ratio = count / total_mistakes * 100
            print(f"  {strategy}: {count}次 ({ratio:.2f}%)")

    def _analyze_strategy_success_rate(self):
        """分析NewAgent策略成功率（只用连击推理，策略与击球严格一一对应）"""
        print("\n" + "-"*80)
        print("13. NewAgent 策略成功率分析")
        print("-"*80)

        strategy_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'defense_total': 0,
            'defense_success': 0,
            'defense_fail': 0
        })

        for game in self.games:
            if 'NewAgent' not in game.strategies:
                continue
            # 只统计NewAgent实际击球的索引和策略
            shot_indices = [idx for idx, shot in enumerate(game.shots) if game.get_agent_for_player(shot[1]) == 'NewAgent']
            strategies = game.strategies['NewAgent']
            n = min(len(shot_indices), len(strategies))
            for j in range(n):
                i = shot_indices[j]
                strategy = strategies[j]
                strategy_stats[strategy]['total'] += 1
                is_combo = False
                # 连击推理
                if i + 1 < len(game.shots):
                    next_player = game.shots[i + 1][1]
                    next_agent = game.get_agent_for_player(next_player)
                    if next_agent == 'NewAgent':
                        strategy_stats[strategy]['success'] += 1
                        is_combo = True
                else:
                    if game.get_agent_for_player(game.shots[i][1]) == game.winner:
                        strategy_stats[strategy]['success'] += 1
                        is_combo = True
                # Safety防守成功率
                if strategy == 'Safety':
                    strategy_stats[strategy]['defense_total'] += 1
                    if is_combo:
                        strategy_stats[strategy]['defense_success'] += 1
                    else:
                        if i + 1 < len(game.shots):
                            next_player = game.shots[i + 1][1]
                            next_agent = game.get_agent_for_player(next_player)
                            if i + 2 < len(game.shots):
                                next2_player = game.shots[i + 2][1]
                                next2_agent = game.get_agent_for_player(next2_player)
                                if next2_agent == next_agent:
                                    strategy_stats[strategy]['defense_fail'] += 1
                                else:
                                    strategy_stats[strategy]['defense_success'] += 1
                            else:
                                if next_agent == game.winner:
                                    strategy_stats[strategy]['defense_fail'] += 1
                                else:
                                    strategy_stats[strategy]['defense_success'] += 1

        if not strategy_stats:
            print("未找到NewAgent的策略记录")
            return

        for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[0]):
            print(f"\n{strategy}:")
            total = stats['total']
            success_rate = stats['success'] / total * 100 if total > 0 else 0
            print(f"  总次数: {total}")
            print(f"  击球成功率: {success_rate:.2f}% ({stats['success']}/{total})")
            if strategy == 'Safety':
                d_total = stats['defense_total']
                d_succ = stats['defense_success']
                d_fail = stats['defense_fail']
                defense_success_rate = d_succ / d_total * 100 if d_total > 0 else 0
                defense_fail_rate = d_fail / d_total * 100 if d_total > 0 else 0
                print(f"  防守成功率: {defense_success_rate:.2f}% ({d_succ}/{d_total})")
                print(f"  防守失败率: {defense_fail_rate:.2f}% ({d_fail}/{d_total})")

    def analyze(self):
        """执行分析"""
        if not self.games:
            print("没有找到有效的游戏数据")
            return
        
        print("\n" + "="*80)
        print(f"日志分析报告")
        print("="*80)
        print(f"\n总局数: {len(self.games)}")
        print(f"Agent A: {self.agent_a_name}")
        print(f"Agent B: {self.agent_b_name}")
        
        # 1. 胜率统计
        self._analyze_win_rate()
        
        # 2. 平均决策时间
        self._analyze_decision_time()
        
        # 3. 每局平均时间
        self._analyze_game_duration()
        
        # 4. 获胜原因分析
        self._analyze_win_reason()
        
        # 5. 平均轮次
        self._analyze_average_rounds()
        
        # 6. 决策占比
        self._analyze_decision_ratio()
        
        # 7. 连续击球分析
        self._analyze_consecutive_shots()
        
        # 8. 开局表现
        self._analyze_first_move_advantage()
        
        # 9. 胜率置信区间
        self._analyze_confidence_interval()
        
        # 10. 决策时间分布
        self._analyze_decision_time_distribution()
        
        # 11. NewAgent 策略占比
        self._analyze_strategy_ratio()
        
        # 12. NewAgent 失误策略分析
        self._analyze_mistake_strategies()
        
        # 13. NewAgent 策略成功率分析
        self._analyze_strategy_success_rate()

    def _analyze_win_rate(self):
        """分析胜率"""
        print("\n" + "-"*80)
        print("1. 胜率统计")
        print("-"*80)
        
        wins = {self.agent_a_name: 0, self.agent_b_name: 0, "平局": 0}
        for game in self.games:
            if game.winner in wins:
                wins[game.winner] += 1
            elif game.winner is None:  # 平局情况
                wins[self.agent_a_name] += 0.5
                wins[self.agent_b_name] += 0.5
                wins["平局"] += 1
        
        total = len(self.games)
        for agent, win_count in wins.items():
            win_rate = win_count / total * 100
            print(f"{agent}: {win_count}胜 / {total}局 = {win_rate:.2f}%")
    
    def _analyze_decision_time(self):
        """分析平均决策时间"""
        print("\n" + "-"*80)
        print("2. 平均决策时间")
        print("-"*80)
        
        agent_times = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            for i in range(0, len(game.shots)-1):
                prev_shot = game.shots[i]
                curr_shot = game.shots[i+1]
                
                # 计算决策时间（当前击球时间 - 上一次击球时间）
                time_diff = (curr_shot[2] - prev_shot[2]).total_seconds()
                
                # 获取当前击球的Agent
                agent = game.get_agent_for_player(prev_shot[1])
                if agent in agent_times:
                    agent_times[agent].append(time_diff)
        
        for agent, times in agent_times.items():
            if times:
                avg_time = statistics.mean(times)
                print(f"{agent}: 平均 {avg_time:.2f}秒 (共{len(times)}次决策)")
            else:
                print(f"{agent}: 无数据")
    
    def _analyze_win_reason(self):
        """分析获胜原因"""
        print("\n" + "-"*80)
        print("4. 获胜原因分析")
        print("-"*80)
        
        win_reasons = {
            self.agent_a_name: {'正常打入黑8': 0, '对方失误': 0, 'opponent_remaining': []},
            self.agent_b_name: {'正常打入黑8': 0, '对方失误': 0, 'opponent_remaining': []}
        }
        
        for game in self.games:
            if not game.shots or not game.winner:
                continue
            
            # 最后一次击球的Player
            last_shot_player = game.shots[-1][1]
            last_shot_agent = game.get_agent_for_player(last_shot_player)
            
            if last_shot_agent == game.winner:
                win_reasons[game.winner]['正常打入黑8'] += 1
                # 记录对方剩余球数
                if game.winner == self.agent_a_name and game.agent_b_remaining is not None:
                    win_reasons[game.winner]['opponent_remaining'].append(game.agent_b_remaining)
                elif game.winner == self.agent_b_name and game.agent_a_remaining is not None:
                    win_reasons[game.winner]['opponent_remaining'].append(game.agent_a_remaining)
            else:
                win_reasons[game.winner]['对方失误'] += 1
        
        for agent, reasons in win_reasons.items():
            total_wins = reasons['正常打入黑8'] + reasons['对方失误']
            if total_wins > 0:
                normal_rate = reasons['正常打入黑8'] / total_wins * 100
                mistake_rate = reasons['对方失误'] / total_wins * 100
                print(f"\n{agent} (共{total_wins}胜):")
                print(f"  正常打入黑8: {reasons['正常打入黑8']}次 ({normal_rate:.1f}%)")
                print(f"  对方失误: {reasons['对方失误']}次 ({mistake_rate:.1f}%)")
                if reasons['opponent_remaining']:
                    avg_remaining = statistics.mean(reasons['opponent_remaining'])
                    print(f"  正常打入黑8时对方平均剩余球数: {avg_remaining:.2f}球")
    
    def _analyze_average_rounds(self):
        """分析平均轮次"""
        print("\n" + "-"*80)
        print("5. 平均轮次统计")
        print("-"*80)
        
        total_shots = [len(game.shots) for game in self.games if game.shots]
        if total_shots:
            avg_shots = statistics.mean(total_shots)
            min_shots = min(total_shots)
            max_shots = max(total_shots)
            median_shots = statistics.median(total_shots)
            
            print(f"平均每局击球次数: {avg_shots:.2f}次")
            print(f"中位数: {median_shots:.0f}次")
            print(f"最少: {min_shots}次, 最多: {max_shots}次")
    
    def _analyze_decision_ratio(self):
        """分析决策占比"""
        print("\n" + "-"*80)
        print("6. 决策占比分析")
        print("-"*80)
        
        agent_shot_counts = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            shot_count = {self.agent_a_name: 0, self.agent_b_name: 0}
            
            for shot in game.shots:
                agent = game.get_agent_for_player(shot[1])
                if agent in shot_count:
                    shot_count[agent] += 1
            
            total = sum(shot_count.values())
            if total > 0:
                for agent in agent_shot_counts:
                    ratio = shot_count[agent] / total
                    agent_shot_counts[agent].append(ratio)
        
        for agent, ratios in agent_shot_counts.items():
            if ratios:
                avg_ratio = statistics.mean(ratios) * 100
                print(f"{agent}: 平均每局决策占比 {avg_ratio:.2f}%")
    
    def _analyze_consecutive_shots(self):
        """分析连续击球"""
        print("\n" + "-"*80)
        print("7. 连续击球分析")
        print("-"*80)
        
        agent_consecutive = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            if len(game.shots) < 2:
                continue
            
            current_agent = game.get_agent_for_player(game.shots[0][1])
            consecutive_count = 1
            
            for i in range(1, len(game.shots)):
                agent = game.get_agent_for_player(game.shots[i][1])
                
                if agent == current_agent:
                    consecutive_count += 1
                else:
                    # 记录连续击球次数
                    if current_agent in agent_consecutive and consecutive_count > 1:
                        agent_consecutive[current_agent].append(consecutive_count)
                    current_agent = agent
                    consecutive_count = 1
            
            # 记录最后一段连续击球
            if current_agent in agent_consecutive and consecutive_count > 1:
                agent_consecutive[current_agent].append(consecutive_count)
        
        for agent, consecutives in agent_consecutive.items():
            if consecutives:
                avg_consecutive = statistics.mean(consecutives)
                max_consecutive = max(consecutives)
                print(f"{agent}: 平均连续击球 {avg_consecutive:.2f}次, 最高 {max_consecutive}次 (共{len(consecutives)}次连击)")
            else:
                print(f"{agent}: 无连续击球记录")
    
    def _analyze_first_move_advantage(self):
        """分析开局优势"""
        print("\n" + "-"*80)
        print("8. 开局表现分析")
        print("-"*80)
        
        first_move_wins = {self.agent_a_name: 0, self.agent_b_name: 0}
        first_move_total = {self.agent_a_name: 0, self.agent_b_name: 0}
        
        for game in self.games:
            if not game.shots or not game.winner:
                continue
            
            # 第一次击球的Agent
            first_agent = game.get_agent_for_player(game.shots[0][1])
            
            if first_agent in first_move_total:
                first_move_total[first_agent] += 1
                if first_agent == game.winner:
                    first_move_wins[first_agent] += 1
        
        for agent in [self.agent_a_name, self.agent_b_name]:
            total = first_move_total[agent]
            wins = first_move_wins[agent]
            if total > 0:
                win_rate = wins / total * 100
                print(f"{agent} 先手: {wins}胜/{total}局 = {win_rate:.2f}%")
    
    def _analyze_confidence_interval(self):
        """分析胜率置信区间（95%置信区间）"""
        print("\n" + "-"*80)
        print("9. 胜率置信区间 (95%)")
        print("-"*80)
        
        wins = {self.agent_a_name: 0, self.agent_b_name: 0}
        for game in self.games:
            if game.winner in wins:
                wins[game.winner] += 1
        
        n = len(self.games)
        z = 1.96  # 95%置信区间的z值
        
        for agent, win_count in wins.items():
            p = win_count / n
            se = math.sqrt(p * (1 - p) / n)
            ci_lower = max(0, p - z * se) * 100
            ci_upper = min(1, p + z * se) * 100
            
            print(f"{agent}: {p*100:.2f}% (95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%])")
    
    def _analyze_decision_time_distribution(self):
        """分析决策时间分布"""
        print("\n" + "-"*80)
        print("10. 决策时间分布")
        print("-"*80)
        
        agent_times = {self.agent_a_name: [], self.agent_b_name: []}
        
        for game in self.games:
            for i in range(0, len(game.shots)-1):
                prev_shot = game.shots[i]
                curr_shot = game.shots[i+1]
                
                # 计算决策时间（当前击球时间 - 上一次击球时间）
                time_diff = (curr_shot[2] - prev_shot[2]).total_seconds()
                
                # 获取当前击球的Agent
                agent = game.get_agent_for_player(prev_shot[1])
                if agent in agent_times:
                    agent_times[agent].append(time_diff)
        
        for agent, times in agent_times.items():
            if times:
                avg = statistics.mean(times)
                median = statistics.median(times)
                stdev = statistics.stdev(times) if len(times) > 1 else 0
                min_time = min(times)
                max_time = max(times)
                
                print(f"\n{agent}:")
                print(f"  平均值: {avg:.2f}秒")
                print(f"  中位数: {median:.2f}秒")
                print(f"  标准差: {stdev:.2f}秒")
                print(f"  范围: [{min_time:.2f}秒, {max_time:.2f}秒]")
    
    def _analyze_game_duration(self):
        """分析每局平均时间"""
        print("\n" + "-"*80)
        print("3. 每局平均时间统计")
        print("-"*80)
        
        # 使用击球时间戳计算的每局时间
        game_durations_from_shots = []
        # 使用日志中记录的duration
        game_durations_from_log = []
        
        for game in self.games:
            # 方法1: 从击球时间戳计算
            shot_duration = game.get_game_duration()
            if shot_duration > 0:
                game_durations_from_shots.append(shot_duration)
            
            # 方法2: 从日志中的duration（如果有）
            if game.duration:
                game_durations_from_log.append(game.duration)
        
        print("\n基于击球时间戳计算:")
        if game_durations_from_shots:
            avg_duration = statistics.mean(game_durations_from_shots)
            median_duration = statistics.median(game_durations_from_shots)
            min_duration = min(game_durations_from_shots)
            max_duration = max(game_durations_from_shots)
            stdev = statistics.stdev(game_durations_from_shots) if len(game_durations_from_shots) > 1 else 0
            
            print(f"  平均耗时: {avg_duration:.2f}秒")
            print(f"  中位数: {median_duration:.2f}秒")
            print(f"  标准差: {stdev:.2f}秒")
            print(f"  范围: [{min_duration:.2f}秒, {max_duration:.2f}秒]")
            print(f"  样本数: {len(game_durations_from_shots)}局")
        else:
            print("  无有效数据")
        
        print("\n基于日志记录的耗时:")
        if game_durations_from_log:
            avg_duration_log = statistics.mean(game_durations_from_log)
            median_duration_log = statistics.median(game_durations_from_log)
            min_duration_log = min(game_durations_from_log)
            max_duration_log = max(game_durations_from_log)
            stdev_log = statistics.stdev(game_durations_from_log) if len(game_durations_from_log) > 1 else 0
            
            print(f"  平均耗时: {avg_duration_log:.2f}秒")
            print(f"  中位数: {median_duration_log:.2f}秒")
            print(f"  标准差: {stdev_log:.2f}秒")
            print(f"  范围: [{min_duration_log:.2f}秒, {max_duration_log:.2f}秒]")
            print(f"  样本数: {len(game_durations_from_log)}局")
        else:
            print("  无有效数据")

def main(log_path):
    """主函数"""
    
    import sys
    from io import StringIO
    
    analyzer = LogAnalyzer()
    
    if os.path.isfile(log_path):
        print(f"正在分析文件: {log_path}")
        analyzer.parse_log_file(log_path)
        output_path = log_path.rsplit('.', 1)[0] + "_analysis.txt"
    elif os.path.isdir(log_path):
        log_files = list(Path(log_path).glob("*.log"))
        if not log_files:
            print(f"在 {log_path} 中没有找到 .log 文件")
            return
        
        print(f"找到 {len(log_files)} 个日志文件")
        for log_file in log_files:
            print(f"  - {log_file.name}")
            analyzer.parse_log_file(str(log_file))
        output_path = os.path.join(log_path, "combined_analysis.txt")
    else:
        print(f"错误: {log_path} 不是有效的文件或文件夹")
        return
    
    # 捕获标准输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    analyzer.analyze()
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)
    
    # 获取输出内容
    output_content = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # 打印到控制台
    print(output_content)
    
    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"\n分析结果已保存到: {output_path}")


if __name__ == "__main__":
    # 直接在这里修改要分析的路径
    log_path = "logs_3"  # 可以是文件路径或文件夹路径
    # log_path = "logs/evaluate_20251212_134237.log"  # 单个文件示例
    
    # main(log_path)
    # main("logs_5")

    main("logs_basic")
    main("logs_pro")
        
    
