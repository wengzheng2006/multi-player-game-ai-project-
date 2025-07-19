"""
贪吃蛇专用AI智能体
"""

import random
import numpy as np
import time
from heapq import heappush, heappop
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent

class SnakeAI(BaseAgent):
    """改进的贪吃蛇AI智能体"""
    
    def __init__(self, name="SnakeAI", player_id=1):
        super().__init__(name, player_id)
        self.path_cache = {}  # 路径缓存
        self.threat_assessment = {}  # 威胁评估缓存
        self.last_positions = []  # 记录历史位置
        self.opponent_predictions = {}  # 对手预测
        self.max_thinking_time = 0.1  # 最大思考时间100ms
        self.game_history = []  # 游戏历史
        self.adaptive_weights = {  # 自适应权重
            'food': 1.0,
            'space': 1.0,
            'threat': 1.0,
            'diversity': 1.0,
            'center': 1.0,
            'safety': 1.0
        }
        self.last_action = None  # 记录上次动作
        self.consecutive_same_actions = 0  # 连续相同动作计数
    
    def get_action(self, observation, env):
        """获取动作（改进版）"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        # 获取当前蛇的信息
        game = env.game
        if self.player_id == 1:
            snake = game.snake1
            opponent_snake = game.snake2
            current_direction = game.direction1
        else:
            snake = game.snake2
            opponent_snake = game.snake1
            current_direction = game.direction2
        
        if not snake:
            return random.choice(valid_actions)
        
        head = snake[0]
        
        # 检查时间限制
        if time.time() - start_time > self.max_thinking_time:
            return random.choice(valid_actions)
        
        # 更新历史位置和游戏状态
        self._update_position_history(head)
        self._update_game_history(game)
        
        # 自适应权重调整
        self._adjust_weights(game, snake, opponent_snake)
        
        # 死循环检测：如果最近8步有6步在同一个点，强制随机
        if len(self.last_positions) >= 8:
            recent = self.last_positions[-8:]
            if max([recent.count(pos) for pos in set(recent)]) >= 6:
                print("[AI] 检测到死循环，强制随机行动")
                return random.choice(valid_actions)
        
        # 预测对手行为（改进版）
        opponent_prediction = self._predict_opponent_moves_advanced(opponent_snake, game)
        
        # 评估所有动作
        action_scores = []
        for action in valid_actions:
            # 检查时间限制
            if time.time() - start_time > self.max_thinking_time:
                break
                
            score = self._evaluate_action_advanced(action, head, game, snake, opponent_snake, opponent_prediction)
            action_scores.append((score, action))
        
        # 只保留安全动作
        safe_actions = [action for score, action in action_scores if self._is_safe_action(action, head, game)]
        if safe_actions:
            # 只在安全动作中选分数最高的
            safe_action_scores = [(score, action) for score, action in action_scores if action in safe_actions]
            safe_action_scores.sort(reverse=True)
            best_action = safe_action_scores[0][1]
        else:
            # 没有安全动作，只能选一个（理论上此时必死）
            action_scores.sort(reverse=True)
            best_action = action_scores[0][1]
        
        # 更新连续动作计数
        if best_action == self.last_action:
            self.consecutive_same_actions += 1
        else:
            self.consecutive_same_actions = 0
        
        self.last_action = best_action
        return best_action
    
    def _update_game_history(self, game):
        """更新游戏历史"""
        game_state = {
            'snake1_length': len(game.snake1) if game.snake1 else 0,
            'snake2_length': len(game.snake2) if game.snake2 else 0,
            'food_count': len(game.foods) if game.foods else 0,
            'board_size': game.board_size
        }
        self.game_history.append(game_state)
        
        # 只保留最近20个状态
        if len(self.game_history) > 20:
            self.game_history.pop(0)
    
    def _adjust_weights(self, game, snake, opponent_snake):
        """自适应权重调整"""
        if len(self.game_history) < 3:
            return
        
        # 根据游戏状态调整权重
        current_state = self.game_history[-1]
        previous_state = self.game_history[-2] if len(self.game_history) > 1 else current_state
        
        # 如果蛇长度增加，增加食物权重
        if len(snake) > previous_state.get('snake1_length' if self.player_id == 1 else 'snake2_length', 0):
            self.adaptive_weights['food'] = min(2.0, self.adaptive_weights['food'] * 1.1)
        
        # 如果对手蛇长度增加，增加威胁权重
        opponent_length = len(opponent_snake) if opponent_snake else 0
        if opponent_length > previous_state.get('snake2_length' if self.player_id == 1 else 'snake1_length', 0):
            self.adaptive_weights['threat'] = min(2.0, self.adaptive_weights['threat'] * 1.2)
        
        # 如果连续相同动作过多，增加多样性权重
        if self.consecutive_same_actions > 3:
            self.adaptive_weights['diversity'] = min(2.0, self.adaptive_weights['diversity'] * 1.3)
        
        # 如果蛇长度很长，增加空间权重
        if len(snake) > game.board_size // 2:
            self.adaptive_weights['space'] = min(2.0, self.adaptive_weights['space'] * 1.1)
    
    def _predict_opponent_moves_advanced(self, opponent_snake, game):
        """高级对手预测"""
        if not opponent_snake:
            return []
        
        opponent_head = opponent_snake[0]
        predictions = []
        
        # 获取对手当前方向
        if self.player_id == 1:
            current_direction = game.direction2
        else:
            current_direction = game.direction1
        
        # 预测对手可能的移动方向
        possible_directions = []
        
        # 1. 当前方向
        new_pos = (opponent_head[0] + current_direction[0], opponent_head[1] + current_direction[1])
        if self._is_position_safe(new_pos, game):
            possible_directions.append(current_direction)
        
        # 2. 其他安全方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (dx, dy) != current_direction:
                new_pos = (opponent_head[0] + dx, opponent_head[1] + dy)
                if self._is_position_safe(new_pos, game):
                    possible_directions.append((dx, dy))
        
        # 根据对手历史行为预测最可能的方向
        if len(self.game_history) > 1:
            # 分析对手是否倾向于追逐食物
            if game.foods:
                nearest_food = self._find_nearest_food(opponent_head, game.foods)
                food_direction = self._get_direction_to_target(opponent_head, nearest_food)
                if food_direction in possible_directions:
                    predictions.append((opponent_head[0] + food_direction[0], opponent_head[1] + food_direction[1]))
        
        # 添加当前方向预测
        if current_direction in possible_directions:
            predictions.append((opponent_head[0] + current_direction[0], opponent_head[1] + current_direction[1]))
        
        return predictions
    
    def _get_direction_to_target(self, current, target):
        """获取到目标的方向"""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        if abs(dx) > abs(dy):
            return (1 if dx > 0 else -1, 0)
        else:
            return (0, 1 if dy > 0 else -1)
    
    def _evaluate_action_advanced(self, action, head, game, snake, opponent_snake, opponent_prediction):
        """高级动作评估"""
        new_head = (head[0] + action[0], head[1] + action[1])
        
        # 基础安全性检查
        if not self._is_position_safe(new_head, game):
            return -1000
        
        score = 0
        
        # 1. 边界感知（改进）
        boundary_penalty = self._calculate_boundary_penalty_advanced(new_head, game)
        score += boundary_penalty * self.adaptive_weights['safety']
        
        # 2. 食物吸引力（改进）
        if game.foods:
            nearest_food = self._find_nearest_food(new_head, game.foods)
            food_distance = abs(new_head[0] - nearest_food[0]) + abs(new_head[1] - nearest_food[1])
            # 动态食物权重
            if food_distance <= 3:
                score += (20 - food_distance * 3) * self.adaptive_weights['food']
            elif food_distance <= 6:
                score += (10 - food_distance) * self.adaptive_weights['food']
            else:
                score += max(0, 5 - food_distance // 2) * self.adaptive_weights['food']
        
        # 3. 空间评估（改进）
        available_space = self._calculate_available_space_advanced(new_head, game, snake)
        score += available_space * 4 * self.adaptive_weights['space']
        
        # 4. 威胁评估（改进）
        threat_level = self._assess_threat_level_advanced(new_head, opponent_snake, opponent_prediction, game)
        score -= threat_level * 12 * self.adaptive_weights['threat']
        
        # 5. 避免死胡同（改进）
        if self._is_dead_end_advanced(new_head, game, snake):
            score -= 60 * self.adaptive_weights['safety']
        
        # 6. 路径多样性（改进）
        path_diversity = self._calculate_path_diversity_advanced(new_head, game, snake)
        score += path_diversity * 3 * self.adaptive_weights['diversity']
        
        # 7. 中心偏好（改进）
        center_preference = self._calculate_center_preference_advanced(new_head, game)
        score += center_preference * self.adaptive_weights['center']
        
        # 8. 历史位置惩罚（改进）
        if hasattr(self, 'last_positions') and new_head in self.last_positions[-8:]:
            score -= 20 * self.adaptive_weights['diversity']
        
        # 9. 长期规划（新增）
        long_term_value = self._evaluate_long_term_value(new_head, game, snake)
        score += long_term_value * 2
        
        # 10. 对手干扰（新增）
        interference_value = self._evaluate_interference_potential(new_head, opponent_snake, game)
        score += interference_value * 5
        
        return score
    
    def _calculate_boundary_penalty_advanced(self, pos, game):
        """高级边界惩罚计算"""
        x, y = pos
        board_size = game.board_size
        
        # 距离边界的距离
        distance_to_edge = min(x, y, board_size - 1 - x, board_size - 1 - y)
        
        # 动态边界惩罚
        if distance_to_edge <= 1:
            return -20
        elif distance_to_edge <= 2:
            return -12
        elif distance_to_edge <= 4:
            return -6
        else:
            return 0
    
    def _calculate_available_space_advanced(self, pos, game, snake):
        """高级可用空间计算"""
        visited = set()
        queue = [pos]
        space_count = 0
        max_depth = 8  # 增加搜索深度
        
        while queue and space_count < max_depth * 4:  # 增加搜索范围
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            space_count += 1
            
            # 检查四个方向
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (self._is_position_safe(next_pos, game) and 
                    next_pos not in visited and 
                    next_pos not in queue):
                    queue.append(next_pos)
        
        return space_count
    
    def _assess_threat_level_advanced(self, pos, opponent_snake, opponent_prediction, game):
        """高级威胁评估"""
        if not opponent_snake:
            return 0
        
        threat_level = 0
        opponent_head = opponent_snake[0]
        
        # 计算与对手头的距离
        distance_to_opponent = abs(pos[0] - opponent_head[0]) + abs(pos[1] - opponent_head[1])
        
        # 动态威胁评估
        if distance_to_opponent <= 1:
            threat_level += 8
        elif distance_to_opponent <= 2:
            threat_level += 4
        elif distance_to_opponent <= 4:
            threat_level += 2
        
        # 检查是否在对手预测路径上
        if pos in opponent_prediction:
            threat_level += 6
        
        # 检查是否会被对手包围
        if self._is_position_surrounded(pos, opponent_snake, game):
            threat_level += 10
        
        return threat_level
    
    def _is_position_surrounded(self, pos, opponent_snake, game):
        """检查位置是否被对手包围"""
        if not opponent_snake:
            return False
        
        # 检查四个方向是否被对手控制
        blocked_directions = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if not self._is_position_safe(next_pos, game):
                blocked_directions += 1
        
        return blocked_directions >= 3
    
    def _is_dead_end_advanced(self, pos, game, snake):
        """高级死胡同检测"""
        available_directions = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_position_safe(next_pos, game):
                available_directions += 1
        
        # 更严格的死胡同检测
        return available_directions <= 1
    
    def _calculate_path_diversity_advanced(self, pos, game, snake):
        """高级路径多样性计算"""
        # 检查从该位置可以到达的不同方向数量
        available_directions = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_position_safe(next_pos, game):
                available_directions += 1
        
        # 考虑路径深度
        if available_directions >= 3:
            return 5
        elif available_directions >= 2:
            return 3
        else:
            return 1
    
    def _calculate_center_preference_advanced(self, pos, game):
        """高级中心偏好计算"""
        x, y = pos
        board_size = game.board_size
        center = board_size // 2
        
        # 计算到中心的距离
        distance_to_center = abs(x - center) + abs(y - center)
        
        # 动态中心偏好
        if distance_to_center <= 1:
            return 8
        elif distance_to_center <= 2:
            return 5
        elif distance_to_center <= 4:
            return 2
        else:
            return -1
    
    def _evaluate_long_term_value(self, pos, game, snake):
        """评估长期价值"""
        # 检查该位置是否通向更大的空间
        future_space = self._calculate_future_space(pos, game, snake)
        return future_space * 0.5
    
    def _calculate_future_space(self, pos, game, snake):
        """计算未来空间"""
        visited = set()
        queue = [pos]
        space_count = 0
        max_depth = 6
        
        while queue and space_count < max_depth * 3:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            space_count += 1
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (self._is_position_safe(next_pos, game) and 
                    next_pos not in visited and 
                    next_pos not in queue):
                    queue.append(next_pos)
        
        return space_count
    
    def _evaluate_interference_potential(self, pos, opponent_snake, game):
        """评估干扰对手的潜力"""
        if not opponent_snake:
            return 0
        
        opponent_head = opponent_snake[0]
        distance_to_opponent = abs(pos[0] - opponent_head[0]) + abs(pos[1] - opponent_head[1])
        
        # 如果距离对手很近，有干扰潜力
        if distance_to_opponent <= 3:
            return 3
        elif distance_to_opponent <= 5:
            return 1
        
        return 0
    
    def _find_nearest_food(self, head, foods):
        """找到最近的食物"""
        min_distance = float('inf')
        nearest_food = foods[0]
        
        for food in foods:
            distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        
        return nearest_food
    
    def _move_towards_target(self, head, target, current_direction, game):
        """向目标移动"""
        head_x, head_y = head
        target_x, target_y = target
        
        # 计算到目标的方向
        dx = target_x - head_x
        dy = target_y - head_y
        
        # 优先级：距离较远的轴优先
        if abs(dx) > abs(dy):
            if dx > 0:
                return (1, 0)  # 下
            elif dx < 0:
                return (-1, 0)  # 上
        
        if dy > 0:
            return (0, 1)  # 右
        elif dy < 0:
            return (0, -1)  # 左
        
        # 如果已经在目标位置，保持当前方向
        return current_direction
    
    def _update_position_history(self, head):
        """更新位置历史"""
        self.last_positions.append(head)
        if len(self.last_positions) > 10:  # 只保留最近10个位置
            self.last_positions.pop(0)
    
    def _predict_opponent_moves(self, opponent_snake, game):
        """预测对手可能的移动"""
        if not opponent_snake:
            return []
        
        opponent_head = opponent_snake[0]
        predictions = []
        
        # 检查四个方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (opponent_head[0] + dx, opponent_head[1] + dy)
            if self._is_position_safe(new_pos, game):
                predictions.append(new_pos)
        
        return predictions
    
    def _predict_opponent_moves_simple(self, opponent_snake, game):
        """简化版对手预测（快速）"""
        if not opponent_snake:
            return []
        
        opponent_head = opponent_snake[0]
        predictions = []
        
        # 只检查当前方向
        if self.player_id == 1:
            current_direction = game.direction2
        else:
            current_direction = game.direction1
        
        new_pos = (opponent_head[0] + current_direction[0], opponent_head[1] + current_direction[1])
        if self._is_position_safe(new_pos, game):
            predictions.append(new_pos)
        
        return predictions
    
    def _evaluate_action_comprehensive(self, action, head, game, snake, opponent_snake, opponent_prediction):
        """综合评估动作"""
        new_head = (head[0] + action[0], head[1] + action[1])
        
        # 基础安全性检查
        if not self._is_position_safe(new_head, game):
            return -1000
        
        score = 0
        
        # 1. 食物吸引力
        if game.foods:
            nearest_food = self._find_nearest_food(new_head, game.foods)
            food_distance = abs(new_head[0] - nearest_food[0]) + abs(new_head[1] - nearest_food[1])
            score += max(0, 20 - food_distance)  # 距离越近分数越高
        
        # 2. 空间评估
        available_space = self._calculate_available_space(new_head, game, snake)
        score += available_space * 5
        
        # 3. 威胁评估
        threat_level = self._assess_threat_level(new_head, opponent_snake, opponent_prediction, game)
        score -= threat_level * 10
        
        # 4. 路径长度评估
        if game.foods:
            path_length = self._calculate_path_length(new_head, game.foods[0], game)
            if path_length > 0:
                score += max(0, 15 - path_length)
        
        # 5. 避免死胡同
        if self._is_dead_end(new_head, game, snake):
            score -= 50
        
        return score
    
    def _evaluate_action_fast(self, action, head, game, snake, opponent_snake, opponent_prediction):
        """快速评估动作（改进版）"""
        new_head = (head[0] + action[0], head[1] + action[1])
        
        # 基础安全性检查
        if not self._is_position_safe(new_head, game):
            return -1000
        
        score = 0
        
        # 1. 边界感知（新增）
        boundary_penalty = self._calculate_boundary_penalty(new_head, game)
        score += boundary_penalty
        
        # 2. 食物吸引力（改进）
        if game.foods:
            nearest_food = self._find_nearest_food(new_head, game.foods)
            food_distance = abs(new_head[0] - nearest_food[0]) + abs(new_head[1] - nearest_food[1])
            # 只有当食物距离合理时才给予高分
            if food_distance <= 5:
                score += max(0, 15 - food_distance)
            else:
                score += max(0, 5 - food_distance)  # 远距离食物分数较低
        else:
            score += 0
        
        # 3. 空间评估（改进）
        available_space = self._calculate_available_space_fast(new_head, game, snake)
        score += available_space * 3  # 增加空间权重
        
        # 4. 威胁评估（改进）
        threat_level = self._assess_threat_level_simple(new_head, opponent_snake, game)
        score -= threat_level * 8  # 增加威胁惩罚
        
        # 5. 避免死胡同（改进）
        if self._is_dead_end_simple(new_head, game, snake):
            score -= 50
        
        # 6. 路径多样性（新增）
        path_diversity = self._calculate_path_diversity(new_head, game, snake)
        score += path_diversity * 2
        
        # 7. 中心偏好（新增）
        center_preference = self._calculate_center_preference(new_head, game)
        score += center_preference
        
        # 8. 历史位置惩罚（防止原地绕圈）
        if hasattr(self, 'last_positions') and new_head in self.last_positions[-6:]:
            score -= 15  # 惩罚分数可调整
        
        return score
    
    def _calculate_boundary_penalty(self, pos, game):
        """计算边界惩罚"""
        x, y = pos
        board_size = game.board_size
        
        # 距离边界的距离
        distance_to_edge = min(x, y, board_size - 1 - x, board_size - 1 - y)
        
        # 如果太靠近边界，给予惩罚
        if distance_to_edge <= 2:
            return -10
        elif distance_to_edge <= 4:
            return -5
        else:
            return 0
    
    def _calculate_path_diversity(self, pos, game, snake):
        """计算路径多样性"""
        # 检查从该位置可以到达的不同方向数量
        available_directions = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_position_safe(next_pos, game):
                available_directions += 1
        
        return available_directions
    
    def _calculate_center_preference(self, pos, game):
        """计算中心偏好"""
        x, y = pos
        board_size = game.board_size
        center = board_size // 2
        
        # 计算到中心的距离
        distance_to_center = abs(x - center) + abs(y - center)
        
        # 偏好中心区域
        if distance_to_center <= 2:
            return 5
        elif distance_to_center <= 4:
            return 2
        else:
            return -2
    
    def _calculate_available_space_fast(self, pos, game, snake):
        """快速计算可用空间（限制深度）"""
        visited = set()
        queue = [pos]
        space_count = 0
        max_depth = 5  # 限制搜索深度
        
        while queue and space_count < max_depth:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            space_count += 1
            
            # 检查四个方向
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (self._is_position_safe(next_pos, game) and 
                    next_pos not in visited and 
                    next_pos not in queue):
                    queue.append(next_pos)
        
        return space_count
    
    def _assess_threat_level_simple(self, pos, opponent_snake, game):
        """简化威胁评估"""
        if not opponent_snake:
            return 0
        
        opponent_head = opponent_snake[0]
        distance_to_opponent = abs(pos[0] - opponent_head[0]) + abs(pos[1] - opponent_head[1])
        
        if distance_to_opponent <= 2:
            return 3
        elif distance_to_opponent <= 4:
            return 1
        
        return 0
    
    def _is_dead_end_simple(self, pos, game, snake):
        """简化死胡同检测"""
        available_directions = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_position_safe(next_pos, game):
                available_directions += 1
        
        return available_directions <= 1
    
    def _is_position_safe(self, pos, game):
        """检查位置是否安全"""
        x, y = pos
        
        # 检查边界
        if x < 0 or x >= game.board_size or y < 0 or y >= game.board_size:
            return False
        
        # 检查是否撞到蛇身
        if pos in game.snake1[:-1] or pos in game.snake2[:-1]:
            return False
        
        return True
    
    def _calculate_available_space(self, pos, game, snake):
        """计算可用空间"""
        visited = set()
        queue = [pos]
        space_count = 0
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            space_count += 1
            
            # 检查四个方向
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (self._is_position_safe(next_pos, game) and 
                    next_pos not in visited and 
                    next_pos not in queue):
                    queue.append(next_pos)
        
        return space_count
    
    def _assess_threat_level(self, pos, opponent_snake, opponent_prediction, game):
        """评估威胁等级"""
        if not opponent_snake:
            return 0
        
        threat_level = 0
        opponent_head = opponent_snake[0]
        
        # 计算与对手头的距离
        distance_to_opponent = abs(pos[0] - opponent_head[0]) + abs(pos[1] - opponent_head[1])
        
        if distance_to_opponent <= 2:
            threat_level += 5
        elif distance_to_opponent <= 4:
            threat_level += 2
        
        # 检查是否在对手预测路径上
        if pos in opponent_prediction:
            threat_level += 3
        
        return threat_level
    
    def _calculate_path_length(self, start, goal, game):
        """计算路径长度（使用简化的A*）"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if self._is_position_safe((nx, ny), game):
                    neighbors.append((nx, ny))
            return neighbors
        
        # 简化的路径查找
        open_set = [(0, start)]
        visited = set()
        
        while open_set:
            current = open_set.pop(0)[1]
            if current == goal:
                return heuristic(start, goal)  # 简化返回
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    open_set.append((heuristic(neighbor, goal), neighbor))
        
        return float('inf')  # 无路径
    
    def _is_dead_end(self, pos, game, snake):
        """检查是否为死胡同"""
        # 简化的死胡同检测
        available_directions = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_position_safe(next_pos, game):
                available_directions += 1
        
        return available_directions <= 1
    
    def _is_safe_action(self, action, head, game):
        """检查动作是否安全"""
        # action已经是方向元组 (dx, dy)
        direction = action
        new_head = (head[0] + direction[0], head[1] + direction[1])
        
        # 检查边界
        if (new_head[0] < 0 or new_head[0] >= game.board_size or
            new_head[1] < 0 or new_head[1] >= game.board_size):
            return False
        
        # 检查是否撞到蛇身
        if new_head in game.snake1[:-1] or new_head in game.snake2[:-1]:
            return False
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """获取SnakeAI信息"""
        info = super().get_info()
        info.update({
            'type': 'SnakeAI',
            'description': '改进的贪吃蛇专用AI',
            'strategy': 'A*寻路 + 安全性评估 + 对手预测',
            'path_cache_size': len(self.path_cache),
            'threat_assessment_size': len(self.threat_assessment),
            'position_history_length': len(self.last_positions)
        })
        return info


class SmartSnakeAI(BaseAgent):
    """更智能的贪吃蛇AI"""
    
    def __init__(self, name="SmartSnakeAI", player_id=1):
        super().__init__(name, player_id)
        self.max_thinking_time = 0.1  # 最大思考时间100ms
    
    def get_action(self, observation, env):
        """使用A*算法寻路的贪吃蛇AI"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        game = env.game
        if self.player_id == 1:
            snake = game.snake1
            current_direction = game.direction1
        else:
            snake = game.snake2
            current_direction = game.direction2
        
        if not snake:
            return random.choice(valid_actions)
        
        head = snake[0]
        
        # 检查时间限制
        if time.time() - start_time > self.max_thinking_time:
            return random.choice(valid_actions)
        
        # 使用A*算法寻找到最近食物的路径
        if game.foods:
            target_food = self._find_nearest_food(head, game.foods)
            path = self._a_star_pathfinding_fast(head, target_food, game, start_time)
            
            if path and len(path) > 1:
                next_pos = path[1]  # path[0]是当前位置
                action = self._pos_to_action(head, next_pos)
                if action in valid_actions:
                    return action
        
        # 如果A*失败，使用安全策略
        return self._get_safe_action(head, game, valid_actions)
    
    def _find_nearest_food(self, head, foods):
        """找到最近的食物"""
        min_distance = float('inf')
        nearest_food = foods[0]
        
        for food in foods:
            distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        
        return nearest_food
    
    def _a_star_pathfinding(self, start, goal, game):
        """A*寻路算法"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < game.board_size and 0 <= ny < game.board_size):
                    # 检查是否撞到蛇身（但允许撞到尾部，因为尾部会移动）
                    if ((nx, ny) not in game.snake1[:-1] and 
                        (nx, ny) not in game.snake2[:-1]):
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 没有找到路径
    
    def _a_star_pathfinding_fast(self, start, goal, game, start_time):
        """快速A*寻路算法（带时间限制）"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < game.board_size and 0 <= ny < game.board_size):
                    # 检查是否撞到蛇身（但允许撞到尾部，因为尾部会移动）
                    if ((nx, ny) not in game.snake1[:-1] and 
                        (nx, ny) not in game.snake2[:-1]):
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        max_iterations = 50  # 限制最大迭代次数
        iteration_count = 0
        
        while open_set and iteration_count < max_iterations:
            # 检查时间限制
            if time.time() - start_time > self.max_thinking_time:
                break
                
            current = heappop(open_set)[1]
            iteration_count += 1
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 没有找到路径或超时
    
    def _pos_to_action(self, current_pos, next_pos):
        """将位置转换为动作"""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        return (dx, dy)
    
    def _get_safe_action(self, head, game, valid_actions):
        """获取安全的动作"""
        safe_actions = []
        
        for action in valid_actions:
            # action已经是方向元组
            new_head = (head[0] + action[0], head[1] + action[1])
            
            # 检查是否安全
            if (0 <= new_head[0] < game.board_size and 
                0 <= new_head[1] < game.board_size and
                new_head not in game.snake1[:-1] and 
                new_head not in game.snake2[:-1]):
                safe_actions.append(action)
        
        if safe_actions:
            return random.choice(safe_actions)
        
        return random.choice(valid_actions)


class AdvancedSnakeAI(BaseAgent):
    """高级贪吃蛇AI，包含对手预测和策略优化"""
    
    def __init__(self, name="AdvancedSnakeAI", player_id=1):
        super().__init__(name, player_id)
    
    def get_action(self, observation, env):
        """高级贪吃蛇AI决策"""
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        
        game = env.game
        if self.player_id == 1:
            snake = game.snake1
            current_direction = game.direction1
            opponent_snake = game.snake2
        else:
            snake = game.snake2
            current_direction = game.direction2
            opponent_snake = game.snake1
        
        if not snake:
            return random.choice(valid_actions)
        
        head = snake[0]
        
        # 评估每个动作的分数
        action_scores = {}
        for action in valid_actions:
            score = self._evaluate_action(action, head, game, snake, opponent_snake)
            action_scores[action] = score
        
        # 选择最高分数的动作
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        return best_action
    
    def _evaluate_action(self, action, head, game, snake, opponent_snake):
        """评估动作的分数"""
        new_head = (head[0] + action[0], head[1] + action[1])
        
        # 基础安全检查
        if not self._is_position_safe(new_head, game):
            return -1000
        
        score = 0
        
        # 食物奖励
        if game.foods:
            nearest_food = self._find_nearest_food(head, game.foods)
            food_distance = abs(new_head[0] - nearest_food[0]) + abs(new_head[1] - nearest_food[1])
            score += 50 / (food_distance + 1)  # 距离越近分数越高
        
        # 生存空间奖励
        available_space = self._calculate_available_space(new_head, game, snake)
        score += available_space * 10
        
        # 对手威胁评估
        opponent_threat = self._assess_opponent_threat(new_head, opponent_snake, game)
        score -= opponent_threat * 20
        
        # 路径长度奖励
        if game.foods:
            path_length = self._calculate_path_length(new_head, game.foods[0], game)
            if path_length is not None:
                score += 30 / (path_length + 1)
        
        return score
    
    def _is_position_safe(self, pos, game):
        """检查位置是否安全"""
        x, y = pos
        if (x < 0 or x >= game.board_size or y < 0 or y >= game.board_size):
            return False
        
        if pos in game.snake1[:-1] or pos in game.snake2[:-1]:
            return False
        
        return True
    
    def _find_nearest_food(self, head, foods):
        """找到最近的食物"""
        min_distance = float('inf')
        nearest_food = foods[0]
        
        for food in foods:
            distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            if distance < min_distance:
                min_distance = distance
                nearest_food = food
        
        return nearest_food
    
    def _calculate_available_space(self, pos, game, snake):
        """计算可用空间"""
        visited = set()
        queue = [pos]
        space_count = 0
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            space_count += 1
            
            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < game.board_size and 0 <= ny < game.board_size and
                    (nx, ny) not in visited and
                    (nx, ny) not in game.snake1 and
                    (nx, ny) not in game.snake2):
                    queue.append((nx, ny))
        
        return space_count
    
    def _assess_opponent_threat(self, pos, opponent_snake, game):
        """评估对手威胁"""
        if not opponent_snake:
            return 0
        
        threat_level = 0
        opponent_head = opponent_snake[0]
        
        # 计算到对手头部的距离
        distance = abs(pos[0] - opponent_head[0]) + abs(pos[1] - opponent_head[1])
        
        if distance <= 2:
            threat_level += 10
        elif distance <= 4:
            threat_level += 5
        
        return threat_level
    
    def _calculate_path_length(self, start, goal, game):
        """计算路径长度"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < game.board_size and 0 <= ny < game.board_size):
                    if ((nx, ny) not in game.snake1[:-1] and 
                        (nx, ny) not in game.snake2[:-1]):
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # 计算路径长度
                path_length = 0
                while current in came_from:
                    path_length += 1
                    current = came_from[current]
                return path_length
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 没有找到路径 