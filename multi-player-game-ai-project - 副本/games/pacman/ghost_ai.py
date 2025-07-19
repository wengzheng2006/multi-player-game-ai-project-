"""
幽灵AI系统
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from games.pacman.entities import Direction, Ghost
from games.pacman.maze import Maze

class GhostAI:
    """幽灵AI基类"""
    
    def __init__(self, ghost: Ghost, maze: Maze):
        self.ghost = ghost
        self.maze = maze
        self.target = None
        self.last_position = None
        self.stuck_counter = 0
        self.aggression_level = 2.0  # 攻击性等级 (0.0-2.0)
        self.patience = 0  # 耐心值，用于避免频繁改变目标
        self.position_history = deque(maxlen=10)  # 位置历史，用于检测卡住
        self.last_target = None
        self.target_stuck_counter = 0
    
    def get_action(self, pacman_pos: Tuple[int, int], pacman_power: bool) -> Direction:
        """获取幽灵动作"""
        current_pos = self.ghost.get_position()
        self.position_history.append(current_pos)
        
        # 检测是否卡住
        if self._is_stuck():
            self.stuck_counter += 1
            if self.stuck_counter > 5:
                self._force_escape()
        else:
            self.stuck_counter = 0
        
        # 更新目标
        self._update_target(pacman_pos, pacman_power)
        
        # 获取可用方向
        available_directions = self._get_available_directions()
        
        if not available_directions:
            return Direction.NONE
        
        # 选择最佳方向
        best_direction = self._select_best_direction(available_directions, pacman_pos, pacman_power)
        
        # 更新位置记录
        self.last_position = current_pos
        
        return best_direction
    
    def _is_stuck(self) -> bool:
        """检测幽灵是否卡住"""
        if len(self.position_history) < 5:
            return False
        
        # 检查最近5个位置是否都在同一区域
        recent_positions = list(self.position_history)[-5:]
        x_coords = [pos[0] for pos in recent_positions]
        y_coords = [pos[1] for pos in recent_positions]
        
        # 如果x和y坐标的变化范围都很小，说明可能卡住了
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        return x_range <= 2 and y_range <= 2
    
    def _force_escape(self):
        """强制逃脱卡住状态"""
        # 随机选择一个远离当前位置的方向
        current_pos = self.ghost.get_position()
        available_directions = self._get_available_directions()
        
        if available_directions:
            # 选择能带我们远离当前位置的方向
            best_escape = None
            max_distance = 0
            
            for direction in available_directions:
                dx, dy = direction.value
                new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
                
                # 计算到当前位置的距离
                distance = abs(new_x - current_pos[0]) + abs(new_y - current_pos[1])
                
                if distance > max_distance:
                    max_distance = distance
                    best_escape = direction
            
            if best_escape:
                self.stuck_counter = 0
                return best_escape
        
        return random.choice(available_directions) if available_directions else Direction.NONE
    
    def _update_target(self, pacman_pos: Tuple[int, int], pacman_power: bool):
        """更新目标位置"""
        if pacman_power:
            # 能量模式下，远离吃豆人
            self.target = self._get_flee_target(pacman_pos)
        else:
            # 正常模式下，追踪吃豆人
            self.target = pacman_pos
        
        # 检测目标是否卡住
        if self.target == self.last_target:
            self.target_stuck_counter += 1
        else:
            self.target_stuck_counter = 0
            self.last_target = self.target
    
    def _get_available_directions(self) -> List[Direction]:
        """获取可用方向"""
        x, y = self.ghost.get_position()
        available = []
        
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            dx, dy = direction.value
            new_x, new_y = x + dx, y + dy
            
            if (self.maze.is_valid_position(new_x, new_y) and 
                not self.maze.is_wall(new_x, new_y)):
                available.append(direction)
        
        return available
    
    def _select_best_direction(self, available_directions: List[Direction], 
                              pacman_pos: Tuple[int, int], pacman_power: bool) -> Direction:
        """选择最佳方向"""
        if not self.target:
            return random.choice(available_directions)
        
        # 计算到目标的距离
        distances = {}
        for direction in available_directions:
            dx, dy = direction.value
            new_x = self.ghost.x + dx
            new_y = self.ghost.y + dy
            new_pos = (new_x, new_y)
            
            # 计算曼哈顿距离
            distance = abs(new_x - self.target[0]) + abs(new_y - self.target[1])
            
            # 应用攻击性调整
            if not pacman_power:
                # 正常模式下，距离越近越好
                adjusted_distance = distance * (1.0 - self.aggression_level * 0.3)
            else:
                # 能量模式下，距离越远越好
                adjusted_distance = distance * (1.0 + self.aggression_level * 0.5)
            
            distances[direction] = adjusted_distance
        
        # 选择最佳方向
        if pacman_power:
            # 能量模式下选择距离最远的方向
            return max(distances.keys(), key=lambda d: distances[d])
        else:
            # 正常模式下选择距离最近的方向
            return min(distances.keys(), key=lambda d: distances[d])
    
    def _get_flee_target(self, pacman_pos: Tuple[int, int]) -> Tuple[int, int]:
        """获取逃跑目标位置"""
        # 选择远离吃豆人的位置
        x, y = self.ghost.get_position()
        px, py = pacman_pos
        
        # 计算逃跑方向
        dx = x - px
        dy = y - py
        
        # 寻找远离吃豆人的安全位置
        for distance in range(5, 15):
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                target_x = int(x + distance * math.cos(rad))
                target_y = int(y + distance * math.sin(rad))
                
                if (self.maze.is_valid_position(target_x, target_y) and 
                    not self.maze.is_wall(target_x, target_y)):
                    return (target_x, target_y)
        
        # 如果找不到理想位置，返回当前位置
        return (x, y)

class AggressiveGhostAI(GhostAI):
    """攻击性幽灵AI"""
    
    def __init__(self, ghost: Ghost, maze: Maze, aggression_level: float = 1.8):
        super().__init__(ghost, maze)
        self.aggression_level = max(1.0, min(2.0, aggression_level))
        self.attack_mode = True
        self.last_attack_time = 0
        self.attack_cooldown = 10
        self.path_finder = PathFinder(maze)
        self.last_pacman_pos = None
        self.pacman_velocity = (0, 0)
    
    def _select_best_direction(self, available_directions: List[Direction], 
                              pacman_pos: Tuple[int, int], pacman_power: bool) -> Direction:
        """攻击性选择方向"""
        if pacman_power:
            return super()._select_best_direction(available_directions, pacman_pos, pacman_power)
        
        # 更新吃豆人速度
        if self.last_pacman_pos:
            self.pacman_velocity = (
                pacman_pos[0] - self.last_pacman_pos[0],
                pacman_pos[1] - self.last_pacman_pos[1]
            )
        self.last_pacman_pos = pacman_pos
        
        # 攻击模式下，使用更智能的路径规划
        x, y = self.ghost.get_position()
        px, py = pacman_pos
        
        # 计算到吃豆人的距离
        current_distance = abs(x - px) + abs(y - py)
        
        best_direction = None
        best_score = float('-inf')
        
        for direction in available_directions:
            dx, dy = direction.value
            new_x, new_y = x + dx, y + dy
            new_distance = abs(new_x - px) + abs(new_y - py)
            
            # 基础攻击性评分：距离减少越多越好
            distance_improvement = current_distance - new_distance
            score = distance_improvement * self.aggression_level * 2.0
            
            # 额外奖励：如果这个方向能直接接近吃豆人
            if new_distance < current_distance:
                score += 10
            
            # 预测吃豆人移动的奖励
            predicted_score = self._predict_pacman_movement(new_x, new_y, pacman_pos)
            score += predicted_score * self.aggression_level
            
            # 路径质量奖励：选择更好的路径
            path_quality = self._evaluate_path_quality(new_x, new_y, pacman_pos)
            score += path_quality
            
            # 如果卡住，增加随机性
            if self.stuck_counter > 3:
                score += random.uniform(-5, 5)
            
            if score > best_score:
                best_score = score
                best_direction = direction
        
        return best_direction or random.choice(available_directions)
    
    def _predict_pacman_movement(self, ghost_x: int, ghost_y: int, pacman_pos: Tuple[int, int]) -> float:
        """预测吃豆人移动并评分"""
        px, py = pacman_pos
        
        # 使用吃豆人的速度进行预测
        if self.pacman_velocity != (0, 0):
            predicted_x = px + self.pacman_velocity[0] * 2
            predicted_y = py + self.pacman_velocity[1] * 2
        else:
            # 如果没有速度信息，使用简单的预测
            predicted_x = px + (px - ghost_x) * 0.3
            predicted_y = py + (py - ghost_y) * 0.3
        
        # 计算到预测位置的距离
        distance_to_predicted = abs(ghost_x - predicted_x) + abs(ghost_y - predicted_y)
        
        # 距离越近，分数越高
        return max(0, 15 - distance_to_predicted)
    
    def _evaluate_path_quality(self, ghost_x: int, ghost_y: int, target_pos: Tuple[int, int]) -> float:
        """评估路径质量"""
        # 检查路径是否通向目标
        path_score = 0
        
        # 检查是否有直接路径到目标
        if self._has_direct_path(ghost_x, ghost_y, target_pos):
            path_score += 8
        
        # 检查是否会进入死胡同
        if self._is_dead_end(ghost_x, ghost_y):
            path_score -= 5
        
        return path_score
    
    def _has_direct_path(self, start_x: int, start_y: int, target_pos: Tuple[int, int]) -> bool:
        """检查是否有直接路径到目标"""
        # 简单的直线检查
        tx, ty = target_pos
        dx = tx - start_x
        dy = ty - start_y
        
        # 检查主要方向是否有障碍
        if abs(dx) > abs(dy):
            # 水平移动为主
            step_x = 1 if dx > 0 else -1
            for x in range(start_x, tx, step_x):
                if self.maze.is_wall(x, start_y):
                    return False
        else:
            # 垂直移动为主
            step_y = 1 if dy > 0 else -1
            for y in range(start_y, ty, step_y):
                if self.maze.is_wall(start_x, y):
                    return False
        
        return True
    
    def _is_dead_end(self, x: int, y: int) -> bool:
        """检查是否是死胡同"""
        # 检查周围有多少个可用方向
        available_count = 0
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            dx, dy = direction.value
            new_x, new_y = x + dx, y + dy
            
            if (self.maze.is_valid_position(new_x, new_y) and 
                not self.maze.is_wall(new_x, new_y)):
                available_count += 1
        
        return available_count <= 1

class PathFinder:
    """路径查找器"""
    
    def __init__(self, maze: Maze):
        self.maze = maze
    
    def find_path(self, start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """使用A*算法查找路径"""
        if start == target:
            return [start]
        
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, target)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == target:
                return self._reconstruct_path(came_from, current)
            
            open_set.remove(current)
            
            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, target)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        return []  # 没有找到路径
    
    def _heuristic(self, pos: Tuple[int, int], target: Tuple[int, int]) -> int:
        """曼哈顿距离启发式"""
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居位置"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            
            if (self.maze.is_valid_position(new_x, new_y) and 
                not self.maze.is_wall(new_x, new_y)):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

class StrategicGhostAI(GhostAI):
    """战略性幽灵AI"""
    
    def __init__(self, ghost: Ghost, maze: Maze, strategy: str = 'ambush'):
        super().__init__(ghost, maze)
        self.strategy = strategy  # 'ambush', 'chase', 'patrol'
        self.ambush_positions = []
        self.patrol_route = []
        self.strategy_timer = 0
        self.aggression_level = 1.6  # 提高攻击性
        self.path_finder = PathFinder(maze)
    
    def _update_target(self, pacman_pos: Tuple[int, int], pacman_power: bool):
        """战略性更新目标"""
        if pacman_power:
            self.target = self._get_flee_target(pacman_pos)
            return
        
        # 根据策略选择目标
        if self.strategy == 'ambush':
            self.target = self._get_ambush_target(pacman_pos)
        elif self.strategy == 'chase':
            self.target = pacman_pos
        elif self.strategy == 'patrol':
            self.target = self._get_patrol_target()
        
        # 定期切换策略，但更倾向于攻击性策略
        self.strategy_timer += 1
        if self.strategy_timer > 30:  # 减少切换间隔
            self._switch_strategy()
            self.strategy_timer = 0
    
    def _get_ambush_target(self, pacman_pos: Tuple[int, int]) -> Tuple[int, int]:
        """获取伏击目标位置"""
        # 寻找吃豆人可能经过的关键位置
        px, py = pacman_pos
        
        # 寻找附近的豆子位置
        nearby_dots = []
        for dot_pos in self.maze.dots:
            distance = abs(dot_pos[0] - px) + abs(dot_pos[1] - py)
            if distance <= 8:
                nearby_dots.append(dot_pos)
        
        if nearby_dots:
            # 选择最近的豆子作为伏击点
            return min(nearby_dots, key=lambda pos: 
                      abs(pos[0] - self.ghost.x) + abs(pos[1] - self.ghost.y))
        
        # 如果没有合适的豆子，选择吃豆人附近的位置
        return pacman_pos
    
    def _get_patrol_target(self) -> Tuple[int, int]:
        """获取巡逻目标位置"""
        if not self.patrol_route:
            self._generate_patrol_route()
        
        if self.patrol_route:
            return self.patrol_route[0]
        
        return self.ghost.get_position()
    
    def _generate_patrol_route(self):
        """生成巡逻路线"""
        # 选择迷宫中的关键位置作为巡逻点
        key_positions = [
            (self.maze.width // 4, self.maze.height // 4),
            (3 * self.maze.width // 4, self.maze.height // 4),
            (self.maze.width // 4, 3 * self.maze.height // 4),
            (3 * self.maze.width // 4, 3 * self.maze.height // 4)
        ]
        
        # 过滤掉墙壁位置
        valid_positions = [pos for pos in key_positions 
                          if self.maze.is_valid_position(*pos) and not self.maze.is_wall(*pos)]
        
        if valid_positions:
            self.patrol_route = valid_positions
    
    def _switch_strategy(self):
        """切换策略，更倾向于攻击性策略"""
        strategies = ['chase', 'ambush', 'patrol']  # 重新排序，chase优先
        current_index = strategies.index(self.strategy)
        next_index = (current_index + 1) % len(strategies)
        self.strategy = strategies[next_index]

class CoordinatedGhostAI(GhostAI):
    """协调性幽灵AI"""
    
    def __init__(self, ghost: Ghost, maze: Maze, ghost_id: int, all_ghosts: List[Ghost]):
        super().__init__(ghost, maze)
        self.ghost_id = ghost_id
        self.all_ghosts = all_ghosts
        self.aggression_level = 1.7  # 提高攻击性
        self.coordination_mode = 'surround'  # 'surround', 'divide', 'corner'
        self.path_finder = PathFinder(maze)
    
    def _update_target(self, pacman_pos: Tuple[int, int], pacman_power: bool):
        """协调性更新目标"""
        if pacman_power:
            self.target = self._get_flee_target(pacman_pos)
            return
        
        # 根据协调模式选择目标
        if self.coordination_mode == 'surround':
            self.target = self._get_surround_target(pacman_pos)
        elif self.coordination_mode == 'divide':
            self.target = self._get_divide_target(pacman_pos)
        elif self.coordination_mode == 'corner':
            self.target = self._get_corner_target(pacman_pos)
    
    def _get_surround_target(self, pacman_pos: Tuple[int, int]) -> Tuple[int, int]:
        """获取包围目标位置"""
        px, py = pacman_pos
        ghost_pos = self.ghost.get_position()
        
        # 计算其他幽灵的位置
        other_ghost_positions = []
        for i, ghost in enumerate(self.all_ghosts):
            if i != self.ghost_id and ghost:
                other_ghost_positions.append(ghost.get_position())
        
        # 寻找包围点
        surround_points = [
            (px - 2, py), (px + 2, py),  # 左右
            (px, py - 2), (px, py + 2),  # 上下
            (px - 1, py - 1), (px + 1, py - 1),  # 对角线
            (px - 1, py + 1), (px + 1, py + 1)
        ]
        
        # 选择未被其他幽灵占据的包围点
        available_points = []
        for point in surround_points:
            if (self.maze.is_valid_position(*point) and 
                not self.maze.is_wall(*point) and
                point not in other_ghost_positions):
                available_points.append(point)
        
        if available_points:
            # 选择距离当前幽灵最近的包围点
            return min(available_points, key=lambda pos: 
                      abs(pos[0] - ghost_pos[0]) + abs(pos[1] - ghost_pos[1]))
        
        return pacman_pos
    
    def _get_divide_target(self, pacman_pos: Tuple[int, int]) -> Tuple[int, int]:
        """获取分割目标位置"""
        # 根据幽灵ID分配不同的区域
        px, py = pacman_pos
        
        # 将迷宫分为四个象限
        quadrants = [
            (0, 0), (self.maze.width // 2, 0),
            (0, self.maze.height // 2), (self.maze.width // 2, self.maze.height // 2)
        ]
        
        target_quadrant = quadrants[self.ghost_id % 4]
        return target_quadrant
    
    def _get_corner_target(self, pacman_pos: Tuple[int, int]) -> Tuple[int, int]:
        """获取角落目标位置"""
        # 将吃豆人逼向角落
        px, py = pacman_pos
        
        # 计算吃豆人距离各个角落的距离
        corners = [
            (1, 1), (self.maze.width - 2, 1),
            (1, self.maze.height - 2), (self.maze.width - 2, self.maze.height - 2)
        ]
        
        # 找到最近的角落
        nearest_corner = min(corners, key=lambda corner: 
                           abs(corner[0] - px) + abs(corner[1] - py))
        
        return nearest_corner

def create_ghost_ai(ghost: Ghost, maze: Maze, ai_type: str = 'aggressive', **kwargs) -> GhostAI:
    """创建幽灵AI"""
    if ai_type == 'aggressive':
        aggression_level = kwargs.get('aggression_level', 1.8)
        return AggressiveGhostAI(ghost, maze, aggression_level=aggression_level)
    elif ai_type == 'strategic':
        strategy = kwargs.get('strategy', 'ambush')
        return StrategicGhostAI(ghost, maze, strategy=strategy)
    elif ai_type == 'coordinated':
        ghost_id = kwargs.get('ghost_id', 0)
        all_ghosts = kwargs.get('all_ghosts', [])
        return CoordinatedGhostAI(ghost, maze, ghost_id, all_ghosts)
    else:
        return GhostAI(ghost, maze) 