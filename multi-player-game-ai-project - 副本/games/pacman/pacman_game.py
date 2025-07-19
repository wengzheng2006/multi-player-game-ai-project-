"""
吃豆人游戏
实现经典吃豆人游戏逻辑
"""

import random
import time
from typing import Dict, List, Tuple, Any, Optional
from games.base_game import BaseGame
from games.pacman.maze import Maze
from games.pacman.entities import Pacman, Ghost, Fruit, Direction
from games.pacman.ghost_ai import create_ghost_ai, AggressiveGhostAI, StrategicGhostAI, CoordinatedGhostAI

class PacmanGame(BaseGame):
    """吃豆人游戏类"""
    
    def __init__(self, game_config: Dict[str, Any]|None = None):
        super().__init__()
        self.game_config = game_config or {}
        
        # 游戏状态
        self.maze = Maze()
        self.pacman = None
        self.ghosts = []
        self.ghost_ais = []  # 幽灵AI列表
        self.fruits = []
        self.score = 0
        self.level = 1
        self.lives = 3
        self.pacman_alive = True
        self.ghosts_alive = True
        self.game_mode = 'single'  # 'single', 'multi'
        self.ghost_difficulty = 'normal'  # 'easy', 'normal', 'hard'
        self.ghost_ai_type = 'aggressive'  # 'aggressive', 'strategic', 'coordinated'
        self.ghost_aggression_level = 1.8  # 攻击性等级
        
        # 计时器
        self.fruit_spawn_timer = 100
        self.ghost_ai_timer = 0
        
        # 初始化游戏
        self._initialize_game()
    
    def _initialize_game(self):
        """初始化游戏"""
        # 创建吃豆人
        spawn_pos = self.maze.get_pacman_spawn_position()
        self.pacman = Pacman(*spawn_pos)
        
        # 创建幽灵和幽灵AI
        self.ghosts.clear()
        self.ghost_ais.clear()
        spawn_positions = self.maze.get_ghost_spawn_positions()
        ghost_types = ['red', 'pink', 'blue', 'orange']
        
        for i in range(min(4, len(spawn_positions))):
            if i < len(spawn_positions):
                ghost = Ghost(*spawn_positions[i], ghost_types[i])
                self.ghosts.append(ghost)
                
                # 创建对应的幽灵AI
                if self.ghost_ai_type == 'coordinated':
                    # 协调性AI需要所有幽灵的信息
                    ghost_ai = create_ghost_ai(ghost, self.maze, self.ghost_ai_type, 
                                             ghost_id=i, all_ghosts=self.ghosts)
                else:
                    ghost_ai = create_ghost_ai(ghost, self.maze, self.ghost_ai_type, 
                                             aggression_level=self.ghost_aggression_level)
                self.ghost_ais.append(ghost_ai)
            else:
                self.ghosts.append(None)
                self.ghost_ais.append(None)
    
    def reset(self) -> Dict[str, Any]:
        """重置游戏"""
        # 重置游戏状态
        self.maze = Maze()
        self.score = 0
        self.level = 1
        self.lives = 3
        self.pacman_alive = True
        self.ghosts_alive = True
        self.fruits = []
        self.fruit_spawn_timer = 100
        self.ghost_ai_timer = 0
        self.ghosts.clear()
        self.ghost_ais.clear()
        
        # 重新初始化
        self._initialize_game()
        
        return self.get_state()
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行一步动作"""
        if not self.pacman_alive:
            return self.get_state(), 0, True, {'message': 'Pacman is dead'}
        
        # 解析动作
        if isinstance(action, Direction):
            direction = action
        elif isinstance(action, str):
            direction = self._string_to_direction(action)
        elif isinstance(action, tuple) and len(action) == 2:
            direction = self._tuple_to_direction(action)
        else:
            direction = Direction.NONE
        
        # 移动吃豆人
        moved = False
        if self.pacman:
            moved = self.pacman.move(direction, self.maze)
        
        # 吃豆子
        ate_dot = False
        if self.pacman:
            ate_dot = self.pacman.eat_dot(self.maze.dots, self.maze.power_dots)
            if ate_dot:
                self.score += 10 if not self.pacman.is_power_mode() else 50
        
        # 使用增强的幽灵AI
        self.ghost_ai_timer += 1
        if self.ghost_ai_timer % 2 == 0:  # 每2帧移动一次，提高响应速度
            self._move_ghosts_with_ai()
        
        # 检查与幽灵的碰撞
        collision_info = self._check_ghost_collision()
        
        # 更新幽灵状态
        self._update_ghosts()
        
        # 更新水果
        self._update_fruits()
        
        # 检查游戏结束条件
        done = self._check_game_end()
        
        # 计算奖励
        reward = self._calculate_reward(ate_dot, collision_info)
        
        # 记录移动
        self.record_move(self.current_player, action, {
            'moved': moved,
            'ate_dot': ate_dot,
            'collision': collision_info,
            'score': self.score
        })
        
        return self.get_state(), reward, done, {
            'moved': moved,
            'ate_dot': ate_dot,
            'collision': collision_info,
            'score': self.score,
            'lives': self.lives
        }
    
    def _move_ghosts_with_ai(self):
        """使用增强的幽灵AI移动幽灵"""
        if not self.pacman:
            return
        
        pacman_pos = self.pacman.get_position()
        pacman_power = self.pacman.is_power_mode()
        
        for i, (ghost, ghost_ai) in enumerate(zip(self.ghosts, self.ghost_ais)):
            if ghost and ghost_ai:
                # 使用幽灵AI获取动作
                action = ghost_ai.get_action(pacman_pos, pacman_power)
                
                # 执行移动
                if action != Direction.NONE:
                    ghost.move(action, self.maze)
    
    def _move_ghosts_simple(self):
        """简单幽灵AI：每个幽灵朝吃豆人方向移动一步（保留作为备用）"""
        if not self.pacman:
            return
        px, py = self.pacman.get_position()
        for ghost in self.ghosts:
            if ghost:
                gx, gy = ghost.get_position()
                # 计算最佳方向
                dx = px - gx
                dy = py - gy
                move_options = []
                if abs(dx) > abs(dy):
                    if dx > 0:
                        move_options.append(Direction.RIGHT)
                    elif dx < 0:
                        move_options.append(Direction.LEFT)
                    if dy > 0:
                        move_options.append(Direction.DOWN)
                    elif dy < 0:
                        move_options.append(Direction.UP)
                else:
                    if dy > 0:
                        move_options.append(Direction.DOWN)
                    elif dy < 0:
                        move_options.append(Direction.UP)
                    if dx > 0:
                        move_options.append(Direction.RIGHT)
                    elif dx < 0:
                        move_options.append(Direction.LEFT)
                # 只选择可行方向
                for direction in move_options:
                    dx, dy = direction.value
                    new_x, new_y = gx + dx, gy + dy
                    if self.maze.is_valid_position(new_x, new_y) and not self.maze.is_wall(new_x, new_y):
                        ghost.move(direction, self.maze)
                        break
    
    def get_valid_actions(self, player: int|None = None) -> List[Any]:
        """获取有效动作"""
        if player == 1:  # 吃豆人
            return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        elif player == 2:  # 幽灵
            return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        else:
            return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return (not self.pacman_alive or 
                self.lives <= 0 or 
                self.maze.get_dots_count() == 0)
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        if self.maze.get_dots_count() == 0:
            return 1  # 吃豆人获胜
        elif not self.pacman_alive or self.lives <= 0:
            return 2  # 幽灵获胜
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前游戏状态"""
        return {
            'maze': self.maze,
            'pacman': self.pacman,
            'ghosts': self.ghosts,
            'fruits': self.fruits,
            'score': self.score,
            'level': self.level,
            'lives': self.lives,
            'pacman_alive': self.pacman_alive,
            'ghosts_alive': self.ghosts_alive,
            'dots_count': self.maze.get_dots_count(),
            'power_dots_count': len(self.maze.power_dots),
            'game_mode': self.game_mode,
            'ghost_difficulty': self.ghost_difficulty,
            'ghost_ai_type': self.ghost_ai_type,
            'ghost_aggression_level': self.ghost_aggression_level
        }
    
    def render(self) -> str:
        """渲染游戏画面"""
        # 创建渲染网格
        render_grid = [[' ' for _ in range(self.maze.width)] for _ in range(self.maze.height)]
        
        # 绘制迷宫
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if self.maze.is_wall(x, y):
                    render_grid[y][x] = '#'
                elif (x, y) in self.maze.dots:
                    render_grid[y][x] = '.'
                elif (x, y) in self.maze.power_dots:
                    render_grid[y][x] = 'o'
        
        # 绘制吃豆人
        if self.pacman and self.pacman_alive:
            px, py = self.pacman.get_position()
            render_grid[py][px] = 'P'
        
        # 绘制幽灵
        for ghost in self.ghosts:
            if ghost:
                gx, gy = ghost.get_position()
                render_grid[gy][gx] = ghost.symbol
        
        # 绘制水果
        for fruit in self.fruits:
            if fruit.is_active():
                fx, fy = fruit.get_position()
                render_grid[fy][fx] = 'F'
        
        # 转换为字符串
        result = []
        for row in render_grid:
            result.append(''.join(row))
        
        # 添加状态信息
        result.append(f"Score: {self.score} | Lives: {self.lives} | Level: {self.level}")
        result.append(f"Dots: {self.maze.get_dots_count()} | Power Dots: {len(self.maze.power_dots)}")
        result.append(f"Ghost AI: {self.ghost_ai_type} | Aggression: {self.ghost_aggression_level}")
        
        return '\n'.join(result)
    
    def _string_to_direction(self, direction_str: str) -> Direction:
        """字符串转方向"""
        direction_map = {
            'up': Direction.UP,
            'down': Direction.DOWN,
            'left': Direction.LEFT,
            'right': Direction.RIGHT,
            'w': Direction.UP,
            's': Direction.DOWN,
            'a': Direction.LEFT,
            'd': Direction.RIGHT
        }
        return direction_map.get(direction_str.lower(), Direction.NONE)
    
    def _tuple_to_direction(self, direction_tuple: Tuple[int, int]) -> Direction:
        """元组转方向"""
        dx, dy = direction_tuple
        if dx == 0 and dy == -1:
            return Direction.UP
        elif dx == 0 and dy == 1:
            return Direction.DOWN
        elif dx == -1 and dy == 0:
            return Direction.LEFT
        elif dx == 1 and dy == 0:
            return Direction.RIGHT
        else:
            return Direction.NONE
    
    def _check_ghost_collision(self) -> Dict[str, Any]:
        """检查与幽灵的碰撞"""
        if not self.pacman or not self.pacman_alive:
            return {'collision': False, 'ghost_type': None, 'pacman_power': False}
        
        pacman_pos = self.pacman.get_position()
        
        for ghost in self.ghosts:
            if ghost and ghost.get_position() == pacman_pos:
                if self.pacman.is_power_mode():
                    # 吃豆人处于能量模式，可以吃掉幽灵
                    self.score += 200
                    # 重置幽灵位置
                    spawn_positions = self.maze.get_ghost_spawn_positions()
                    if spawn_positions:
                        ghost.set_position(*random.choice(spawn_positions))
                    return {
                        'collision': True,
                        'ghost_type': ghost.ghost_type,
                        'pacman_power': True,
                        'ghost_eaten': True
                    }
                else:
                    # 吃豆人被幽灵吃掉
                    self.lives -= 1
                    if self.lives <= 0:
                        self.pacman_alive = False
                    else:
                        # 重置吃豆人位置
                        spawn_pos = self.maze.get_pacman_spawn_position()
                        self.pacman.set_position(*spawn_pos)
                    
                    return {
                        'collision': True,
                        'ghost_type': ghost.ghost_type,
                        'pacman_power': False,
                        'ghost_eaten': False
                    }
        
        return {'collision': False, 'ghost_type': None, 'pacman_power': False}
    
    def _update_ghosts(self):
        """更新幽灵状态"""
        for ghost in self.ghosts:
            if ghost:
                # 更新恐惧状态
                ghost.update_frightened()
                
                # 更新能量模式对幽灵的影响
                if self.pacman and self.pacman.is_power_mode():
                    if not ghost.is_frightened():
                        ghost.set_frightened(20)
    
    def _update_fruits(self):
        """更新水果状态"""
        # 水果生成逻辑
        if self.fruit_spawn_timer <= 0 and len(self.fruits) == 0:
            # 生成水果
            fruit_pos = self.maze.get_random_empty_position()
            fruit_type = random.choice(['cherry', 'strawberry', 'orange', 'apple'])
            fruit = Fruit(*fruit_pos, fruit_type)
            fruit.activate(30)
            self.fruits.append(fruit)
            self.fruit_spawn_timer = 100  # 重置计时器
        
        # 更新水果状态
        for fruit in self.fruits:
            fruit.update()
        
        # 移除过期水果
        self.fruits = [f for f in self.fruits if f.is_active()]
        
        # 更新计时器
        if self.fruit_spawn_timer > 0:
            self.fruit_spawn_timer -= 1
    
    def _check_game_end(self) -> bool:
        """检查游戏结束条件"""
        # 检查豆子是否吃完
        if self.maze.get_dots_count() == 0:
            self.level += 1
            # 重新生成豆子
            self.maze._place_dots()
            return False
        
        # 检查吃豆人是否死亡
        if not self.pacman_alive or self.lives <= 0:
            return True
        
        return False
    
    def _calculate_reward(self, ate_dot: bool, collision_info: Dict[str, Any]) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 吃豆子奖励
        if ate_dot:
            reward += 10.0
        
        # 碰撞奖励/惩罚
        if collision_info['collision']:
            if collision_info['ghost_eaten']:
                reward += 200.0  # 吃掉幽灵
            else:
                reward -= 100.0  # 被幽灵吃掉
        
        # 生存奖励
        reward += 1.0
        
        return reward
    
    def move_ghost(self, ghost_index: int, direction: Direction):
        """移动指定幽灵"""
        if 0 <= ghost_index < len(self.ghosts):
            ghost = self.ghosts[ghost_index]
            if ghost:
                ghost.move(direction, self.maze)
    
    def get_ghost_count(self) -> int:
        """获取幽灵数量"""
        return len(self.ghosts)
    
    def get_pacman_position(self) -> Optional[Tuple[int, int]]:
        """获取吃豆人位置"""
        if self.pacman:
            return self.pacman.get_position()
        return None
    
    def get_ghost_positions(self) -> List[Tuple[int, int]]:
        """获取所有幽灵位置"""
        positions = []
        for ghost in self.ghosts:
            if ghost:
                positions.append(ghost.get_position())
        return positions
    
    def set_game_mode(self, mode: str):
        """设置游戏模式"""
        self.game_mode = mode
    
    def set_ghost_difficulty(self, difficulty: str):
        """设置幽灵难度"""
        self.ghost_difficulty = difficulty
    
    def set_ghost_ai_type(self, ai_type: str):
        """设置幽灵AI类型"""
        self.ghost_ai_type = ai_type
        # 重新初始化幽灵AI
        self._initialize_game()
    
    def set_ghost_aggression_level(self, level: float):
        """设置幽灵攻击性等级"""
        self.ghost_aggression_level = max(0.0, min(2.0, level))
        # 重新初始化幽灵AI
        self._initialize_game()
    
    def clone(self) -> 'PacmanGame':
        """克隆游戏状态"""
        # 简化实现：创建新游戏
        new_game = PacmanGame(self.game_config)
        new_game.maze = self.maze
        new_game.score = self.score
        new_game.level = self.level
        new_game.lives = self.lives
        new_game.pacman_alive = self.pacman_alive
        new_game.ghosts_alive = self.ghosts_alive
        new_game.game_mode = self.game_mode
        new_game.ghost_difficulty = self.ghost_difficulty
        new_game.ghost_ai_type = self.ghost_ai_type
        new_game.ghost_aggression_level = self.ghost_aggression_level
        # 克隆实体
        if self.pacman:
            new_game.pacman = Pacman(self.pacman.x, self.pacman.y)
            new_game.pacman.score = self.pacman.score
            new_game.pacman.lives = self.pacman.lives
            new_game.pacman.power_mode = self.pacman.power_mode
            new_game.pacman.power_timer = self.pacman.power_timer
            new_game.pacman.direction = self.pacman.direction
        for i, ghost in enumerate(self.ghosts):
            if ghost:
                new_ghost = Ghost(ghost.x, ghost.y, ghost.ghost_type)
                new_ghost.direction = ghost.direction
                new_ghost.frightened = ghost.frightened
                new_ghost.frightened_timer = ghost.frightened_timer
                new_ghost.symbol = ghost.symbol
                new_game.ghosts[i] = new_ghost
        return new_game 