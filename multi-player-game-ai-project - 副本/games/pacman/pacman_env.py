"""
吃豆人游戏环境包装器
实现gym风格的环境接口
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from games.base_env import BaseEnv
from games.pacman.pacman_game import PacmanGame
from games.pacman.entities import Direction

class PacmanEnv(BaseEnv):
    """吃豆人游戏环境"""
    
    def __init__(self, game_config: Dict[str, Any]|None = None):
        self.game = PacmanGame(game_config)
        
        # 如果提供了幽灵AI配置，设置到游戏中
        if game_config:
            if 'ghost_ai_type' in game_config:
                self.game.set_ghost_ai_type(game_config['ghost_ai_type'])
            if 'ghost_aggression_level' in game_config:
                self.game.set_ghost_aggression_level(game_config['ghost_aggression_level'])
        
        super().__init__(self.game)
    
    def _setup_spaces(self) -> None:
        """设置观察空间和动作空间"""
        # 观察空间：迷宫大小 + 额外信息
        maze_height = self.game.maze.height
        maze_width = self.game.maze.width
        self.observation_space = np.zeros((maze_height, maze_width, 5))  # 5个通道
        
        # 动作空间：4个方向
        self.action_space = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        maze = self.game.maze
        height, width = maze.height, maze.width
        
        # 创建5通道的观察
        observation = np.zeros((height, width, 5))
        
        # 通道0：墙壁
        for y in range(height):
            for x in range(width):
                if maze.is_wall(x, y):
                    observation[y, x, 0] = 1
        
        # 通道1：豆子
        for y in range(height):
            for x in range(width):
                if maze.has_dot(x, y):
                    observation[y, x, 1] = 1
                elif maze.has_power_dot(x, y):
                    observation[y, x, 1] = 2  # 能量豆用2表示
        
        # 通道2：吃豆人
        if self.game.pacman and self.game.pacman_alive:
            px, py = self.game.pacman.get_position()
            observation[py, px, 2] = 1
        
        # 通道3：幽灵
        for ghost in self.game.ghosts:
            if ghost:
                gx, gy = ghost.get_position()
                if ghost.is_frightened():
                    observation[gy, gx, 3] = 2  # 恐惧状态用2表示
                else:
                    observation[gy, gx, 3] = 1
        
        # 通道4：水果
        for fruit in self.game.fruits:
            if fruit.is_active():
                fx, fy = fruit.get_position()
                observation[fy, fx, 4] = 1
        
        return observation
    
    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        # 所有动作都有效
        return np.ones(4, dtype=bool)
    
    def get_pacman_action(self, action: Any) -> Direction:
        """获取吃豆人动作"""
        if isinstance(action, Direction):
            return action
        elif isinstance(action, int):
            return self.action_space[action]
        elif isinstance(action, str):
            return self._string_to_direction(action)
        else:
            return Direction.NONE
    
    def get_ghost_action(self, ghost_index: int, action: Any) -> Direction:
        """获取幽灵动作"""
        if isinstance(action, Direction):
            return action
        elif isinstance(action, int):
            return self.action_space[action]
        elif isinstance(action, str):
            return self._string_to_direction(action)
        else:
            return Direction.NONE
    
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
    
    def step_pacman(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """吃豆人执行动作"""
        direction = self.get_pacman_action(action)
        return self.step(direction)
    
    def step_ghost(self, ghost_index: int, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """幽灵执行动作"""
        direction = self.get_ghost_action(ghost_index, action)
        
        # 移动幽灵
        self.game.move_ghost(ghost_index, direction)
        
        # 检查碰撞
        collision_info = self.game._check_ghost_collision()
        
        # 更新游戏状态
        self.game._update_ghosts()
        self.game._update_fruits()
        
        # 检查游戏结束
        done = self.game._check_game_end()
        
        # 计算奖励（幽灵的奖励与吃豆人相反）
        reward = -self.game._calculate_reward(False, collision_info)
        
        return self._get_observation(), reward, done, False, {
            'ghost_index': ghost_index,
            'collision': collision_info,
            'score': self.game.score
        }
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        info = {
            'pacman_position': self.game.get_pacman_position(),
            'ghost_positions': self.game.get_ghost_positions(),
            'dots_count': self.game.maze.get_dots_count(),
            'power_dots_count': len(self.game.maze.power_dots),
            'lives': self.game.lives,
            'level': self.game.level,
            'game_mode': self.game.game_mode
        }
        return info
    
    def get_pacman_state(self) -> Dict[str, Any]:
        """获取吃豆人状态"""
        if not self.game.pacman:
            return {}
        
        return {
            'position': self.game.pacman.get_position(),
            'direction': self.game.pacman.direction,
            'score': self.game.pacman.score,
            'lives': self.game.pacman.lives,
            'power_mode': self.game.pacman.is_power_mode(),
            'power_timer': self.game.pacman.power_timer
        }
    
    def get_ghost_states(self) -> List[Dict[str, Any]]:
        """获取所有幽灵状态"""
        ghost_states = []
        for ghost in self.game.ghosts:
            if ghost:
                ghost_states.append({
                    'position': ghost.get_position(),
                    'direction': ghost.direction,
                    'ghost_type': ghost.ghost_type,
                    'frightened': ghost.is_frightened(),
                    'frightened_timer': ghost.frightened_timer,
                    'color': ghost.color
                })
        return ghost_states
    
    def set_game_mode(self, mode: str):
        """设置游戏模式"""
        self.game.set_game_mode(mode)
    
    def get_maze_info(self) -> Dict[str, Any]:
        """获取迷宫信息"""
        maze = self.game.maze
        return {
            'width': maze.width,
            'height': maze.height,
            'walls': list(maze.walls),
            'dots': list(maze.dots),
            'power_dots': list(maze.power_dots),
            'empty_cells': list(maze.empty_cells)
        }
    
    def render_text(self) -> str:
        """文本渲染"""
        return self.game.render()
    
    def render_ascii(self) -> str:
        """ASCII艺术渲染"""
        maze = self.game.maze
        result = []
        
        for y in range(maze.height):
            row = ""
            for x in range(maze.width):
                cell = " "
                
                # 墙壁
                if maze.is_wall(x, y):
                    cell = "█"
                # 豆子
                elif maze.has_dot(x, y):
                    cell = "·"
                # 能量豆
                elif maze.has_power_dot(x, y):
                    cell = "●"
                # 吃豆人
                elif (self.game.pacman and self.game.pacman_alive and 
                      self.game.pacman.get_position() == (x, y)):
                    cell = "C"
                # 幽灵
                else:
                    for ghost in self.game.ghosts:
                        if ghost and ghost.get_position() == (x, y):
                            if ghost.is_frightened():
                                cell = "F"
                            else:
                                cell = "G"
                            break
                
                row += cell
            result.append(row)
        
        # 添加状态信息
        result.append(f"分数: {self.game.score} | 生命: {self.game.lives} | 关卡: {self.game.level}")
        result.append(f"豆子: {maze.get_dots_count()} | 能量豆: {len(maze.power_dots)}")
        
        return "\n".join(result) 

    def set_ghost_ai_type(self, ai_type: str):
        """设置幽灵AI类型"""
        self.game.set_ghost_ai_type(ai_type)
    
    def set_ghost_aggression_level(self, level: float):
        """设置幽灵攻击性等级"""
        self.game.set_ghost_aggression_level(level) 