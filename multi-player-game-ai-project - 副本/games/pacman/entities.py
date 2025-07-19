"""
吃豆人游戏实体
定义游戏中的各种实体类
"""

import random
from typing import List, Tuple, Optional
from enum import Enum

class Direction(Enum):
    """方向枚举"""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    NONE = (0, 0)

class Entity:
    """游戏实体基类"""
    
    def __init__(self, x: int, y: int, symbol: str = ' '):
        self.x = x
        self.y = y
        self.symbol = symbol
        self.direction = Direction.NONE
    
    def move(self, dx: int, dy: int, maze):
        """移动实体"""
        new_x = self.x + dx
        new_y = self.y + dy
        
        if maze.is_valid_position(new_x, new_y) and not maze.is_wall(new_x, new_y):
            self.x = new_x
            self.y = new_y
            return True
        return False
    
    def get_position(self) -> Tuple[int, int]:
        """获取位置"""
        return (self.x, self.y)
    
    def set_position(self, x: int, y: int):
        """设置位置"""
        self.x = x
        self.y = y

class Pacman(Entity):
    """吃豆人类"""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y, 'P')
        self.score = 0
        self.lives = 3
        self.power_mode = False
        self.power_timer = 0
        self.direction = Direction.RIGHT
    
    def move(self, direction: Direction, maze) -> bool:
        """移动吃豆人"""
        if direction != Direction.NONE:
            self.direction = direction
        
        dx, dy = self.direction.value
        return super().move(dx, dy, maze)
    
    def eat_dot(self, dots, power_dots):
        """吃豆子"""
        pos = self.get_position()
        
        # 检查普通豆子
        if pos in dots:
            dots.remove(pos)
            self.score += 10
            return True
        
        # 检查能量豆
        if pos in power_dots:
            power_dots.remove(pos)
            self.score += 50
            self.power_mode = True
            self.power_timer = 20  # 能量模式持续20步
            return True
        
        return False
    
    def update_power_mode(self):
        """更新能量模式"""
        if self.power_mode:
            self.power_timer -= 1
            if self.power_timer <= 0:
                self.power_mode = False
    
    def is_power_mode(self) -> bool:
        """是否处于能量模式"""
        return self.power_mode

class Ghost(Entity):
    """幽灵类"""
    
    def __init__(self, x: int, y: int, ghost_type: str = 'ghost'):
        super().__init__(x, y, 'G')
        self.ghost_type = ghost_type
        self.direction = Direction.LEFT
        self.frightened = False
        self.frightened_timer = 0
        self.target = None
        self.color = self._get_ghost_color(ghost_type)
    
    def _get_ghost_color(self, ghost_type: str) -> str:
        """获取幽灵颜色"""
        colors = {
            'red': 'red',
            'pink': 'pink', 
            'blue': 'blue',
            'orange': 'orange'
        }
        return colors.get(ghost_type, 'white')
    
    def move(self, direction: Direction, maze) -> bool:
        """移动幽灵"""
        if direction != Direction.NONE:
            self.direction = direction
        
        dx, dy = self.direction.value
        return super().move(dx, dy, maze)
    
    def set_frightened(self, duration: int = 20):
        """设置恐惧状态"""
        self.frightened = True
        self.frightened_timer = duration
        self.symbol = 'F'  # 恐惧状态的符号
    
    def update_frightened(self):
        """更新恐惧状态"""
        if self.frightened:
            self.frightened_timer -= 1
            if self.frightened_timer <= 0:
                self.frightened = False
                self.symbol = 'G'
    
    def is_frightened(self) -> bool:
        """是否处于恐惧状态"""
        return self.frightened
    
    def set_target(self, target: Tuple[int, int]):
        """设置目标位置"""
        self.target = target
    
    def get_target(self) -> Optional[Tuple[int, int]]:
        """获取目标位置"""
        return self.target

class Dot:
    """豆子类"""
    
    def __init__(self, x: int, y: int, is_power: bool = False):
        self.x = x
        self.y = y
        self.is_power = is_power
        self.symbol = 'o' if is_power else '.'
    
    def get_position(self) -> Tuple[int, int]:
        """获取位置"""
        return (self.x, self.y)
    
    def __eq__(self, other):
        if isinstance(other, tuple):
            return (self.x, self.y) == other
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))

class Fruit(Entity):
    """水果类"""
    
    def __init__(self, x: int, y: int, fruit_type: str = 'cherry'):
        super().__init__(x, y, 'F')
        self.fruit_type = fruit_type
        self.points = self._get_fruit_points(fruit_type)
        self.active = False
        self.active_timer = 0
    
    def _get_fruit_points(self, fruit_type: str) -> int:
        """获取水果分数"""
        points = {
            'cherry': 100,
            'strawberry': 300,
            'orange': 500,
            'apple': 700,
            'melon': 1000,
            'galaxian': 2000,
            'bell': 3000,
            'key': 5000
        }
        return points.get(fruit_type, 100)
    
    def activate(self, duration: int = 30):
        """激活水果"""
        self.active = True
        self.active_timer = duration
    
    def update(self):
        """更新水果状态"""
        if self.active:
            self.active_timer -= 1
            if self.active_timer <= 0:
                self.active = False
    
    def is_active(self) -> bool:
        """水果是否激活"""
        return self.active 