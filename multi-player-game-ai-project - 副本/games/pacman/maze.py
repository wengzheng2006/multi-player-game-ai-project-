"""
吃豆人地图系统
定义迷宫生成、路径查找等功能
"""

import random
from typing import List, Tuple, Set, Optional
from collections import deque

class Maze:
    """迷宫类"""
    
    def __init__(self, width: int = 28, height: int = 31):
        self.width = width
        self.height = height
        self.grid = []
        self.dots = set()  # 普通豆子位置
        self.power_dots = set()  # 能量豆位置
        self.walls = set()  # 墙壁位置
        self.empty_cells = set()  # 空位置
        
        self._generate_maze()
        self._place_dots()
    
    def _generate_maze(self):
        """生成迷宫"""
        # 初始化网格
        self.grid = [['#' for _ in range(self.width)] for _ in range(self.height)]
        
        # 重新设计的迷宫布局 - 确保没有死胡同，所有路径连通
        maze_layout = [
            "############################",
            "#..#....#....##.....#....#.#",
            "#.#.##.###.#.##.###.#.##.#.#",
            "#.#..#.....#....#.....#..#.#",
            "#.#.####.######.####.###...#",
            "#......#..............#....#",
            "###.##.##.########.##.#..###",
            "#....#..#..######..#..#...#",
            "#.##.##.##.####..#.##.#..##",
            "#..#....#..#..#.......#...#",
            "##.####.##.##...#.##.##.#.#",
            "#......#..............#....#",
            "#.##.##.##.########.#.....#",
            "#..#..#..#..##..#..#......#",
            "##.##.##.##.##..##.#......#",
            "#....#....#....#....#.....#",
            "#.####.####.##.####.#.....#",
            "#......#..................#",
            "#.##.##.##.########.##....#",
            "#..#..#..#..##..#..#..#...#",
            "##.##.##.##.##..##.##.##.##",
            "#....#....#....#....#....#.#",
            "#.####.####.##.####.#####..",
            "#..........................#",
            "#.#.##.###.#.##.###.#.##.#.#",
            "#..#....#....##.....#....#.#",
            "#.####.#####.##.#####.####.#",
            "#..........................#",
            "#..#....#....##.....#....#.#",
            "#..........................#",
            "############################"
        ]
        
        # 补齐到31行
        while len(maze_layout) < 31:
            maze_layout.insert(-1, "#..........................#")
        
        # 应用迷宫布局
        for y, row in enumerate(maze_layout):
            if y < self.height:
                for x, cell in enumerate(row):
                    if x < self.width:
                        if cell == '#':
                            self.grid[y][x] = '#'
                            self.walls.add((x, y))
                        else:
                            self.grid[y][x] = ' '
                            self.empty_cells.add((x, y))
                    # 确保最右侧为墙壁
                    if x == self.width - 1:
                        self.grid[y][x] = '#'
                        self.walls.add((x, y))
        
        # 修复死胡同 - 确保所有空位置都至少有两个出口
        self._fix_dead_ends()
    
    def _fix_dead_ends(self):
        """修复死胡同"""
        changed = True
        while changed:
            changed = False
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y][x] == ' ':
                        # 检查是否为死胡同（只有一个出口）
                        exits = 0
                        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            new_x, new_y = x + dx, y + dy
                            if self.is_valid_position(new_x, new_y) and not self.is_wall(new_x, new_y):
                                exits += 1
                        
                        # 如果只有一个出口，尝试添加另一个出口
                        if exits <= 1:
                            # 尝试在某个方向添加出口
                            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                new_x, new_y = x + dx, y + dy
                                if (self.is_valid_position(new_x, new_y) and 
                                    self.is_wall(new_x, new_y) and
                                    self._can_break_wall(new_x, new_y)):
                                    # 打破墙壁
                                    self.grid[new_y][new_x] = ' '
                                    self.walls.remove((new_x, new_y))
                                    self.empty_cells.add((new_x, new_y))
                                    changed = True
                                    break
    
    def _can_break_wall(self, x: int, y: int) -> bool:
        """检查是否可以打破墙壁（不会创建新的死胡同）"""
        # 检查打破墙壁后是否会导致新的死胡同
        temp_grid = [row[:] for row in self.grid]
        temp_grid[y][x] = ' '
        
        # 检查相邻位置是否变成死胡同
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_x, new_y = x + dx, y + dy
            if (self.is_valid_position(new_x, new_y) and 
                temp_grid[new_y][new_x] == ' '):
                # 检查这个位置是否变成死胡同
                exits = 0
                for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    check_x, check_y = new_x + ddx, new_y + ddy
                    if (self.is_valid_position(check_x, check_y) and 
                        temp_grid[check_y][check_x] == ' '):
                        exits += 1
                if exits <= 1:
                    return False
        
        return True
    
    def _place_dots(self):
        """放置豆子"""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == ' ':
                    # 能量豆位置（四个角落）- 确保位置可达
                    if ((x == 1 and y == 1) or 
                        (x == self.width-2 and y == 1) or
                        (x == 1 and y == self.height-2) or
                        (x == self.width-2 and y == self.height-2)):
                        # 检查位置是否为空且可达
                        if self.is_empty(x, y) and self._is_accessible(x, y):
                            self.power_dots.add((x, y))
                    else:
                        # 普通豆子
                        self.dots.add((x, y))
    
    def _is_accessible(self, x: int, y: int) -> bool:
        """检查位置是否可达"""
        # 检查是否有至少一个相邻的空位置
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y) and not self.is_wall(new_x, new_y):
                return True
        return False
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否有效"""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_wall(self, x: int, y: int) -> bool:
        """检查是否为墙壁"""
        if not self.is_valid_position(x, y):
            return True
        return self.grid[y][x] == '#'
    
    def is_empty(self, x: int, y: int) -> bool:
        """检查是否为空位置"""
        if not self.is_valid_position(x, y):
            return False
        return self.grid[y][x] == ' '
    
    def has_dot(self, x: int, y: int) -> bool:
        """检查是否有豆子"""
        return (x, y) in self.dots
    
    def has_power_dot(self, x: int, y: int) -> bool:
        """检查是否有能量豆"""
        return (x, y) in self.power_dots
    
    def remove_dot(self, x: int, y: int):
        """移除豆子"""
        pos = (x, y)
        if pos in self.dots:
            self.dots.remove(pos)
    
    def remove_power_dot(self, x: int, y: int):
        """移除能量豆"""
        pos = (x, y)
        if pos in self.power_dots:
            self.power_dots.remove(pos)
    
    def get_dots_count(self) -> int:
        """获取豆子总数"""
        return len(self.dots) + len(self.power_dots)
    
    def get_available_directions(self, x: int, y: int) -> List[Tuple[int, int]]:
        """获取可用方向"""
        directions = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 上下左右
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y) and not self.is_wall(new_x, new_y):
                directions.append((dx, dy))
        return directions
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """使用BFS查找路径"""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (x, y), path = queue.popleft()
            
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (new_pos == end):
                    return path + [new_pos]
                
                if (self.is_valid_position(new_x, new_y) and 
                    not self.is_wall(new_x, new_y) and 
                    new_pos not in visited):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return None
    
    def find_nearest_dot(self, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """查找最近的豆子"""
        if not self.dots and not self.power_dots:
            return None
        
        queue = deque([start])
        visited = {start}
        
        while queue:
            x, y = queue.popleft()
            
            # 检查当前位置是否有豆子
            if (x, y) in self.dots or (x, y) in self.power_dots:
                return (x, y)
            
            # 检查相邻位置
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (self.is_valid_position(new_x, new_y) and 
                    not self.is_wall(new_x, new_y) and 
                    new_pos not in visited):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return None
    
    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """计算两点间的曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_random_empty_position(self) -> Tuple[int, int]:
        """获取随机空位置"""
        empty_positions = list(self.empty_cells)
        if empty_positions:
            return random.choice(empty_positions)
        return (1, 1)  # 默认位置
    
    def get_ghost_spawn_positions(self) -> List[Tuple[int, int]]:
        """获取幽灵出生位置"""
        # 幽灵出生在迷宫中央区域
        center_x = self.width // 2
        center_y = self.height // 2
        
        spawn_positions = [
            (center_x - 1, center_y - 1),
            (center_x + 1, center_y - 1),
            (center_x - 1, center_y + 1),
            (center_x + 1, center_y + 1)
        ]
        
        # 过滤掉墙壁位置
        valid_positions = []
        for pos in spawn_positions:
            if self.is_valid_position(*pos) and not self.is_wall(*pos):
                valid_positions.append(pos)
        
        return valid_positions
    
    def get_pacman_spawn_position(self) -> Tuple[int, int]:
        """获取吃豆人出生位置"""
        # 吃豆人出生在迷宫底部中央
        spawn_x = self.width // 2
        spawn_y = self.height - 2
        
        if self.is_valid_position(spawn_x, spawn_y) and not self.is_wall(spawn_x, spawn_y):
            return (spawn_x, spawn_y)
        
        # 如果默认位置不可用，寻找最近的空位置
        return self.get_random_empty_position()
    
    def render(self) -> str:
        """渲染迷宫"""
        result = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell == ' ':
                    if (x, y) in self.power_dots:
                        row += 'o'
                    elif (x, y) in self.dots:
                        row += '.'
                    else:
                        row += ' '
                else:
                    row += cell
            result.append(row)
        return '\n'.join(result) 