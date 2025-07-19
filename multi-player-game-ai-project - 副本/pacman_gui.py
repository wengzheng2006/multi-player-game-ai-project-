"""
吃豆人游戏GUI
支持单人对战AI和双人对战模式
"""

import pygame
import sys
import time
import os
from typing import Dict, List, Tuple, Optional
from games.pacman import PacmanEnv, Direction
from agents.human.human_agent import HumanAgent

class PacmanGUI:
    """吃豆人游戏GUI"""
    
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pacman Game")
        
        # 游戏状态
        self.ai_ghost = None
        self.game_over = False
        self.winner = None
        self.running = True  # 添加running属性
        
        # 连续移动状态
        self.pacman_continuous_moving = False
        self.pacman_target_direction = None
        
        # 移动速度控制
        self.pacman_move_counter = 0
        self.pacman_move_interval = 3  # 每3帧移动一次，5步/秒
        self.ghost_move_counter = 0
        self.ghost_move_interval = 3  # 每3帧移动一次，与吃豆人速度一致
        
        # 输入处理 - 简化的输入处理
        self.keys_pressed = set()
        self.continuous_movement = True  # 启用连续移动
        
        # 幽灵AI设置
        self.ghost_ai_type = 'aggressive'  # 'aggressive', 'strategic', 'coordinated'
        self.ghost_aggression_level = 2.0  # 提高攻击性等级到2.0
        
        # 创建游戏环境
        game_config = {
            'ghost_ai_type': self.ghost_ai_type,
            'ghost_aggression_level': self.ghost_aggression_level
        }
        self.env = PacmanEnv(game_config)
        # 重置游戏状态，确保游戏正常开始
        self.env.game.reset()
        
        # 确保幽灵AI已正确初始化
        if not self.env.game.ghost_ais:
            print("警告：幽灵AI未正确初始化，重新设置...")
            self.env.game.set_ghost_ai_type(self.ghost_ai_type)
            self.env.game.set_ghost_aggression_level(self.ghost_aggression_level)
        
        # 颜色定义
        self.colors = {
            'background': (0, 0, 0),      # 黑色背景
            'wall': (0, 0, 255),          # 蓝色墙壁
            'dot': (255, 255, 255),       # 白色豆子
            'power_dot': (255, 255, 0),   # 黄色能量豆
            'pacman': (255, 255, 0),      # 黄色吃豆人
            'ghost_easy': (0, 255, 0),      # 绿色 - 简单
            'ghost_normal': (255, 0, 0),    # 红色 - 普通
            'ghost_hard': (255, 0, 255),    # 紫色 - 困难
            'ghost_frightened': (0, 0, 255),  # 蓝色恐惧状态
            'fruit': (255, 165, 0),       # 橙色水果
            'ui_bg': (50, 50, 50),        # 深灰色UI背景
            'ui_border': (100, 100, 100), # 浅灰色UI边框
            'text': (255, 255, 255),      # 白色文字
            'highlight': (255, 255, 0),   # 黄色高亮
            'button': (70, 70, 70),       # 按钮背景
            'button_hover': (90, 90, 90), # 按钮悬停
        }
        
        # 字体 - 使用中文字体
        self.font_path = self._get_chinese_font()
        if self.font_path:
            self.title_font = pygame.font.Font(self.font_path, 36)
            self.info_font = pygame.font.Font(self.font_path, 24)
            self.small_font = pygame.font.Font(self.font_path, 18)
        else:
            # 备用字体
            self.title_font = pygame.font.SysFont('arial', 36)
            self.info_font = pygame.font.SysFont('arial', 24)
            self.small_font = pygame.font.SysFont('arial', 18)
        
        # 迷宫渲染参数
        self.cell_size = 20
        self.maze_offset_x = 350
        self.maze_offset_y = 50
        
        # 初始化AI幽灵
        self._init_ai_ghost()
    
    def _get_chinese_font(self):
        """获取中文字体路径"""
        # 尝试不同系统的中文字体
        font_paths = [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            # Windows
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path

        # 如果没有找到中文字体，使用pygame默认字体
        return None
    
    def _init_ai_ghost(self):
        """初始化AI幽灵 - 现在使用游戏内置的幽灵AI"""
        # 不再需要HeuristicBot，使用游戏内置的幽灵AI
        pass
    
    def run(self):
        """运行游戏主循环"""
        clock = pygame.time.Clock()
        
        while self.running:
            # 处理事件
            self._handle_events()
            
            # 更新游戏状态
            if not self.game_over:
                self._update_game()
            
            # 渲染
            self._render()
            
            # 控制帧率
            clock.tick(15)  # 提高帧率到15 FPS
        
        pygame.quit()
        sys.exit()
    
    def _handle_events(self):
        """处理事件 - 简化版"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown_simple(event.key)
            elif event.type == pygame.KEYUP:
                self._handle_keyup(event.key)

    def _handle_keydown_simple(self, key):
        """简化的按键处理 - 完全移除冷却机制"""
        # 功能键处理
        if key == pygame.K_ESCAPE:
            self.running = False
            return
        # 游戏进行时的移动按键处理
        if not self.game_over:
            self._handle_movement_key(key)

    def _handle_movement_key(self, key):
        """处理移动按键 - 极简版本"""
        # 只有方向键控制吃豆人
        if key == pygame.K_UP:
            self._start_pacman_continuous_movement(Direction.UP)
        elif key == pygame.K_DOWN:
            self._start_pacman_continuous_movement(Direction.DOWN)
        elif key == pygame.K_LEFT:
            self._start_pacman_continuous_movement(Direction.LEFT)
        elif key == pygame.K_RIGHT:
            self._start_pacman_continuous_movement(Direction.RIGHT)
        # 其他所有按键都忽略，不做任何处理

    def _handle_keyup(self, key):
        """处理按键释放"""
        if key in self.keys_pressed:
            self.keys_pressed.remove(key)
    
    def _start_pacman_continuous_movement(self, direction: Direction):
        """开始吃豆人连续移动"""
        self.pacman_continuous_moving = True
        self.pacman_target_direction = direction
        # 立即执行一次移动
        self._move_pacman(direction)
    

    
    def _check_pacman_wall_collision(self, direction: Direction) -> bool:
        """检查吃豆人是否会撞墙"""
        if not self.env.game.pacman:
            return True
        
        x, y = self.env.game.pacman.get_position()
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        # 检查新位置是否是墙壁
        return self.env.game.maze.is_wall(new_x, new_y)
    

    
    def _move_pacman(self, direction: Direction):
        """移动吃豆人"""
        if not self.env.game.pacman_alive:
            return
        
        observation, reward, done, truncated, info = self.env.step_pacman(direction)
        
        if done:
            self._handle_game_end()
    
    def _move_ghost(self, ghost_index: int, direction: Direction):
        """移动幽灵"""
        if ghost_index >= len(self.env.game.ghosts):
            return
        
        observation, reward, done, truncated, info = self.env.step_ghost(ghost_index, direction)
        
        if done:
            self._handle_game_end()
    
    def _update_continuous_movement(self):
        """更新连续移动状态"""
        # 处理吃豆人连续移动
        if self.pacman_continuous_moving and self.pacman_target_direction:
            # 速度控制：每3帧移动一次，5步/秒
            self.pacman_move_counter += 1
            if self.pacman_move_counter >= self.pacman_move_interval:
                self.pacman_move_counter = 0
                # 检查是否会撞墙
                if self._check_pacman_wall_collision(self.pacman_target_direction):
                    # 撞墙了，停止连续移动
                    self.pacman_continuous_moving = False
                    self.pacman_target_direction = None
                else:
                    # 继续移动
                    self._move_pacman(self.pacman_target_direction)
    
    def _update_game(self):
        """更新游戏状态"""
        # 处理连续移动
        self._update_continuous_movement()
        
        # 更新吃豆人能量模式
        if self.env.game.pacman:
            self.env.game.pacman.update_power_mode()
        
        # 更新幽灵 - 确保幽灵主动移动
        self._update_ghosts()
        
        # 检查游戏结束
        if self.env.game.is_terminal():
            self._handle_game_end()
    
    def _update_ghosts(self):
        """更新幽灵 - 确保幽灵主动移动追逐玩家"""
        # 速度控制：每2帧移动一次，提高幽灵响应速度
        self.ghost_move_counter += 1
        if self.ghost_move_counter >= 2:  # 减少间隔，提高幽灵移动频率
            self.ghost_move_counter = 0
            
            # 获取吃豆人位置
            pacman_pos = self.env.game.get_pacman_position()
            if not pacman_pos:
                return
            
            # 更新所有幽灵
            for i, ghost in enumerate(self.env.game.ghosts):
                if ghost and self.env.game.ghost_ais and i < len(self.env.game.ghost_ais):
                    ghost_ai = self.env.game.ghost_ais[i]
                    if ghost_ai:
                        # 获取幽灵AI的动作
                        pacman_power = self.env.game.pacman.is_power_mode() if self.env.game.pacman else False
                        action = ghost_ai.get_action(pacman_pos, pacman_power)
                        
                        # 执行幽灵移动
                        if action and action != Direction.NONE:
                            # 检查移动是否有效
                            dx, dy = action.value
                            gx, gy = ghost.get_position()
                            new_x, new_y = gx + dx, gy + dy
                            
                            # 检查新位置是否有效
                            if (self.env.game.maze.is_valid_position(new_x, new_y) and 
                                not self.env.game.maze.is_wall(new_x, new_y)):
                                ghost.move(action, self.env.game.maze)
            
            # 检查幽灵与吃豆人的碰撞
            collision_info = self.env.game._check_ghost_collision()
            
            # 更新幽灵状态
            self.env.game._update_ghosts()
            
            # 如果发生碰撞，处理游戏逻辑
            if collision_info.get('collision', False):
                if collision_info.get('pacman_power', False):
                    # 吃豆人处于能量模式，幽灵被吃掉
                    print("幽灵被吃掉！")
                else:
                    # 吃豆人被幽灵吃掉
                    print("吃豆人被幽灵吃掉！")
                    self.env.game.lives -= 1
                    if self.env.game.lives <= 0:
                        self._handle_game_end()
    
    def _handle_game_end(self):
        """处理游戏结束"""
        self.game_over = True
        winner = self.env.game.get_winner()
        if winner == 1:
            self.winner = "Pacman Wins!"
        elif winner == 2:
            self.winner = "Ghost Wins!"
        else:
            self.winner = "Draw!"
    

    

    

    

    
    def _render(self):
        """渲染游戏"""
        # 清屏
        self.screen.fill(self.colors['background'])
        
        # 渲染迷宫
        self._render_maze()
        
        # 渲染UI
        self._render_ui()
        
        # 更新显示
        pygame.display.flip()
    
    def _render_maze(self):
        """渲染迷宫"""
        maze = self.env.game.maze
        
        for y in range(maze.height):
            for x in range(maze.width):
                screen_x = self.maze_offset_x + x * self.cell_size
                screen_y = self.maze_offset_y + y * self.cell_size
                
                # 绘制墙壁
                if maze.is_wall(x, y):
                    pygame.draw.rect(self.screen, self.colors['wall'],
                                   (screen_x, screen_y, self.cell_size, self.cell_size))
                
                # 绘制豆子
                elif maze.has_dot(x, y):
                    center_x = screen_x + self.cell_size // 2
                    center_y = screen_y + self.cell_size // 2
                    pygame.draw.circle(self.screen, self.colors['dot'],
                                     (center_x, center_y), 3)
                
                # 绘制能量豆
                elif maze.has_power_dot(x, y):
                    center_x = screen_x + self.cell_size // 2
                    center_y = screen_y + self.cell_size // 2
                    pygame.draw.circle(self.screen, self.colors['power_dot'],
                                     (center_x, center_y), 8)
        
        # 绘制吃豆人
        if self.env.game.pacman and self.env.game.pacman_alive:
            px, py = self.env.game.pacman.get_position()
            screen_x = self.maze_offset_x + px * self.cell_size
            screen_y = self.maze_offset_y + py * self.cell_size
            center_x = screen_x + self.cell_size // 2
            center_y = screen_y + self.cell_size // 2
            
            # 绘制吃豆人（圆形）
            pygame.draw.circle(self.screen, self.colors['pacman'],
                             (center_x, center_y), self.cell_size // 2)
            
            # 绘制吃豆人眼睛
            eye_offset = 3
            pygame.draw.circle(self.screen, (0, 0, 0),
                             (center_x - eye_offset, center_y - eye_offset), 2)
            pygame.draw.circle(self.screen, (0, 0, 0),
                             (center_x + eye_offset, center_y - eye_offset), 2)
        
        # 绘制幽灵
        for ghost in self.env.game.ghosts:
            if ghost:
                gx, gy = ghost.get_position()
                screen_x = self.maze_offset_x + gx * self.cell_size
                screen_y = self.maze_offset_y + gy * self.cell_size
                center_x = screen_x + self.cell_size // 2
                center_y = screen_y + self.cell_size // 2
                
                # 选择颜色
                if ghost.is_frightened():
                    color = self.colors['ghost_frightened']
                else:
                    # 使用普通颜色
                    color = self.colors['ghost_normal']
                
                # 绘制幽灵（圆形）
                pygame.draw.circle(self.screen, color,
                                 (center_x, center_y), self.cell_size // 2)
                
                # 绘制幽灵眼睛
                eye_offset = 3
                pygame.draw.circle(self.screen, (255, 255, 255),
                                 (center_x - eye_offset, center_y - eye_offset), 2)
                pygame.draw.circle(self.screen, (255, 255, 255),
                                 (center_x + eye_offset, center_y - eye_offset), 2)
        
        # 绘制水果
        for fruit in self.env.game.fruits:
            if fruit.is_active():
                fx, fy = fruit.get_position()
                screen_x = self.maze_offset_x + fx * self.cell_size
                screen_y = self.maze_offset_y + fy * self.cell_size
                center_x = screen_x + self.cell_size // 2
                center_y = screen_y + self.cell_size // 2
                
                pygame.draw.circle(self.screen, self.colors['fruit'],
                                 (center_x, center_y), self.cell_size // 3)
    
    def _render_ui(self):
        """渲染UI"""
        # 左侧信息面板
        self._render_info_panel()
        
        # 右侧控制说明面板
        self._render_control_panel()
        
        # 游戏结束覆盖层
        if self.game_over:
            self._render_game_over_overlay()
    
    def _render_info_panel(self):
        """渲染信息面板"""
        panel_x = 20
        panel_y = 20
        panel_width = 300
        panel_height = 250  # 增加高度以容纳更多信息
        
        # 绘制面板背景
        pygame.draw.rect(self.screen, self.colors['ui_bg'],
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['ui_border'],
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # 游戏标题
        title_surface = self.title_font.render("吃豆人游戏", True, self.colors['highlight'])
        self.screen.blit(title_surface, (panel_x + 10, panel_y + 10))
        
        # 游戏信息
        info_y = panel_y + 60
        info_texts = [
            f"得分: {self.env.game.score}",
            f"生命: {self.env.game.lives}",
            f"关卡: {self.env.game.level}",
            f"豆子: {self.env.game.maze.get_dots_count()}",
            f"能量豆: {len(self.env.game.maze.power_dots)}"
        ]
        
        for i, text in enumerate(info_texts):
            surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(surface, (panel_x + 10, info_y + i * 25))
        
        # 幽灵AI信息
        ai_y = info_y + len(info_texts) * 25 + 10
        
        # AI类型
        ai_type_map = {
            'aggressive': '攻击型',
            'strategic': '战略型', 
            'coordinated': '协调型'
        }
        ai_type_text = f"幽灵AI: {ai_type_map.get(self.ghost_ai_type, self.ghost_ai_type)}"
        ai_type_surface = self.small_font.render(ai_type_text, True, self.colors['highlight'])
        self.screen.blit(ai_type_surface, (panel_x + 10, ai_y))
        
        # 攻击性等级
        aggression_text = f"攻击性: {self.ghost_aggression_level:.1f}"
        aggression_surface = self.small_font.render(aggression_text, True, self.colors['text'])
        self.screen.blit(aggression_surface, (panel_x + 10, ai_y + 25))
        
        # 速度信息
        speed_y = ai_y + 50
        speed_text = f"速度: {15 // self.pacman_move_interval} 步/秒"
        speed_surface = self.small_font.render(speed_text, True, self.colors['text'])
        self.screen.blit(speed_surface, (panel_x + 10, speed_y))
        
        # 连续移动状态
        status_y = speed_y + 25
        if self.pacman_continuous_moving:
            direction_map = {'UP': '上', 'DOWN': '下', 'LEFT': '左', 'RIGHT': '右'}
            direction_name = direction_map.get(self.pacman_target_direction.name, self.pacman_target_direction.name) if self.pacman_target_direction else "未知"
            status_text = f"吃豆人: 移动中 {direction_name}..."
            status_color = self.colors['highlight']
        else:
            status_text = "吃豆人: 就绪"
            status_color = self.colors['text']
        
        status_surface = self.small_font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (panel_x + 10, status_y))
    
    def _render_control_panel(self):
        """渲染控制说明面板"""
        panel_x = self.width - 350
        panel_y = 20
        panel_width = 330
        panel_height = 400
        
        # 绘制面板背景
        pygame.draw.rect(self.screen, self.colors['ui_bg'],
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['ui_border'],
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # 控制说明标题
        title_surface = self.small_font.render("控制说明", True, self.colors['highlight'])
        self.screen.blit(title_surface, (panel_x + 10, panel_y + 10))
        
        # 基本控制
        control_y = panel_y + 50
        basic_controls = [
            "方向键控制吃豆人",
            "ESC - 退出游戏",
            "游戏进行时只接受移动按键和功能键"
        ]
        
        for i, control in enumerate(basic_controls):
            surface = self.small_font.render(control, True, self.colors['text'])
            self.screen.blit(surface, (panel_x + 10, control_y + i * 20))
        

        
        # 游戏规则说明
        rules_y = panel_y + 280
        rules_title = self.small_font.render("游戏规则", True, self.colors['highlight'])
        self.screen.blit(rules_title, (panel_x + 10, rules_y))
        
        rules = [
            "吃豆子获得分数",
            "吃能量豆可以吃幽灵",
            "避开幽灵否则失去生命",
            "吃完所有豆子获胜"
        ]
        
        for i, rule in enumerate(rules):
            surface = self.small_font.render(rule, True, self.colors['text'])
            self.screen.blit(surface, (panel_x + 10, rules_y + 25 + i * 15))
    
    def _render_game_over_overlay(self):
        """渲染游戏结束覆盖层"""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # 游戏结束文本
        if self.winner:
            # 翻译获胜信息
            winner_map = {
                "Pacman Wins!": "吃豆人获胜！",
                "Ghost Wins!": "幽灵获胜！",
                "Draw!": "平局！"
            }
            winner_text = winner_map.get(self.winner, self.winner)
            text_surface = self.title_font.render(winner_text, True, self.colors['highlight'])
            text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(text_surface, text_rect)
        

        
        # 退出提示
        exit_text = self.small_font.render("按ESC键退出", True, self.colors['text'])
        exit_rect = exit_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
        self.screen.blit(exit_text, exit_rect)
    


def main():
    """主函数"""
    gui = PacmanGUI()
    gui.run()

if __name__ == "__main__":
    main() 