"""
多游戏图形界面
支持五子棋和贪吃蛇的人机对战，修复中文显示问题
"""

import pygame
import sys
import time
import os
from typing import Optional, Tuple, Dict, Any
from games.gomoku import GomokuGame, GomokuEnv
from games.snake import SnakeGame, SnakeEnv
from agents import RandomBot, MinimaxBot, MCTSBot, HumanAgent, SnakeAI, SmartSnakeAI, AdvancedSnakeAI
import config

# 颜色定义
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "BROWN": (139, 69, 19),
    "LIGHT_BROWN": (205, 133, 63),
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "GRAY": (128, 128, 128),
    "LIGHT_GRAY": (211, 211, 211),
    "DARK_GRAY": (64, 64, 64),
    "YELLOW": (255, 255, 0),
    "ORANGE": (255, 165, 0),
    "PURPLE": (128, 0, 128),
    "CYAN": (0, 255, 255),
}


class MultiGameGUI:
    """多游戏图形界面"""

    def __init__(self):
        # 初始化pygame
        pygame.init()

        # 设置中文字体
        self.font_path = self._get_chinese_font()
        self.font_large = pygame.font.Font(self.font_path, 28)
        self.font_medium = pygame.font.Font(self.font_path, 20)
        self.font_small = pygame.font.Font(self.font_path, 16)

        self.window_width = 900
        self.window_height = 700
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("多游戏AI对战平台")
        self.clock = pygame.time.Clock()

        # 游戏状态
        self.current_game = "gomoku"  # "gomoku" 或 "snake"
        self.env = None
        self.human_agent = None
        self.ai_agent = None
        self.current_agent = None
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.thinking = False
        self.selected_ai = "RandomBot"
        self.paused = False

        # 贪吃蛇专用变量
        self.last_key_time = 0
        self.key_cooldown = 0.2  # 200ms冷却时间
        self.last_processed_key = None
        self.last_processed_time = 0
        self.auto_move_interval = 0.3  # 0.3秒自动移动一次
        self.last_auto_move_time = 0
        self.human_direction = (0, 1)  # 人类蛇的默认方向（向右）
        self.ai_direction = (0, -1)   # AI蛇的默认方向（向左）

        # UI元素
        self.buttons = self._create_buttons()
        self.cell_size = 25
        self.margin = 50

        # 游戏计时
        self.last_update = time.time()
        self.update_interval = 0.3  # 贪吃蛇更新间隔

        self._switch_game("gomoku")

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

    def _create_buttons(self) -> Dict[str, Dict[str, Any]]:
        """创建UI按钮"""
        button_width = 120
        button_height = 30
        start_x = 650

        buttons = {
            # 游戏选择
            "gomoku_game": {
                "rect": pygame.Rect(start_x, 50, button_width, button_height),
                "text": "Gomoku",
                "color": COLORS["YELLOW"],
            },
            "snake_game": {
                "rect": pygame.Rect(start_x, 90, button_width, button_height),
                "text": "Snake",
                "color": COLORS["LIGHT_GRAY"],
            },
            # AI选择
            "random_ai": {
                "rect": pygame.Rect(start_x, 150, button_width, button_height),
                "text": "Random AI",
                "color": COLORS["YELLOW"],
            },
            "minimax_ai": {
                "rect": pygame.Rect(start_x, 190, button_width, button_height),
                "text": "Minimax AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            "mcts_ai": {
                "rect": pygame.Rect(start_x, 230, button_width, button_height),
                "text": "MCTS AI",
                "color": COLORS["LIGHT_GRAY"],
            },
            # 控制按钮
            "new_game": {
                "rect": pygame.Rect(start_x, 290, button_width, button_height),
                "text": "New Game",
                "color": COLORS["GREEN"],
            },
            "pause": {
                "rect": pygame.Rect(start_x, 330, button_width, button_height),
                "text": "Pause",
                "color": COLORS["ORANGE"],
            },
            "quit": {
                "rect": pygame.Rect(start_x, 370, button_width, button_height),
                "text": "Quit",
                "color": COLORS["RED"],
            },
        }

        return buttons

    def _switch_game(self, game_type):
        """切换游戏类型"""
        self.current_game = game_type

        # 更新游戏选择按钮颜色
        for btn_name in ["gomoku_game", "snake_game"]:
            self.buttons[btn_name]["color"] = COLORS["LIGHT_GRAY"]
        self.buttons[f"{game_type}_game"]["color"] = COLORS["YELLOW"]

        # 创建对应的环境和智能体
        if game_type == "gomoku":
            self.env = GomokuEnv(board_size=15, win_length=5)
            self.cell_size = 30
            self.update_interval = 1.0  # 五子棋不需要频繁更新
        elif game_type == "snake":
            self.env = SnakeEnv(board_size=20)
            self.cell_size = 25
            self.update_interval = 0.3  # 贪吃蛇需要频繁更新

        if self.env is None:
            print(f"Error: Failed to create environment for {game_type}")
            return

        self.human_agent = HumanAgent(name="Human Player", player_id=1)
        self._create_ai_agent()
        self.reset_game()

    def _create_ai_agent(self):
        """创建AI智能体"""
        try:
            if self.selected_ai == "RandomBot":
                self.ai_agent = RandomBot(name="Random AI", player_id=2)
            elif self.selected_ai == "MinimaxBot":
                if self.current_game == "gomoku":
                    self.ai_agent = MinimaxBot(name="Minimax AI", player_id=2, max_depth=3)
                else:
                    # 贪吃蛇游戏使用SnakeAI而不是MinimaxBot
                    self.ai_agent = SnakeAI(name="Snake AI", player_id=2)
            elif self.selected_ai == "MCTSBot":
                if self.current_game == "gomoku":
                    self.ai_agent = MCTSBot(
                        name="MCTS AI", player_id=2, simulation_count=300
                    )
                else:
                    # 贪吃蛇游戏使用SmartSnakeAI
                    self.ai_agent = SmartSnakeAI(name="Smart Snake AI", player_id=2)
            else:
                # 默认使用RandomBot
                self.ai_agent = RandomBot(name="Random AI", player_id=2)
                self.selected_ai = "RandomBot"
            
            print(f"Created AI agent: {self.ai_agent.name} for {self.current_game}")
        except Exception as e:
            print(f"Error creating AI agent: {e}")
            # 创建默认AI
            self.ai_agent = RandomBot(name="Random AI", player_id=2)
            self.selected_ai = "RandomBot"

    def reset_game(self):
        """重置游戏"""
        if self.env is None:
            print("Error: Environment is None, cannot reset game")
            return
            
        try:
            self.env.reset()
            self.game_over = False
            self.winner = None
            self.last_move = None
            self.thinking = False
            self.current_agent = self.human_agent
            self.last_update = time.time()
            self.paused = False
            self.buttons["pause"]["text"] = "Pause"
            
            # 贪吃蛇专用重置
            if self.current_game == "snake":
                self.last_auto_move_time = time.time()
                self.human_direction = (0, 1)  # 人类蛇默认向右
                self.ai_direction = (0, -1)    # AI蛇默认向左
            
            print(f"Game reset successfully for {self.current_game}")
        except Exception as e:
            print(f"Error resetting game: {e}")

    def handle_events(self) -> bool:
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                # 处理贪吃蛇的键盘输入
                if (
                    self.current_game == "snake"
                    and isinstance(self.current_agent, HumanAgent)
                    and not self.game_over
                    and not self.thinking
                    and not self.paused
                ):
                    self._handle_snake_input(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    mouse_pos = pygame.mouse.get_pos()

                    # 检查按钮点击
                    click_result = self._handle_button_click(mouse_pos)
                    if click_result is None:
                        return False
                    elif click_result is True:
                        # 如果点击了按钮，重置游戏状态,避免多余处理
                        self.reset_game()
                    # 检查五子棋棋盘点击
                    if (
                        self.current_game == "gomoku"
                        and not self.game_over
                        and isinstance(self.current_agent, HumanAgent)
                        and not self.thinking
                        and not self.paused
                    ):
                        self._handle_gomoku_click(mouse_pos)

        return True

    def _handle_button_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """处理按钮点击"""
        for button_name, button_info in self.buttons.items():
            if button_info["rect"].collidepoint(mouse_pos):
                if button_name == "new_game":
                    self.reset_game()
                elif button_name == "quit":
                    return False  # 改为False而不是None
                elif button_name == "pause":
                    self.paused = not self.paused
                    self.buttons["pause"]["text"] = "Resume" if self.paused else "Pause"
                elif button_name in ["gomoku_game", "snake_game"]:
                    game_type = button_name.split("_")[0]
                    self._switch_game(game_type)
                elif button_name.endswith("_ai"):
                    # 更新选中的AI
                    old_ai = f"{self.selected_ai.lower()}_ai"
                    if old_ai in self.buttons:
                        self.buttons[old_ai]["color"] = COLORS["LIGHT_GRAY"]

                    if button_name == "random_ai":
                        self.selected_ai = "RandomBot"
                    elif button_name == "minimax_ai":
                        self.selected_ai = "MinimaxBot"
                    elif button_name == "mcts_ai":
                        self.selected_ai = "MCTSBot"

                    self.buttons[button_name]["color"] = COLORS["YELLOW"]
                    self._create_ai_agent()
                    self.reset_game()

                return True
        return False

    def _handle_gomoku_click(self, mouse_pos: Tuple[int, int]):
        """处理五子棋棋盘点击"""
        x, y = mouse_pos
        board_x = x - self.margin
        board_y = y - self.margin

        if board_x < 0 or board_y < 0:
            return

        col = round(board_x / self.cell_size)
        row = round(board_y / self.cell_size)

        if 0 <= row < 15 and 0 <= col < 15:
            action = (row, col)
            if hasattr(self.env, 'get_valid_actions') and action in self.env.get_valid_actions():
                self._make_move(action)

    def _handle_snake_input(self, key):
        """处理贪吃蛇键盘输入"""
        current_time = time.time()
        
        # 更严格的防重复机制
        # 1. 检查时间冷却
        if current_time - self.last_key_time < self.key_cooldown:
            print(f"时间冷却中，忽略按键: {key} (间隔: {current_time - self.last_key_time:.3f}s)")
            return
        
        # 2. 检查是否是相同的按键
        if (self.last_processed_key == key and 
            current_time - self.last_processed_time < self.key_cooldown * 2):
            print(f"相同按键冷却中，忽略按键: {key}")
            return
        
        key_to_action = {
            pygame.K_UP: (-1, 0),    # 上
            pygame.K_w: (-1, 0),
            pygame.K_DOWN: (1, 0),   # 下
            pygame.K_s: (1, 0),
            pygame.K_LEFT: (0, -1),  # 左
            pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1),  # 右
            pygame.K_d: (0, 1)
        }
        
        if key in key_to_action:
            action = key_to_action[key]
            print(f"处理键盘输入: {key} -> 动作: {action} (时间: {current_time:.3f})")
            
            # 更新防重复状态
            self.last_key_time = current_time
            self.last_processed_key = key
            self.last_processed_time = current_time
            
            # 只更新人类蛇的方向，不立即移动
            # 检查是否是有效的方向（不能反向移动）
            if action != (-self.human_direction[0], -self.human_direction[1]):
                self.human_direction = action
                print(f"更新人类蛇方向: {action}")
            else:
                print(f"无效方向: {action} (不能反向移动)")

    def _make_move(self, human_action=None, ai_action=None):
        """执行移动 - 支持双蛇同时移动"""
        if self.game_over or self.paused:
            return

        if self.env is None:
            print("Error: Environment is None, cannot make move")
            return

        # 额外的防重复保护
        current_time = time.time()
        if hasattr(self, 'last_move_time') and current_time - self.last_move_time < 0.1:
            print(f"移动冷却中，忽略移动")
            return

        try:
            # 对于贪吃蛇游戏，我们需要特殊处理
            if (self.current_game == "snake" and 
                hasattr(self.env.game, 'snake1') and 
                hasattr(self.env.game, 'snake2')):
                
                # 记录移动前的状态
                print(f"\n=== 移动前状态 ===")
                print(f"人类蛇: alive={getattr(self.env.game, 'alive1', False)}, 位置={getattr(self.env.game, 'snake1', [])}, 长度={len(getattr(self.env.game, 'snake1', []))}")
                print(f"AI蛇: alive={getattr(self.env.game, 'alive2', False)}, 位置={getattr(self.env.game, 'snake2', [])}, 长度={len(getattr(self.env.game, 'snake2', []))}")
                print(f"食物位置: {getattr(self.env.game, 'foods', [])}")
                
                # 更新蛇的方向
                if human_action:
                    setattr(self.env.game, 'direction1', human_action)
                    print(f"人类蛇方向: {human_action}")
                
                if ai_action:
                    setattr(self.env.game, 'direction2', ai_action)
                    print(f"AI蛇方向: {ai_action}")
                
                # 移动人类蛇
                if getattr(self.env.game, 'alive1', False):
                    print(f"移动人类蛇: 从 {getattr(self.env.game, 'snake1', [[]])[0]} 到 ({getattr(self.env.game, 'snake1', [[]])[0][0] + getattr(self.env.game, 'direction1', (0,0))[0]}, {getattr(self.env.game, 'snake1', [[]])[0][1] + getattr(self.env.game, 'direction1', (0,0))[1]})")
                    if hasattr(self.env.game, '_move_snake'):
                        self.env.game._move_snake(1)
                    print(f"人类蛇移动完成: alive={getattr(self.env.game, 'alive1', False)}, 新位置={getattr(self.env.game, 'snake1', [])}")
                else:
                    print(f"人类蛇已死亡，跳过移动")
                
                # 移动AI蛇
                if getattr(self.env.game, 'alive2', False):
                    print(f"移动AI蛇: 从 {getattr(self.env.game, 'snake2', [[]])[0]} 到 ({getattr(self.env.game, 'snake2', [[]])[0][0] + getattr(self.env.game, 'direction2', (0,0))[0]}, {getattr(self.env.game, 'snake2', [[]])[0][1] + getattr(self.env.game, 'direction2', (0,0))[1]})")
                    if hasattr(self.env.game, '_move_snake'):
                        self.env.game._move_snake(2)
                    print(f"AI蛇移动完成: alive={getattr(self.env.game, 'alive2', False)}, 新位置={getattr(self.env.game, 'snake2', [])}")
                else:
                    print(f"AI蛇已死亡，跳过移动")
                
                # 记录移动时间
                self.last_move_time = current_time
                
                # 检查游戏是否结束
                print(f"\n=== 移动后状态 ===")
                print(f"人类蛇: alive={getattr(self.env.game, 'alive1', False)}, 位置={getattr(self.env.game, 'snake1', [])}")
                print(f"AI蛇: alive={getattr(self.env.game, 'alive2', False)}, 位置={getattr(self.env.game, 'snake2', [])}")
                
                if not (getattr(self.env.game, 'alive1', False) and getattr(self.env.game, 'alive2', False)):
                    self.game_over = True
                    self.winner = self.env.get_winner()
                    print(f"游戏结束，获胜者: {self.winner}")
                    print(f"游戏结束原因:")
                    if not getattr(self.env.game, 'alive1', False):
                        print(f"  - 人类蛇死亡")
                    if not getattr(self.env.game, 'alive2', False):
                        print(f"  - AI蛇死亡")
                else:
                    # 生成新的食物
                    if hasattr(self.env.game, '_generate_foods'):
                        self.env.game._generate_foods()
                    print(f"双蛇移动完成，游戏继续")
                
                self.last_move = human_action if human_action else ai_action
            else:
                # 其他游戏（如五子棋）的正常回合制逻辑
                action = human_action if human_action else ai_action
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.last_move = action
                print(f"Move executed: {action}, reward: {reward}")

                # 检查游戏是否结束
                if terminated or truncated:
                    self.game_over = True
                    self.winner = self.env.get_winner()
                    print(f"Game over! Winner: {self.winner}")
                else:
                    # 切换玩家
                    self._switch_player()
                    if self.current_agent:
                        print(f"Switched to player: {self.current_agent.name}")
                    else:
                        print("Switched to player: Unknown")

        except Exception as e:
            print(f"Move execution failed: {e}")
            import traceback
            traceback.print_exc()

    def _switch_player(self):
        """切换玩家"""
        if self.current_agent is None:
            print("Error: current_agent is None")
            return
            
        if isinstance(self.current_agent, HumanAgent):
            if self.ai_agent is None:
                print("Error: ai_agent is None")
                return
            self.current_agent = self.ai_agent
            self.thinking = True
            print(f"Switched to AI: {self.ai_agent.name}")
        else:
            if self.human_agent is None:
                print("Error: human_agent is None")
                return
            self.current_agent = self.human_agent
            print(f"Switched to Human: {self.human_agent.name}")

    def update_game(self):
        """更新游戏"""
        if self.game_over or self.paused:
            return

        current_time = time.time()

        # 贪吃蛇游戏的自动移动逻辑
        if self.current_game == "snake":
            # 检查是否需要自动移动
            if current_time - self.last_auto_move_time < self.auto_move_interval:
                return
            
            self.last_auto_move_time = current_time
            
            # 获取AI的决策
            ai_action = None
            if (self.env and hasattr(self.env, 'game') and 
                getattr(self.env.game, 'alive2', False)):
                try:
                    if hasattr(self.env, '_get_observation'):
                        observation = self.env._get_observation()
                        if self.ai_agent and hasattr(self.ai_agent, 'get_action'):
                            ai_action = self.ai_agent.get_action(observation, self.env)
                            if ai_action:
                                self.ai_direction = ai_action
                                print(f"AI决策: {ai_action}")
                            else:
                                # 如果AI没有返回动作，使用当前方向
                                ai_action = self.ai_direction
                                print(f"AI决策失败，使用当前方向: {ai_action}")
                        else:
                            ai_action = self.ai_direction
                            print(f"AI智能体不可用，使用当前方向: {ai_action}")
                    else:
                        ai_action = self.ai_direction
                        print(f"环境不可用，使用当前方向: {ai_action}")
                except Exception as e:
                    print(f"AI决策异常: {e}")
                    # 使用当前方向，避免AI蛇意外死亡
                    ai_action = self.ai_direction
                    print(f"AI使用当前方向: {ai_action}")
            else:
                print(f"AI蛇已死亡，跳过AI决策")
            
            # 执行双蛇同时移动
            self._make_move(human_action=self.human_direction, ai_action=ai_action)
        else:
            # 五子棋等回合制游戏的逻辑
            # 检查是否需要更新
            if current_time - self.last_update < self.update_interval:
                return

            self.last_update = current_time

            # AI回合 - 只对五子棋等回合制游戏
            if (not isinstance(self.current_agent, HumanAgent) and self.thinking):
                try:
                    # 获取AI动作
                    if self.env is None:
                        print("Error: Environment is None during AI turn")
                        self.thinking = False
                        return
                        
                    observation = self.env._get_observation()
                    if self.current_agent is None:
                        print("Error: current_agent is None during AI turn")
                        self.thinking = False
                        return
                        
                    action = self.current_agent.get_action(observation, self.env)

                    if action:
                        self._make_move(action)

                    self.thinking = False

                except Exception as e:
                    print(f"AI thinking failed: {e}")
                    self.thinking = False

    def draw(self):
        """绘制游戏界面"""
        # 清空屏幕
        self.screen.fill(COLORS["WHITE"])

        # 绘制游戏区域
        if self.current_game == "gomoku":
            self._draw_gomoku()
        elif self.current_game == "snake":
            self._draw_snake()

        # 绘制UI
        self._draw_ui()

        # 绘制游戏状态
        self._draw_game_status()

        # 更新显示
        pygame.display.flip()

    def _draw_gomoku(self):
        """绘制五子棋"""
        board_size = 15

        # 绘制棋盘背景
        board_rect = pygame.Rect(
            self.margin - 20,
            self.margin - 20,
            board_size * self.cell_size + 40,
            board_size * self.cell_size + 40,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_BROWN"], board_rect)

        # 绘制网格线
        for i in range(board_size):
            # 垂直线
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (
                self.margin + i * self.cell_size,
                self.margin + (board_size - 1) * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)

            # 水平线
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (
                self.margin + (board_size - 1) * self.cell_size,
                self.margin + i * self.cell_size,
            )
            pygame.draw.line(self.screen, COLORS["BLACK"], start_pos, end_pos, 2)

        # 绘制星位
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["BLACK"], center, 4)

        # 绘制棋子
        if self.env is not None and self.env.game is not None:
            # 使用get_state()方法获取棋盘状态
            state = self.env.game.get_state()
            board = state['board']
            for row in range(board_size):
                for col in range(board_size):
                    if board[row, col] != 0:
                        center = (
                            self.margin + col * self.cell_size,
                            self.margin + row * self.cell_size,
                        )

                        if board[row, col] == 1:  # 人类玩家
                            color = COLORS["BLACK"]
                            border_color = COLORS["WHITE"]
                        else:  # AI玩家
                            color = COLORS["WHITE"]
                            border_color = COLORS["BLACK"]

                        pygame.draw.circle(self.screen, color, center, 12)
                        pygame.draw.circle(self.screen, border_color, center, 12, 2)

        # 绘制最后一步标记
        if (
            self.last_move
            and isinstance(self.last_move, tuple)
            and len(self.last_move) == 2
        ):
            row, col = self.last_move
            center = (
                self.margin + col * self.cell_size,
                self.margin + row * self.cell_size,
            )
            pygame.draw.circle(self.screen, COLORS["RED"], center, 6, 3)

    def _draw_snake(self):
        """绘制贪吃蛇"""
        board_size = 20

        # 绘制游戏区域背景
        game_rect = pygame.Rect(
            self.margin,
            self.margin,
            board_size * self.cell_size,
            board_size * self.cell_size,
        )
        pygame.draw.rect(self.screen, COLORS["LIGHT_GRAY"], game_rect)
        pygame.draw.rect(self.screen, COLORS["BLACK"], game_rect, 2)

        # 绘制网格
        for i in range(board_size + 1):
            # 垂直线
            x = self.margin + i * self.cell_size
            pygame.draw.line(
                self.screen,
                COLORS["GRAY"],
                (x, self.margin),
                (x, self.margin + board_size * self.cell_size),
                1,
            )
            # 水平线
            y = self.margin + i * self.cell_size
            pygame.draw.line(
                self.screen,
                COLORS["GRAY"],
                (self.margin, y),
                (self.margin + board_size * self.cell_size, y),
                1,
            )

        # 绘制游戏元素
        if self.env is not None and self.env.game is not None:
            state = self.env.game.get_state()
            board = state['board']
            for row in range(board_size):
                for col in range(board_size):
                    if board[row, col] != 0:
                        x = self.margin + col * self.cell_size + 2
                        y = self.margin + row * self.cell_size + 2
                        rect = pygame.Rect(x, y, self.cell_size - 4, self.cell_size - 4)

                        if board[row, col] == 1:  # 蛇1头部
                            pygame.draw.rect(self.screen, COLORS["BLUE"], rect)
                        elif board[row, col] == 2:  # 蛇1身体
                            pygame.draw.rect(self.screen, COLORS["CYAN"], rect)
                        elif board[row, col] == 3:  # 蛇2头部
                            pygame.draw.rect(self.screen, COLORS["RED"], rect)
                        elif board[row, col] == 4:  # 蛇2身体
                            pygame.draw.rect(self.screen, COLORS["ORANGE"], rect)
                        elif board[row, col] == 5:  # 食物
                            pygame.draw.rect(self.screen, COLORS["GREEN"], rect)

    def _draw_ui(self):
        """绘制UI界面"""
        # 绘制按钮
        for button_name, button_info in self.buttons.items():
            pygame.draw.rect(self.screen, button_info["color"], button_info["rect"])
            pygame.draw.rect(self.screen, COLORS["BLACK"], button_info["rect"], 2)

            text_surface = self.font_medium.render(
                button_info["text"], True, COLORS["BLACK"]
            )
            text_rect = text_surface.get_rect(center=button_info["rect"].center)
            self.screen.blit(text_surface, text_rect)

        # 绘制标题
        title_text = self.font_medium.render("Game Selection:", True, COLORS["BLACK"])
        self.screen.blit(title_text, (self.buttons["gomoku_game"]["rect"].x, 25))

        ai_title_text = self.font_medium.render("AI Selection:", True, COLORS["BLACK"])
        self.screen.blit(ai_title_text, (self.buttons["random_ai"]["rect"].x, 125))

        # 绘制操作说明
        if self.current_game == "gomoku":
            instructions = [
                "Gomoku Controls:",
                "• Click to place stone",
                "• Connect 5 to win",
            ]
        else:
            instructions = [
                "Snake Controls:",
                "• Arrow keys/WASD to change direction",
                "• Snakes move automatically every 0.3s",
                "• Eat food to grow",
                "• Avoid collision",
            ]

        start_y = 420
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, COLORS["DARK_GRAY"])
            self.screen.blit(
                text, (self.buttons["new_game"]["rect"].x, start_y + i * 20)
            )

    def _draw_game_status(self):
        """绘制游戏状态"""
        status_x = 20
        status_y = self.window_height - 100

        if self.paused:
            status_text = "Game Paused..."
            color = COLORS["ORANGE"]
        elif self.game_over:
            if self.winner == 1:
                status_text = "Congratulations! You Win!"
                color = COLORS["GREEN"]
            elif self.winner == 2:
                status_text = "AI Wins! Try Again!"
                color = COLORS["RED"]
            else:
                status_text = "Draw!"
                color = COLORS["ORANGE"]
        else:
            if isinstance(self.current_agent, HumanAgent):
                if self.current_game == "gomoku":
                    status_text = "Your Turn - Click to Place Stone"
                else:
                    status_text = "Auto Moving - Use Arrow Keys"
                color = COLORS["BLUE"]
            else:
                if self.thinking:
                    ai_name = self.ai_agent.name if self.ai_agent else "AI"
                    status_text = f"{ai_name} is Thinking..."
                    color = COLORS["ORANGE"]
                else:
                    ai_name = self.ai_agent.name if self.ai_agent else "AI"
                    status_text = f"{ai_name}'s Turn"
                    color = COLORS["RED"]

        text_surface = self.font_large.render(status_text, True, color)
        self.screen.blit(text_surface, (status_x, status_y))

        # 游戏信息
        info_y = status_y + 40
        if self.current_game == "gomoku":
            ai_name = self.ai_agent.name if self.ai_agent else 'AI'
            player_info = f"Black: Human Player  White: {ai_name}"
        else:
            if self.env and hasattr(self.env, 'game') and self.env.game:
                # 使用get_state()方法获取游戏状态
                state = self.env.game.get_state()
                if 'snake1' in state and 'snake2' in state:
                    len1 = len(state['snake1']) if state.get('alive1', False) else 0
                    len2 = len(state['snake2']) if state.get('alive2', False) else 0
                    alive1 = "Alive" if state.get('alive1', False) else "Dead"
                    alive2 = "Alive" if state.get('alive2', False) else "Dead"
                    player_info = f"Blue Snake(You): {len1} segments({alive1})  Red Snake(AI): {len2} segments({alive2})"
                else:
                    player_info = "Snake Battle in Progress..."
            else:
                player_info = "Snake Battle in Progress..."

        info_surface = self.font_small.render(player_info, True, COLORS["DARK_GRAY"])
        self.screen.blit(info_surface, (status_x, info_y))
        
        # 贪吃蛇游戏显示移动间隔
        if self.current_game == "snake":
            move_info = f"Move every {self.auto_move_interval}s"
            move_surface = self.font_small.render(move_info, True, COLORS["DARK_GRAY"])
            self.screen.blit(move_surface, (status_x, info_y + 20))

    def run(self):
        """运行游戏主循环"""
        running = True

        while running:
            # 处理事件
            running = self.handle_events()

            # 更新游戏
            self.update_game()

            # 绘制界面
            self.draw()

            # 控制帧率
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    """主函数"""
    print("Starting Multi-Game AI Battle Platform...")
    print("Supported Games:")
    print("- Gomoku: Click to place stones")
    print("- Snake: Arrow keys/WASD to control")
    print("- Multiple AI difficulty levels")
    print("- Real-time human vs AI battles")

    try:
        game = MultiGameGUI()
        game.run()
    except Exception as e:
        print(f"Game error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
