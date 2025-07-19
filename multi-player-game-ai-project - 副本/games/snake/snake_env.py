"""
贪吃蛇环境
实现gym风格接口
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from games.base_env import BaseEnv
from games.snake.snake_game import SnakeGame


class SnakeEnv(BaseEnv):
    """贪吃蛇环境"""
    
    def __init__(self, board_size=20, **kwargs):
        self.board_size = board_size
        self.game = SnakeGame(board_size)
        super().__init__(self.game)

    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        self.observation_space = None
        self.action_space = None

    def _get_observation(self):
        """获取观察"""
        state = self.game.get_state()
        return state['board']

    def _get_action_mask(self):
        """获取动作掩码"""
        # 贪吃蛇所有方向都可能有效，但要避免直接掉头
        return np.ones(4, dtype=bool)  # [up, down, left, right]

    def get_valid_actions(self):
        """获取有效动作"""
        return self.game.get_valid_actions()

    def is_terminal(self):
        """检查游戏是否结束"""
        return self.game.is_terminal()

    def get_winner(self):
        """获取获胜者"""
        return self.game.get_winner()

    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            # 打印游戏状态到控制台
            self._print_game_state()
        state = self.game.get_state()
        return state['board']
    
    def _print_game_state(self):
        """打印游戏状态到控制台"""
        state = self.game.get_state()
        board = state['board']
        board_size = self.board_size
        
        print(f"\n=== 贪吃蛇游戏状态 ===")
        print(f"棋盘大小: {board_size}x{board_size}")
        print(f"蛇1位置: {state['snake1']}")
        print(f"蛇2位置: {state['snake2']}")
        print(f"食物位置: {state['foods']}")
        print(f"当前玩家: {state['current_player']}")
        print(f"蛇1存活: {state['alive1']}")
        print(f"蛇2存活: {state['alive2']}")
        
        # 打印简化的棋盘
        print("\n棋盘 (1=蛇1头, 2=蛇1身, 3=蛇2头, 4=蛇2身, 5=食物, 0=空):")
        for i in range(board_size):
            row = []
            for j in range(board_size):
                cell = board[i, j]
                if cell == 0:
                    row.append(".")
                elif cell == 1:
                    row.append("H1")  # 蛇1头
                elif cell == 2:
                    row.append("B1")  # 蛇1身
                elif cell == 3:
                    row.append("H2")  # 蛇2头
                elif cell == 4:
                    row.append("B2")  # 蛇2身
                elif cell == 5:
                    row.append("F")   # 食物
                else:
                    row.append("?")
            print(" ".join(row))
        print()

    def get_board_state(self):
        """获取棋盘状态"""
        state = self.game.get_state()
        return state['board']

    def get_snake_positions(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """获取蛇的位置"""
        state = self.game.get_state()
        return state['snake1'], state['snake2']
    
    def get_food_positions(self) -> List[Tuple[int, int]]:
        """获取食物位置"""
        state = self.game.get_state()
        return state['foods']
    
    def is_valid_move(self, action: Tuple[int, int]) -> bool:
        """检查移动是否有效"""
        return action in self.game.get_valid_actions()
    
    def get_game_info(self) -> Dict[str, Any]:
        """获取游戏信息"""
        info = self.game.get_game_info()
        info.update({
            'board_size': self.board_size,
            'initial_length': self.game.initial_length,
            'food_count': self.game.food_count,
            'snake1_length': len(self.game.snake1),
            'snake2_length': len(self.game.snake2),
            'alive1': self.game.alive1,
            'alive2': self.game.alive2
        })
        return info
    
    def clone(self):
        """克隆环境"""
        cloned_game = self.game.clone()
        cloned_env = SnakeEnv(self.board_size)
        cloned_env.game = cloned_game
        return cloned_env 