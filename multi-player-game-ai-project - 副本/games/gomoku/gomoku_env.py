"""
五子棋环境
实现gym风格接口
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from games.base_env import BaseEnv
from games.gomoku.gomoku_game import GomokuGame


class GomokuEnv(BaseEnv):
    """五子棋环境"""
    
    def __init__(self, board_size: int = 15, win_length: int = 5):
        self.board_size = board_size
        self.win_length = win_length
        game = GomokuGame(board_size, win_length)
        super().__init__(game)
        self.game = game
    
    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        self.observation_space = None
        self.action_space = None
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        # 修复：直接访问棋盘属性，避免调用不存在的get_board方法
        return self.game.board.copy()
    
    def _get_action_mask(self) -> np.ndarray:
        """获取动作掩码"""
        mask = np.zeros((self.board_size, self.board_size), dtype=bool)
        for (i, j) in self.game.get_valid_actions():
            mask[i, j] = True
        return mask
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """获取有效动作"""
        return self.game.get_valid_actions()
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.game.is_terminal()
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        return self.game.get_winner()
    
    def render(self, mode: str = 'human'):
        """渲染环境"""
        if mode == 'human':
            print(self.game.get_board_string())
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            return self.game.render()
    
    def _render_rgb_array(self) -> np.ndarray:
        """渲染为RGB数组"""
        board = self.game.board
        height, width = board.shape
        
        # 创建RGB图像
        img = np.ones((height * 30, width * 30, 3), dtype=np.uint8) * 255
        
        for i in range(height):
            for j in range(width):
                x, y = j * 30, i * 30
                
                # 绘制网格
                img[y:y+30, x:x+30] = [240, 240, 240]
                
                # 绘制棋子
                if board[i, j] == 1:  # 玩家1 (红色)
                    img[y+5:y+25, x+5:x+25] = [255, 0, 0]
                elif board[i, j] == 2:  # 玩家2 (蓝色)
                    img[y+5:y+25, x+5:x+25] = [0, 0, 255]
        
        return img
    
    def get_board_state(self) -> np.ndarray:
        """获取棋盘状态"""
        return self.game.board.copy()
    
    def clone(self) -> 'GomokuEnv':
        """克隆环境"""
        cloned_game = self.game.clone()
        cloned_env = GomokuEnv(self.board_size, self.win_length)
        cloned_env.game = cloned_game
        return cloned_env 