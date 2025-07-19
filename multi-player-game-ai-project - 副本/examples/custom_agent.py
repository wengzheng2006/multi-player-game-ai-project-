"""
自定义智能体示例
演示如何创建自己的AI智能体
"""

import random
import time
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent


class GreedyBot(BaseAgent):
    """贪心Bot - 总是选择看起来最好的动作"""
    
    def __init__(self, name: str = "GreedyBot", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """贪心选择动作"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("没有可用的动作")
        
        # 评估每个动作
        best_action = None
        best_score = float('-inf')
        
        for action in valid_actions:
            score = self._evaluate_action(action, env)
            if score > best_score:
                best_score = score
                best_action = action
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return best_action
    
    def _evaluate_action(self, action: Any, env: Any) -> float:
        """评估动作的分数"""
        # 克隆环境
        cloned_env = env.clone()
        
        # 执行动作
        cloned_env.step(action)
        
        # 检查是否获胜
        if cloned_env.is_terminal():
            winner = cloned_env.get_winner()
            if winner == self.player_id:
                return 1000.0  # 获胜
            elif winner == 3 - self.player_id:
                return -1000.0  # 失败
            else:
                return 0.0  # 平局
        
        # 对于五子棋，计算连子数量
        if hasattr(cloned_env, 'board_size'):
            return self._evaluate_gomoku_position(cloned_env)
        
        return 0.0
    
    def _evaluate_gomoku_position(self, env: Any) -> float:
        """评估五子棋位置"""
        board = env.get_board_state()
        board_size = env.board_size
        win_length = env.win_length
        
        score = 0.0
        
        # 评估每个方向的连子
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] != 0:
                    player = board[i, j]
                    multiplier = 1 if player == self.player_id else -1
                    
                    for dr, dc in directions:
                        count = self._count_consecutive(board, i, j, dr, dc, player)
                        if count >= win_length:
                            score += 1000 * multiplier
                        elif count == win_length - 1:
                            score += 100 * multiplier
                        elif count == win_length - 2:
                            score += 10 * multiplier
                        elif count == win_length - 3:
                            score += 1 * multiplier
        
        return score
    
    def _count_consecutive(self, board: Any, row: int, col: int, dr: int, dc: int, player: int) -> int:
        """计算连续棋子数量"""
        count = 0
        r, c = row, col
        
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r += dr
            c += dc
        
        return count


class DefensiveBot(BaseAgent):
    """防守Bot - 优先防守，然后进攻"""
    
    def __init__(self, name: str = "DefensiveBot", player_id: int = 1):
        super().__init__(name, player_id)
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """防守优先的策略"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("没有可用的动作")
        
        # 策略1: 如果有获胜机会，立即行动
        winning_move = self._find_winning_move(env)
        if winning_move:
            action = winning_move
        # 策略2: 如果对手有获胜机会，阻止
        elif self._need_block(env):
            action = self._find_blocking_move(env)
        # 策略3: 寻找最佳防守位置
        else:
            action = self._find_defensive_move(env)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return action
    
    def _find_winning_move(self, env: Any) -> Optional[Any]:
        """寻找获胜动作"""
        valid_actions = env.get_valid_actions()
        for action in valid_actions:
            cloned_env = env.clone()
            cloned_env.step(action)
            if cloned_env.is_terminal() and cloned_env.get_winner() == self.player_id:
                return action
        return None
    
    def _need_block(self, env: Any) -> bool:
        """检查是否需要阻止对手"""
        valid_actions = env.get_valid_actions()
        opponent = 3 - self.player_id
        
        for action in valid_actions:
            cloned_env = env.clone()
            cloned_env.step(action)
            if cloned_env.is_terminal() and cloned_env.get_winner() == opponent:
                return True
        return False
    
    def _find_blocking_move(self, env: Any) -> Any:
        """寻找阻止动作"""
        valid_actions = env.get_valid_actions()
        opponent = 3 - self.player_id
        
        for action in valid_actions:
            cloned_env = env.clone()
            cloned_env.step(action)
            if cloned_env.is_terminal() and cloned_env.get_winner() == opponent:
                return action
        
        # 如果没有找到阻止动作，随机选择
        return random.choice(valid_actions)
    
    def _find_defensive_move(self, env: Any) -> Any:
        """寻找防守动作"""
        valid_actions = env.get_valid_actions()
        
        # 优先选择边缘位置
        edge_actions = []
        center_actions = []
        
        for action in valid_actions:
            if hasattr(env, 'board_size'):
                row, col = action
                board_size = env.board_size
                
                # 检查是否为边缘位置
                if (row == 0 or row == board_size - 1 or 
                    col == 0 or col == board_size - 1):
                    edge_actions.append(action)
                else:
                    center_actions.append(action)
        
        # 优先选择边缘位置
        if edge_actions:
            return random.choice(edge_actions)
        elif center_actions:
            return random.choice(center_actions)
        else:
            return random.choice(valid_actions)


class PatternBot(BaseAgent):
    """模式Bot - 使用预定义的模式"""
    
    def __init__(self, name: str = "PatternBot", player_id: int = 1):
        super().__init__(name, player_id)
        self.patterns = self._create_patterns()
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """使用模式选择动作"""
        start_time = time.time()
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            raise ValueError("没有可用的动作")
        
        # 尝试应用模式
        for pattern in self.patterns:
            action = pattern(env, valid_actions)
            if action and action in valid_actions:
                break
        else:
            # 如果没有模式匹配，随机选择
            action = random.choice(valid_actions)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return action
    
    def _create_patterns(self) -> List:
        """创建模式列表"""
        patterns = [
            self._pattern_center_first,    # 优先选择中心
            self._pattern_corner_first,    # 优先选择角落
            self._pattern_edge_first,      # 优先选择边缘
            self._pattern_random           # 随机选择
        ]
        return patterns
    
    def _pattern_center_first(self, env: Any, valid_actions: List[Any]) -> Optional[Any]:
        """中心优先模式"""
        if hasattr(env, 'board_size'):
            center = env.board_size // 2
            center_action = (center, center)
            if center_action in valid_actions:
                return center_action
        return None
    
    def _pattern_corner_first(self, env: Any, valid_actions: List[Any]) -> Optional[Any]:
        """角落优先模式"""
        if hasattr(env, 'board_size'):
            board_size = env.board_size
            corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
            
            for corner in corners:
                if corner in valid_actions:
                    return corner
        return None
    
    def _pattern_edge_first(self, env: Any, valid_actions: List[Any]) -> Optional[Any]:
        """边缘优先模式"""
        if hasattr(env, 'board_size'):
            board_size = env.board_size
            edge_actions = []
            
            for action in valid_actions:
                row, col = action
                if (row == 0 or row == board_size - 1 or 
                    col == 0 or col == board_size - 1):
                    edge_actions.append(action)
            
            if edge_actions:
                return random.choice(edge_actions)
        return None
    
    def _pattern_random(self, env: Any, valid_actions: List[Any]) -> Optional[Any]:
        """随机模式"""
        return random.choice(valid_actions)


def test_custom_agents():
    """测试自定义智能体"""
    print("=== 测试自定义智能体 ===")
    
    from games.gomoku import GomokuEnv
    from agents import RandomBot
    
    # 创建环境
    env = GomokuEnv(board_size=9, win_length=5)
    
    # 创建自定义智能体
    greedy_bot = GreedyBot(name="贪心Bot", player_id=1)
    defensive_bot = DefensiveBot(name="防守Bot", player_id=2)
    pattern_bot = PatternBot(name="模式Bot", player_id=3)
    
    # 测试智能体
    agents = [greedy_bot, defensive_bot, pattern_bot, RandomBot(name="随机Bot", player_id=4)]
    
    print("智能体信息:")
    for agent in agents:
        info = agent.get_info()
        print(f"- {info['name']}: {info['description']}")
    
    # 进行简单测试
    print("\n进行简单测试...")
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i >= j:
                continue
            
            print(f"\n{agent1.name} vs {agent2.name}")
            
            # 重置环境
            observation, info = env.reset()
            
            # 进行游戏
            step_count = 0
            while not env.is_terminal() and step_count < 50:
                current_agent = agent1 if env.game.current_player == 1 else agent2
                action = current_agent.get_action(observation, env)
                observation, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                
                if terminated or truncated:
                    break
            
            winner = env.get_winner()
            print(f"  结果: {winner} (步数: {step_count})")


if __name__ == "__main__":
    test_custom_agents() 