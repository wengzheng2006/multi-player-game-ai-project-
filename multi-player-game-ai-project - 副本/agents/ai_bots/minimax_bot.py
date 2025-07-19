from agents.base_agent import BaseAgent
import copy
import time
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional

class MinimaxBot(BaseAgent):
    """
    改进的Minimax Bot
    
    特性：
    - 防守优先策略：优先防守对手威胁
    - 智能进攻：无威胁时积极寻找进攻机会
    - 战略布局：围绕现有棋局思考，避免边角位置
    """
    
    def __init__(self, name="MinimaxBot", player_id=1, max_depth=4, use_alpha_beta=True, timeout=3):
        """
        初始化Minimax Bot
        """
        super().__init__(name, player_id)
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.timeout = timeout
        self.start_time = None
        self.transposition_table = {} 
        self.nodes_evaluated = 0  
        self.pruning_count = 0 
    
    def get_action(self, observation, env):
        """
        获取最佳动作（防守优先，无威胁时积极进攻）
        
        决策优先级：
        1. 直接获胜机会（最高优先级）
        2. 防守威胁（成五、活四、冲四）
        3. 进攻机会（成五、活四、冲四、活三）
        4. 战略布局（围绕现有棋局）
        5. Minimax搜索（深度搜索最佳动作）
        """
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
            
        # 初始化搜索状态
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.pruning_count = 0

        # 第一步：处理AI第一步的特殊情况
        board = env.game.board
        board_size = env.game.board_size
        my_count = np.sum(board == self.player_id)
        opp_count = np.sum(board == (3 - self.player_id))
        
        if my_count == 0:
            return self._handle_first_move(board, board_size, valid_actions)

        # 第二步：检查直接获胜机会（最高优先级）
        winning_moves = self._find_winning_moves(env)
        if winning_moves:
            return winning_moves[0]  # 直接获胜

        # 第三步：检查防守需求（成五、活四、冲四威胁）
        defensive_moves = self._find_defensive_moves(env)
        if defensive_moves:
            return self._select_best_defense(defensive_moves, env)

        # 第四步：无威胁时积极进攻
        offensive_moves = self._find_offensive_moves(env)
        if offensive_moves:
            return self._select_best_offense(offensive_moves, env)
        
        # 第五步：无进攻机会时，围绕现有棋局思考
        strategic_moves = self._find_strategic_moves(env)
        if strategic_moves:
            return strategic_moves[0]
        
        # 第六步：使用minimax搜索最佳动作
        return self._minimax_search(valid_actions, env, defensive_moves)
    
    def _handle_first_move(self, board, board_size, valid_actions):
        # 找到玩家首子
        opp_pos = np.argwhere(board == (3 - self.player_id))
        if len(opp_pos) > 0:
            opp_first = opp_pos[0]
            # 以玩家首子为中心，优先在其周围落子
            candidates = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = opp_first[0] + dx, opp_first[1] + dy
                    if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == 0:
                        candidates.append((x, y))
            if candidates:
                # 优先选择靠近中心的位置
                center = board_size // 2
                candidates.sort(key=lambda pos: abs(pos[0]-center)+abs(pos[1]-center))
                return candidates[0]
        
        # 否则优先选择中心位置
        center = board_size // 2
        if board[center, center] == 0:
            return (center, center)
        
        # 最后选择第一个有效动作
        return valid_actions[0]
    
    def _select_best_defense(self, defensive_moves, env):
        """
        选择最佳防守点
        
        Args:
            defensive_moves: 防守点列表
            env: 游戏环境
            
        Returns:
            最佳防守动作
        """
        # 按威胁等级排序：成五 > 活四 > 冲四
        for move in defensive_moves:
            if self._can_win_at_position(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size):
                return move  # 成五威胁
        for move in defensive_moves:
            if self._can_form_live_four(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size):
                return move  # 活四威胁
        for move in defensive_moves:
            if self._can_form_blocked_four(env.game.board, move[0], move[1], 3 - self.player_id, env.game.board_size):
                return move  # 冲四威胁
        return defensive_moves[0]  # 默认选择第一个防守点
    
    def _select_best_offense(self, offensive_moves, env):
        """
        选择最佳进攻点
        
        Args:
            offensive_moves: 进攻点列表
            env: 游戏环境
            
        Returns:
            最佳进攻动作
        """
        # 按威胁等级排序：成五 > 活四 > 冲四 > 活三
        for move in offensive_moves:
            if self._can_win_at_position(env.game.board, move[0], move[1], self.player_id, env.game.board_size):
                return move  # 成五机会
        for move in offensive_moves:
            if self._can_form_live_four(env.game.board, move[0], move[1], self.player_id, env.game.board_size):
                return move  # 活四机会
        for move in offensive_moves:
            if self._can_form_blocked_four(env.game.board, move[0], move[1], self.player_id, env.game.board_size):
                return move  # 冲四机会
        return offensive_moves[0]  # 默认选择第一个进攻点
    
    def _minimax_search(self, valid_actions, env, defensive_moves):
        """
        使用Minimax算法搜索最佳动作
        
        Args:
            valid_actions: 有效动作列表
            env: 游戏环境
            defensive_moves: 防守点列表（用于调整搜索深度）
            
        Returns:
            最佳动作
        """
        # 计算最优搜索深度
        current_depth = self._calculate_optimal_depth(valid_actions)
        
        # 无威胁时增加搜索深度
        if not defensive_moves:
            current_depth = min(current_depth + 1, 6)
        
        # 按启发式价值排序动作
        sorted_actions = self._sort_actions_by_value(valid_actions, env)
        best_score = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        
        # 遍历所有动作
        for action in sorted_actions:
            # 检查时间限制
            if time.time() - self.start_time > self.timeout:
                break
                
            # 评估动作
            game_copy = env.clone()
            game_copy.step(action)
            score = self.minimax(game_copy, current_depth - 1, False, 3 - self.player_id)
            
            # 更新最佳动作
            if score > best_score or best_action is None:
                best_score = score
                best_action = action
                
            # Alpha-Beta剪枝
            if self.use_alpha_beta:
                alpha = max(alpha, score)
        
        # 更新统计信息
        move_time = time.time() - self.start_time
        self.total_moves += 1
        self.total_time += move_time
        
        # 保证返回一个动作
        if best_action is None:
            best_action = sorted_actions[0] if sorted_actions else valid_actions[0]
            
        return best_action
    
    def _select_best_defense_old(self, defensive_moves, env):
        """选择最佳防守点（旧版本，保留兼容性）"""
        board = env.game.board
        board_size = env.game.board_size
        best_defense = None
        best_threat_level = -1
        
        for move in defensive_moves:
            threat_level = 0
            # 检查这个防守点能阻止的威胁等级
            if self._can_win_at_position(board, move[0], move[1], 3 - self.player_id, board_size):
                threat_level = 4  # 直接成五威胁
            elif self._can_form_live_four(board, move[0], move[1], 3 - self.player_id, board_size):
                threat_level = 3  # 活四威胁
            elif self._can_form_blocked_four(board, move[0], move[1], 3 - self.player_id, board_size):
                threat_level = 2  # 冲四威胁
            elif self._can_form_live_three(board, move[0], move[1], 3 - self.player_id, board_size):
                threat_level = 1  # 活三威胁
            
            if threat_level > best_threat_level:
                best_threat_level = threat_level
                best_defense = move
        
        return best_defense if best_defense else defensive_moves[0]
    
    def minimax(self, game, depth, maximizing, player_id):
        """
        标准Minimax算法实现
        
        Args:
            game: 游戏状态
            depth: 当前搜索深度
            maximizing: 是否为最大化玩家
            player_id: 当前玩家ID
            
        Returns:
            评估分数
        """
        # 终止条件：达到最大深度或游戏结束
        if depth == 0 or game.is_terminal():
            return self.evaluate_position(game, self.player_id)  # 始终以AI为视角
            
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return self.evaluate_position(game, self.player_id)
        
        if maximizing:
            # 最大化玩家：寻找最高分数
            max_score = float('-inf')
            for action in valid_actions:
                game_copy = game.clone()
                game_copy.step(action)
                score = self.minimax(game_copy, depth - 1, False, 3 - player_id)
                max_score = max(max_score, score)
            return max_score
        else:
            # 最小化玩家：寻找最低分数
            min_score = float('inf')
            for action in valid_actions:
                game_copy = game.clone()
                game_copy.step(action)
                score = self.minimax(game_copy, depth - 1, True, 3 - player_id)
                min_score = min(min_score, score)
            return min_score

    def _calculate_optimal_depth(self, valid_actions):
        """
        动态计算最优搜索深度
        
        根据以下因素调整搜索深度：
        - 有效动作数量：动作少时可以搜索更深
        - 剩余时间：时间充足时增加深度
        - 基础深度：确保最小搜索深度
        
        Args:
            valid_actions: 有效动作列表
            
        Returns:
            调整后的搜索深度
        """
        action_count = len(valid_actions)
        if self.start_time is None:
            return self.max_depth
        
        # 计算剩余时间
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.timeout - elapsed_time)
        
        # 根据动作数量和剩余时间调整深度
        if action_count <= 5 and remaining_time > 3:
            # 动作少且时间充足：增加深度
            return min(self.max_depth + 1, 6)
        elif action_count <= 10 and remaining_time > 2:
            # 动作中等且时间充足：使用标准深度
            return self.max_depth
        else:
            # 动作多或时间紧张：减少深度
            return max(2, self.max_depth - 1)
    
    def _find_winning_moves(self, env):
        """查找直接获胜的动作（加强版）"""
        winning_moves = []
        board = env.game.board
        board_size = env.game.board_size
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:  # 空位置
                    # 检查直接获胜（成五）
                    if self._can_win_at_position(board, i, j, self.player_id, board_size):
                        winning_moves.append((i, j))
        
        return winning_moves
    
    def _find_defensive_moves(self, env):
        """只检测成五、活四、冲四防守点"""
        defensive_moves = []
        board = env.game.board
        board_size = env.game.board_size
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    if self._can_win_at_position(board, i, j, 3 - self.player_id, board_size):
                        defensive_moves.append((i, j))
                    elif self._can_form_live_four(board, i, j, 3 - self.player_id, board_size):
                        defensive_moves.append((i, j))
                    elif self._can_form_blocked_four(board, i, j, 3 - self.player_id, board_size):
                        defensive_moves.append((i, j))
        return defensive_moves
    
    def _find_offensive_moves(self, env):
        """查找进攻机会（成五、活四、冲四、活三）"""
        offensive_moves = []
        board = env.game.board
        board_size = env.game.board_size
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    if self._can_win_at_position(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
                    elif self._can_form_live_four(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
                    elif self._can_form_blocked_four(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
                    elif self._can_form_live_three(board, i, j, self.player_id, board_size):
                        offensive_moves.append((i, j))
        return offensive_moves
    
    def _find_strategic_moves(self, env):
        """查找战略位置（围绕现有棋局，避免边角）"""
        strategic_moves = []
        board = env.game.board
        board_size = env.game.board_size
        
        # 找到所有现有棋子的位置
        existing_pieces = []
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] != 0:
                    existing_pieces.append((i, j))
        
        if not existing_pieces:
            # 如果没有棋子，优先选择中心位置
            center = board_size // 2
            if board[center, center] == 0:
                return [(center, center)]
            else:
                # 选择中心附近的空位
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        x, y = center + dx, center + dy
                        if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == 0:
                            strategic_moves.append((x, y))
                return strategic_moves if strategic_moves else []
        
        # 计算现有棋子的中心
        avg_x = sum(pos[0] for pos in existing_pieces) / len(existing_pieces)
        avg_y = sum(pos[1] for pos in existing_pieces) / len(existing_pieces)
        
        # 在现有棋子周围寻找战略位置
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:  # 空位置
                    # 避免边角位置
                    if (i == 0 or i == board_size-1) and (j == 0 or j == board_size-1):
                        continue
                    
                    # 计算到现有棋子中心的距离
                    distance_to_center = abs(i - avg_x) + abs(j - avg_y)
                    
                    # 检查是否在现有棋子附近（距离不超过3）
                    near_existing = False
                    for piece in existing_pieces:
                        if abs(i - piece[0]) + abs(j - piece[1]) <= 3:
                            near_existing = True
                            break
                    
                    if near_existing and distance_to_center <= 5:
                        strategic_moves.append((i, j, distance_to_center))
        
        # 按距离排序，优先选择靠近现有棋子的位置
        strategic_moves.sort(key=lambda x: x[2])
        return [(x[0], x[1]) for x in strategic_moves]
    
    def _can_form_live_four(self, board, row, col, player, board_size):
        """检查是否能形成活四"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        has_live_four = False
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            blocked = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 检查是否形成活四
            if count >= 4 and blocked == 0:
                has_live_four = True
                break
        
        # 恢复原值
        board[row, col] = original_value
        return has_live_four
    
    def _can_form_blocked_four(self, board, row, col, player, board_size):
        """检查是否能形成冲四"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        has_blocked_four = False
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            blocked = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 检查是否形成冲四
            if count >= 4 and blocked == 1:
                has_blocked_four = True
                break
        
        # 恢复原值
        board[row, col] = original_value
        return has_blocked_four
    
    def _can_form_live_three(self, board, row, col, player, board_size):
        """检查是否能形成活三"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        has_live_three = False
        
        # 检查四个方向
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            blocked = 0
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x, y] == player:
                        count += 1
                    elif board[x, y] != 0:
                        blocked += 1
                        break
                    else:
                        break
                else:
                    blocked += 1
                    break
            
            # 检查是否形成活三
            if count >= 3 and blocked == 0:
                has_live_three = True
                break
        
        # 恢复原值
        board[row, col] = original_value
        return has_live_three
    
    def _can_win_at_position(self, board, row, col, player, board_size):
        """检查在指定位置落子是否能直接获胜"""
        # 临时在此位置放置棋子
        original_value = board[row, col]
        board[row, col] = player
        
        # 检查四个方向是否能成五
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            count = 1  # 当前位置
            
            # 向一个方向检查
            for k in range(1, 5):
                x, y = row + dx*k, col + dy*k
                if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == player:
                    count += 1
                else:
                    break
            
            # 向相反方向检查
            for k in range(1, 5):
                x, y = row - dx*k, col - dy*k
                if 0 <= x < board_size and 0 <= y < board_size and board[x, y] == player:
                    count += 1
                else:
                    break
            
            if count >= 5:
                board[row, col] = original_value
                return True
        
        # 恢复原值
        board[row, col] = original_value
        return False
    
    def _sort_actions_by_value(self, actions, game):
        """按价值排序动作"""
        action_values = []
        for action in actions:
            value = self._quick_evaluate_action(game, action)
            action_values.append((value, action))
        
        # 按价值降序排序
        action_values.sort(reverse=True)
        return [action for value, action in action_values]
    
    def _quick_evaluate_action(self, game, action):
        """快速评估动作价值"""
        game_copy = game.clone()
        game_copy.step(action)
        return self._quick_evaluate_position(game_copy)
    
    def _quick_evaluate_position(self, game):
        """快速评估位置价值"""
        if hasattr(game, 'board'):
            return self._quick_evaluate_gomoku(game)
        elif hasattr(game, 'snake1') and hasattr(game, 'snake2'):
            return self._quick_evaluate_snake(game)
        return 0
    
    def _quick_evaluate_gomoku(self, game):
        """快速评估五子棋位置"""
        board = game.board
        board_size = len(board)
        score = 0
        
        # 检查威胁
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:  # 空位置
                    # 我方威胁
                    if self._can_win_at_position(board, i, j, self.player_id, board_size):
                        score += 10000
                    # 对手威胁
                    elif self._can_win_at_position(board, i, j, 3 - self.player_id, board_size):
                        score -= 10000
        
        # 位置价值（中心优先）
        center = board_size // 2
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == self.player_id:
                    distance_to_center = abs(i - center) + abs(j - center)
                    score += max(0, 10 - distance_to_center)
                elif board[i, j] != 0:
                    distance_to_center = abs(i - center) + abs(j - center)
                    score -= max(0, 10 - distance_to_center)
        
        return score
    
    def _quick_evaluate_snake(self, game):
        """快速评估贪吃蛇位置"""
        if not hasattr(game, 'snake1') or not hasattr(game, 'snake2'):
            return 0
        
        len1 = len(game.snake1) if game.alive1 else 0
        len2 = len(game.snake2) if game.alive2 else 0
        
        if self.player_id == 1:
            return (len1 - len2) * 10
        else:
            return (len2 - len1) * 10

    def minimax_alpha_beta(self, game, depth, maximizing, alpha, beta):
        """带Alpha-Beta剪枝的Minimax算法"""
        # 检查转置表
        game_hash = self._get_game_hash(game)
        if game_hash in self.transposition_table:
            stored_depth, stored_score = self.transposition_table[game_hash]
            if stored_depth >= depth:
                return stored_score
        
        if depth == 0 or game.is_terminal():
            score = self.evaluate_position(game)
            self.nodes_evaluated += 1
            # 存储到转置表
            self.transposition_table[game_hash] = (depth, score)
            return score
        
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            score = self.evaluate_position(game)
            self.nodes_evaluated += 1
            return score
        
        # 动作排序
        sorted_actions = self._sort_actions_by_value(valid_actions, game)
        
        if maximizing:
            max_score = float('-inf')
            for action in sorted_actions:
                game_copy = game.clone()
                game_copy.step(action)
                score = self.minimax_alpha_beta(game_copy, depth - 1, False, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    self.pruning_count += 1
                    break  # Beta剪枝
            return max_score
        else:
            min_score = float('inf')
            for action in sorted_actions:
                game_copy = game.clone()
                game_copy.step(action)
                score = self.minimax_alpha_beta(game_copy, depth - 1, True, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    self.pruning_count += 1
                    break  # Alpha剪枝
            return min_score
    
    def _get_game_hash(self, game):
        """获取游戏状态的哈希值"""
        if hasattr(game, 'board'):
            return hash(game.board.tobytes())
        elif hasattr(game, 'snake1') and hasattr(game, 'snake2'):
            return hash(str(game.snake1) + str(game.snake2) + str(game.foods))
        else:
            return hash(str(game.get_state()))
    
    def evaluate_position(self, game, player_id=None):
        if player_id is None:
            player_id = self.player_id
            
        # 检查游戏结果
        winner = game.get_winner()
        if winner == player_id:
            return 100000  # 获胜极大奖励
        elif winner is not None:
            return -100000  # 失败极大惩罚
        elif game.is_terminal():
            return 0
        
        # 根据游戏类型选择评估方法
        if hasattr(game, 'board'):
            # 五子棋：评估连子情况和位置控制
            return self._evaluate_gomoku(game)
        elif hasattr(game, 'snake1') and hasattr(game, 'snake2'):
            # 贪吃蛇：评估长度差异和生存空间
            return self._evaluate_snake(game)
        else:
            return 0
    
    def _evaluate_gomoku(self, game):
        """
        评估五子棋位置（防守优先，无威胁时积极进攻）
        
        评估策略：
        1. 威胁检测：检查对手是否有紧急威胁
        2. 防守优先：有威胁时优先防守
        3. 积极进攻：无威胁时寻找进攻机会
        4. 位置控制：评估中心区域控制
        5. 连子评估：评估已有棋子的连子情况
        
        Args:
            game: 游戏状态
            
        Returns:
            五子棋位置评估分数
        """
        score = 0
        board = game.board
        board_size = len(board)
        
        # 第一步：检查是否有紧急威胁需要防守
        has_urgent_threat = False
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0:
                    # 检查对手在此位置的威胁等级
                    if (self._can_win_at_position(board, i, j, 3 - self.player_id, board_size) or
                        self._can_form_live_four(board, i, j, 3 - self.player_id, board_size) or
                        self._can_form_blocked_four(board, i, j, 3 - self.player_id, board_size)):
                        has_urgent_threat = True
                        break
            if has_urgent_threat:
                break
        
        # 第二步：根据威胁情况选择评估策略
        if has_urgent_threat:
            # 有紧急威胁时，优先防守
            opponent_threats = 0
            for i in range(board_size):
                for j in range(board_size):
                    if board[i, j] == 0:
                        # 评估对手威胁等级
                        if self._can_win_at_position(board, i, j, 3 - self.player_id, board_size):
                            opponent_threats += 1000000  # 成五威胁
                        elif self._can_form_live_four(board, i, j, 3 - self.player_id, board_size):
                            opponent_threats += 100000   # 活四威胁
                        elif self._can_form_blocked_four(board, i, j, 3 - self.player_id, board_size):
                            opponent_threats += 50000    # 冲四威胁
            score -= opponent_threats * 10  # 防守权重
        else:
            # 无威胁时，积极寻找进攻机会
            my_threats = 0
            for i in range(board_size):
                for j in range(board_size):
                    if board[i, j] == 0:
                        # 评估我方威胁等级
                        if self._can_win_at_position(board, i, j, self.player_id, board_size):
                            my_threats += 1000000  # 成五机会
                        elif self._can_form_live_four(board, i, j, self.player_id, board_size):
                            my_threats += 100000   # 活四机会
                        elif self._can_form_blocked_four(board, i, j, self.player_id, board_size):
                            my_threats += 50000    # 冲四机会
                        elif self._can_form_live_three(board, i, j, self.player_id, board_size):
                            my_threats += 10000    # 活三机会
            score += my_threats * 2  # 进攻权重（无威胁时加倍奖励）
        
        # 第三步：位置控制评估
        center_control = self._evaluate_center_control(game)
        score += center_control
        
        # 第四步：连子评估
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] != 0:
                    player = board[i, j]
                    multiplier = 1 if player == self.player_id else -1
                    connected_score = self._evaluate_connected_pieces(game, i, j)
                    score += multiplier * connected_score
        
        return score
    
    def _evaluate_connected_pieces(self, game, row, col):
        """评估已有棋子的连子情况"""
        board = game.board
        board_size = len(board)
        player = board[row, col]
        score = 0
        
        # 检查四个方向的连子情况
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
            line_score = self._evaluate_line_at_position(board, row, col, dx, dy, player, board_size)
            score += line_score
        
        return score
    
    def _evaluate_line_at_position(self, board, row, col, dx, dy, player, board_size):
        """评估指定位置在某个方向的连子情况"""
        count = 1  # 当前位置
        blocked = 0
        
        # 向一个方向检查
        for k in range(1, 5):
            x, y = row + dx*k, col + dy*k
            if 0 <= x < board_size and 0 <= y < board_size:
                if board[x, y] == player:
                    count += 1
                elif board[x, y] != 0:
                    blocked += 1
                    break
                else:
                    break
            else:
                blocked += 1
                break
        
        # 向相反方向检查
        for k in range(1, 5):
            x, y = row - dx*k, col - dy*k
            if 0 <= x < board_size and 0 <= y < board_size:
                if board[x, y] == player:
                    count += 1
                elif board[x, y] != 0:
                    blocked += 1
                    break
                else:
                    break
            else:
                blocked += 1
                break
        
        # 根据连子数和阻塞情况评分
        if count >= 5:
            return 10000  # 成五
        elif count == 4 and blocked == 0:
            return 1000   # 活四
        elif count == 4 and blocked == 1:
            return 100    # 冲四
        elif count == 3 and blocked == 0:
            return 50     # 活三
        elif count == 3 and blocked == 1:
            return 10     # 眠三
        elif count == 2 and blocked == 0:
            return 5      # 活二
        elif count == 2 and blocked == 1:
            return 1      # 眠二
        else:
            return 0
    
    def _evaluate_center_control(self, game):
        """评估中心区域控制和位置价值"""
        board = game.board
        board_size = len(board)
        center = board_size // 2
        score = 0
        
        # 检查中心区域的控制情况
        center_range = 2
        for i in range(max(0, center - center_range), min(board_size, center + center_range + 1)):
            for j in range(max(0, center - center_range), min(board_size, center + center_range + 1)):
                if board[i, j] == self.player_id:
                    score += 3
                elif board[i, j] != 0:
                    score -= 3
        
        # 位置价值评估（避免边角）
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == self.player_id:
                    # 边角位置给予惩罚
                    if (i == 0 or i == board_size-1) and (j == 0 or j == board_size-1):
                        score -= 5
                    # 边缘位置给予轻微惩罚
                    elif i == 0 or i == board_size-1 or j == 0 or j == board_size-1:
                        score -= 2
                    # 中心位置给予奖励
                    else:
                        distance_to_center = abs(i - center) + abs(j - center)
                        score += max(0, 5 - distance_to_center)
        
        return score
    
    def _evaluate_board_balance(self, game):
        """评估棋盘平衡性"""
        board = game.board
        board_size = len(board)
        score = 0
        
        # 检查棋子的分布是否合理
        player_pieces = 0
        opponent_pieces = 0
        
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == self.player_id:
                    player_pieces += 1
                elif board[i, j] != 0:
                    opponent_pieces += 1
        
        # 如果棋子数量差异过大，给予惩罚
        if abs(player_pieces - opponent_pieces) > 3:
            if player_pieces < opponent_pieces:
                score -= 20  # 我方棋子太少
            else:
                score += 10   # 我方棋子较多，但不要过度奖励
        
        return score
    
    def _evaluate_snake(self, game):
        """评估贪吃蛇位置（优化版，生存优先）"""
        if not hasattr(game, 'snake1') or not hasattr(game, 'snake2'):
            return 0
        
        # 基础分数：长度差异
        len1 = len(game.snake1) if game.alive1 else 0
        len2 = len(game.snake2) if game.alive2 else 0
        
        if self.player_id == 1:
            base_score = (len1 - len2) * 10
        else:
            base_score = (len2 - len1) * 10
        
        # 获取我的蛇头
        my_snake = game.snake1 if self.player_id == 1 else game.snake2
        my_head = my_snake[0] if my_snake else None
        
        if not my_head:
            return -1e6  # 死亡极大惩罚
        
        # 空间评估（最高优先级）
        safe_area = self._count_safe_area(my_head, game)
        if safe_area <= 2:
            return -1e5  # 死路极大惩罚
        elif safe_area <= 5:
            return -5000  # 空间极小
        elif safe_area <= 10:
            return -1000  # 空间较小
        
        # 空间越大分数越高
        area_score = safe_area * 20
        
        # 食物分数（仅在空间足够大时考虑）
        food_score = 0
        if safe_area > 10 and game.foods:
            for food in game.foods:
                dist = abs(my_head[0] - food[0]) + abs(my_head[1] - food[1])
                if dist <= 5:  # 只考虑较近的食物
                    food_score += max(0, 100 - dist * 15)
        
        # 位置优势（中心偏好）
        center_x, center_y = game.board_size // 2, game.board_size // 2
        distance_to_center = abs(my_head[0] - center_x) + abs(my_head[1] - center_y)
        position_score = max(0, 50 - distance_to_center * 5)
        
        # 优先级：空间 > 食物 > 位置 > 长度
        return area_score + food_score + position_score + base_score
    

    
    def _count_safe_area(self, head, game):
        """BFS统计蛇头可达的安全格子数"""
        from collections import deque
        if not head:
            return 0
        
        visited = set()
        queue = deque([head])
        board_size = game.board_size
        my_snake = set(game.snake1 if self.player_id == 1 else game.snake2)
        opponent_snake = set(game.snake2 if self.player_id == 1 else game.snake1)
        
        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                npos = (nx, ny)
                if (0 <= nx < board_size and 0 <= ny < board_size and 
                    npos not in visited and npos not in my_snake and npos not in opponent_snake):
                    queue.append(npos)
        
        return len(visited)
    
    def reset(self):
        """重置Bot状态"""
        super().reset()
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取Bot详细信息
        
        返回的信息包括：
        - 基础信息：名称、类型、描述
        - 策略信息：搜索深度、Alpha-Beta剪枝
        - 性能统计：评估节点数、剪枝次数、转置表大小
        - 时间控制：超时设置
        
        Returns:
            包含Bot详细信息的字典
        """
        info = super().get_info()
        info.update({
            'type': 'Minimax',
            'description': '使用改进的Minimax算法的Bot，具有防守优先策略',
            'strategy': f'Minimax with depth {self.max_depth}',
            'alpha_beta': self.use_alpha_beta,
            'timeout': self.timeout,
            'nodes_evaluated': self.nodes_evaluated,
            'pruning_count': self.pruning_count,
            'transposition_table_size': len(self.transposition_table),
            'features': [
                '防守优先策略',
                '智能进攻选择',
                '战略布局思考',
            ]
        })
        return info 