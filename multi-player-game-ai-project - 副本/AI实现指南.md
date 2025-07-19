# AI实现指南

## 🎯 基础AI算法实现指导

本指南提供了几种基础AI算法的详细实现方法，适合初学者循序渐进地学习游戏AI开发。

## 📚 AI算法难度等级

### 🟢 入门级（推荐新手）
1. **改进随机AI** - 在随机选择的基础上添加基本规则
2. **基于规则的AI** - 使用if-else条件判断的策略AI

### 🟡 中级
3. **贪心算法AI** - 每步选择当前最优动作
4. **简单搜索AI** - 使用BFS或DFS进行路径搜索

### 🔴 进阶级（选做）
5. **启发式AI** - 结合多种启发式函数
6. **强化学习AI** - 使用Q-learning等方法
7. **大语言模型AI** - 接入GPT等大模型

## 🔧 1. 改进随机AI

### 基本思路
- 在随机选择的基础上，过滤掉明显不好的动作
- 添加安全性检查
- 简单的位置偏好

### 实现步骤

```python
class ImprovedRandomBot(BaseAgent):
    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        
        # 1. 过滤危险动作
        safe_actions = self.filter_dangerous_actions(valid_actions, observation, env)
        
        # 2. 选择偏好动作
        preferred_actions = self.get_preferred_actions(safe_actions, observation, env)
        
        # 3. 随机选择
        if preferred_actions:
            return random.choice(preferred_actions)
        elif safe_actions:
            return random.choice(safe_actions)
        else:
            return random.choice(valid_actions)
```

### 五子棋示例
```python
def filter_dangerous_actions(self, actions, observation, env):
    """过滤危险动作：避免让对手立即获胜"""
    safe_actions = []
    board = observation['board']
    opponent = 3 - self.player_id
    
    for action in actions:
        # 检查这个动作是否会让对手在下一步获胜
        if not self.enables_opponent_win(action, board, opponent):
            safe_actions.append(action)
    
    return safe_actions

def get_preferred_actions(self, actions, observation, env):
    """选择偏好位置：中心区域优先"""
    board = observation['board']
    center = board.shape[0] // 2
    
    # 按距离中心的远近排序
    actions_with_distance = []
    for action in actions:
        row, col = action
        distance = abs(row - center) + abs(col - center)
        actions_with_distance.append((action, distance))
    
    # 选择最近的1/3动作
    actions_with_distance.sort(key=lambda x: x[1])
    num_preferred = max(1, len(actions_with_distance) // 3)
    
    return [action for action, _ in actions_with_distance[:num_preferred]]
```

## 🎯 2. 基于规则的AI

### 基本思路
- 定义一系列优先级规则
- 按优先级顺序检查规则
- 执行第一个匹配的规则

### 规则设计框架

```python
class RuleBasedAI(BaseAgent):
    def __init__(self):
        super().__init__()
        self.rules = [
            self.rule_win_immediately,      # 优先级1：立即获胜
            self.rule_block_opponent,       # 优先级2：阻止对手获胜
            self.rule_create_threat,        # 优先级3：创造威胁
            self.rule_improve_position,     # 优先级4：改善位置
            self.rule_safe_move,           # 优先级5：安全移动
            self.rule_random              # 优先级6：随机选择
        ]
    
    def get_action(self, observation, env):
        for rule in self.rules:
            action = rule(observation, env)
            if action is not None:
                return action
        
        # 兜底：随机选择
        return random.choice(env.get_valid_actions())
```

### 五子棋规则示例

```python
def rule_win_immediately(self, observation, env):
    """规则1：如果能获胜，立即获胜"""
    valid_actions = env.get_valid_actions()
    board = observation['board']
    
    for action in valid_actions:
        if self.check_win_after_move(board, action, self.player_id):
            return action
    return None

def rule_block_opponent(self, observation, env):
    """规则2：阻止对手获胜"""
    valid_actions = env.get_valid_actions()
    board = observation['board']
    opponent = 3 - self.player_id
    
    for action in valid_actions:
        if self.check_win_after_move(board, action, opponent):
            return action
    return None

def rule_create_threat(self, observation, env):
    """规则3：创造威胁（连成3子或4子）"""
    valid_actions = env.get_valid_actions()
    board = observation['board']
    
    best_action = None
    max_threat = 0
    
    for action in valid_actions:
        threat_level = self.calculate_threat_level(board, action, self.player_id)
        if threat_level > max_threat:
            max_threat = threat_level
            best_action = action
    
    return best_action if max_threat > 0 else None
```

### 贪吃蛇规则示例

```python
def rule_avoid_collision(self, observation, env):
    """规则1：避免碰撞"""
    valid_actions = env.get_valid_actions()
    safe_actions = []
    
    for action in valid_actions:
        if not self.will_collide(observation, action):
            safe_actions.append(action)
    
    return safe_actions if safe_actions else None

def rule_eat_food(self, observation, env):
    """规则2：朝最近的食物移动"""
    snake = observation[f'snake{self.player_id}']
    foods = observation['foods']
    
    if not foods:
        return None
    
    head = snake[0]
    nearest_food = min(foods, key=lambda f: self.manhattan_distance(head, f))
    
    return self.get_direction_to_target(head, nearest_food)

def rule_stay_center(self, observation, env):
    """规则3：保持在中心区域"""
    snake = observation[f'snake{self.player_id}']
    head = snake[0]
    board_size = observation['board'].shape[0]
    center = (board_size // 2, board_size // 2)
    
    return self.get_direction_to_target(head, center)
```

## 🎯 3. 贪心算法AI

### 基本思路
- 为每个可能的动作计算分数
- 选择分数最高的动作
- 分数函数是关键

### 实现框架

```python
class GreedyAI(BaseAgent):
    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        best_action = None
        best_score = float('-inf')
        
        for action in valid_actions:
            score = self.evaluate_action(action, observation, env)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def evaluate_action(self, action, observation, env):
        """评估函数：核心在于如何计算分数"""
        return self.calculate_score(action, observation, env)
```

### 评估函数设计

#### 五子棋评估函数
```python
def calculate_score(self, action, observation, env):
    """五子棋位置评估"""
    board = observation['board']
    row, col = action
    score = 0
    
    # 1. 连子奖励
    score += self.count_connections(board, row, col, self.player_id) * 10
    
    # 2. 阻止对手奖励
    opponent = 3 - self.player_id
    score += self.count_connections(board, row, col, opponent) * 8
    
    # 3. 位置奖励（中心区域更好）
    center = board.shape[0] // 2
    distance_to_center = abs(row - center) + abs(col - center)
    score += (10 - distance_to_center)
    
    # 4. 威胁等级
    score += self.calculate_threat_potential(board, row, col, self.player_id) * 5
    
    return score

def count_connections(self, board, row, col, player):
    """计算在某位置可以形成的连接数"""
    max_connections = 0
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    
    for dr, dc in directions:
        count = 1  # 当前位置
        
        # 正方向计数
        r, c = row + dr, col + dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r += dr
            c += dc
        
        # 负方向计数
        r, c = row - dr, col - dc
        while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
               board[r, c] == player):
            count += 1
            r -= dr
            c -= dc
        
        max_connections = max(max_connections, count)
    
    return max_connections
```

#### 贪吃蛇评估函数
```python
def calculate_score(self, action, observation, env):
    """贪吃蛇动作评估"""
    snake = observation[f'snake{self.player_id}']
    foods = observation['foods']
    head = snake[0]
    
    # 模拟移动后的新位置
    new_head = (head[0] + action[0], head[1] + action[1])
    score = 0
    
    # 1. 食物距离奖励（距离越近分数越高）
    if foods:
        min_food_distance = min(self.manhattan_distance(new_head, food) for food in foods)
        score += 100 / (min_food_distance + 1)
    
    # 2. 安全性奖励
    if self.is_safe_position(new_head, observation, env):
        score += 50
    else:
        score -= 1000  # 危险位置严重惩罚
    
    # 3. 空间奖励（避免被困）
    free_space = self.count_free_space_around(new_head, observation)
    score += free_space * 2
    
    # 4. 中心位置奖励
    board_size = observation['board'].shape[0]
    center = board_size // 2
    distance_to_center = abs(new_head[0] - center) + abs(new_head[1] - center)
    score += 10 / (distance_to_center + 1)
    
    return score
```

## 🔍 4. 简单搜索AI

### BFS搜索示例

```python
from collections import deque

class SearchAI(BaseAgent):
    def get_action(self, observation, env):
        """使用BFS寻找到目标的最短路径"""
        current_pos = self.get_current_position(observation)
        target_pos = self.get_target_position(observation)
        
        path = self.bfs_search(current_pos, target_pos, observation, env)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self.position_to_action(current_pos, next_pos)
        
        return random.choice(env.get_valid_actions())
    
    def bfs_search(self, start, target, observation, env):
        """BFS搜索算法"""
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            position, path = queue.popleft()
            
            if position == target:
                return path
            
            # 探索邻居节点
            for neighbor in self.get_neighbors(position, observation, env):
                if neighbor not in visited and self.is_valid_position(neighbor, observation, env):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # 未找到路径
    
    def get_neighbors(self, position, observation, env):
        """获取邻居位置"""
        row, col = position
        neighbors = []
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # 上下左右
            new_row, new_col = row + dr, col + dc
            neighbors.append((new_row, new_col))
        
        return neighbors
    
    def is_valid_position(self, position, observation, env):
        """检查位置是否有效"""
        row, col = position
        board = observation['board']
        
        # 检查边界
        if not (0 <= row < board.shape[0] and 0 <= col < board.shape[1]):
            return False
        
        # 检查是否有障碍物
        if board[row, col] != 0:  # 0表示空位
            return False
        
        return True
```

### A*搜索示例

```python
import heapq

class AStarAI(BaseAgent):
    def a_star_search(self, start, target, observation, env):
        """A*搜索算法"""
        open_set = [(0, start, [start])]
        closed_set = set()
        
        while open_set:
            f_score, position, path = heapq.heappop(open_set)
            
            if position in closed_set:
                continue
            
            closed_set.add(position)
            
            if position == target:
                return path
            
            for neighbor in self.get_neighbors(position, observation, env):
                if neighbor in closed_set or not self.is_valid_position(neighbor, observation, env):
                    continue
                
                g_score = len(path)  # 实际距离
                h_score = self.heuristic(neighbor, target)  # 启发式距离
                f_score = g_score + h_score
                
                heapq.heappush(open_set, (f_score, neighbor, path + [neighbor]))
        
        return []
    
    def heuristic(self, pos1, pos2):
        """启发式函数：曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

## 🧪 测试和调优

### 性能测试

```bash
# 测试基础AI性能
python evaluate_ai.py --agents improved_random rule_based greedy_snake --benchmark --games 100

# 比较不同AI
python evaluate_ai.py --agents random improved_random rule_based --compare --games 50

# 生成详细报告
python evaluate_ai.py --agents rule_based --benchmark --games 200 --plot --save rule_based_results.json
```

### 调优建议

1. **参数调优**
   - 调整评估函数中各项的权重
   - 修改搜索深度和宽度
   - 优化规则优先级

2. **性能优化**
   - 添加缓存避免重复计算
   - 使用更高效的数据结构
   - 限制搜索时间

3. **策略改进**
   - 分析失败的游戏找出问题
   - 添加新的规则或评估项
   - 结合多种算法的优势

## 💡 实现技巧

### 1. 调试技巧
```python
def get_action(self, observation, env):
    action = self.calculate_best_action(observation, env)
    
    # 调试输出
    if self.debug:
        print(f"Player {self.player_id} chose action {action}")
        print(f"Board state: {observation['board']}")
    
    return action
```

### 2. 异常处理
```python
def get_action(self, observation, env):
    try:
        return self.smart_action(observation, env)
    except Exception as e:
        print(f"Error in {self.name}: {e}")
        # 降级到简单策略
        return random.choice(env.get_valid_actions())
```

### 3. 时间控制
```python
import time

def get_action(self, observation, env):
    start_time = time.time()
    timeout = 5.0  # 5秒超时
    
    for depth in range(1, 10):
        if time.time() - start_time > timeout:
            break
        
        action = self.search_with_depth(observation, env, depth)
    
    return action
```

## 🚀 5. 强化学习AI（挑战选做）

### 基本思路
- 通过与环境交互学习最优策略
- 使用Q-learning或简化的深度强化学习
- 需要大量训练数据和时间

### Q-Learning实现示例

```python
import pickle
import numpy as np

class QLearningBot(BaseAgent):
    def __init__(self, name="QLearningBot", player_id=1):
        super().__init__(name, player_id)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # 探索率
        self.training = True
        
    def get_action(self, observation, env):
        state = self.observation_to_state(observation)
        valid_actions = env.get_valid_actions()
        
        # ε-贪心策略
        if self.training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 选择Q值最高的动作
        q_values = {}
        for action in valid_actions:
            q_values[action] = self.q_table.get((state, action), 0.0)
        
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    def observation_to_state(self, observation):
        """将观察转换为状态表示"""
        # 简化状态表示（可以改进）
        if 'board' in observation:
            # 五子棋：使用棋盘的简化表示
            board = observation['board']
            return tuple(board.flatten())
        else:
            # 贪吃蛇：使用关键信息
            snake = observation.get(f'snake{self.player_id}', [])
            foods = observation.get('foods', [])
            return (tuple(snake[:3]), tuple(foods[:2]))  # 简化表示
    
    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        current_q = self.q_table.get((state, action), 0.0)
        
        # 获取下一状态的最大Q值
        next_actions = self.get_possible_actions(next_state)
        max_next_q = 0.0
        if next_actions:
            max_next_q = max([self.q_table.get((next_state, a), 0.0) 
                             for a in next_actions])
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
    
    def train(self, env, episodes=1000):
        """训练Q-learning智能体"""
        print(f"开始训练 {episodes} 轮...")
        
        for episode in range(episodes):
            observation, _ = env.reset()
            state = self.observation_to_state(observation)
            total_reward = 0
            
            while not env.is_terminal():
                action = self.get_action(observation, env)
                next_obs, reward, done, info = env.step(action)
                next_state = self.observation_to_state(next_obs)
                
                # 更新Q值
                self.update_q_value(state, action, reward, next_state)
                
                state = next_state
                observation = next_obs
                total_reward += reward
                
                if done:
                    break
            
            # 衰减探索率
            if episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.995)
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")
    
    def save_model(self, filename):
        """保存训练好的模型"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_model(self, filename):
        """加载训练好的模型"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            self.training = False
            print(f"成功加载模型: {filename}")
        except FileNotFoundError:
            print(f"模型文件不存在: {filename}")
```

### 训练脚本示例

```python
def train_q_learning_bot():
    """训练Q-learning智能体"""
    from games.gomoku import GomokuEnv
    from agents import RandomBot
    
    env = GomokuEnv(board_size=9, win_length=5)
    q_bot = QLearningBot("Q-Learning Bot", 1)
    random_bot = RandomBot("Random Bot", 2)
    
    # 训练阶段
    q_bot.train(env, episodes=5000)
    
    # 保存模型
    q_bot.save_model("q_learning_gomoku.pkl")
    
    # 测试阶段
    q_bot.training = False
    q_bot.epsilon = 0  # 关闭探索
    
    # 评估性能
    wins = 0
    test_games = 100
    for _ in range(test_games):
        observation, _ = env.reset()
        while not env.is_terminal():
            if env.game.current_player == 1:
                action = q_bot.get_action(observation, env)
            else:
                action = random_bot.get_action(observation, env)
            observation, _, done, _ = env.step(action)
            if done:
                break
        
        if env.get_winner() == 1:
            wins += 1
    
    print(f"训练后胜率: {wins/test_games:.2%}")
```

## 🤖 6. 大语言模型AI（创新选做）

### 基本思路
- 将游戏状态转换为自然语言描述
- 使用提示工程指导大模型做决策
- 可以使用API或本地模型

### 基础实现框架

```python
import json
import requests
from typing import Optional

class LLMBot(BaseAgent):
    def __init__(self, name="LLMBot", player_id=1, model_type="openai"):
        super().__init__(name, player_id)
        self.model_type = model_type
        self.api_key = None  # 设置你的API密钥
        
    def get_action(self, observation, env):
        try:
            # 转换游戏状态为文字描述
            game_description = self.observation_to_text(observation, env)
            
            # 构建提示词
            prompt = self.build_prompt(game_description, env)
            
            # 调用大模型
            response = self.call_llm(prompt)
            
            # 解析回复获取动作
            action = self.parse_action(response, env)
            
            if action and action in env.get_valid_actions():
                return action
            else:
                # 如果解析失败，降级到规则策略
                return self.fallback_strategy(observation, env)
                
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return self.fallback_strategy(observation, env)
    
    def observation_to_text(self, observation, env):
        """将游戏状态转换为文字描述"""
        if hasattr(env, 'board_size'):  # 五子棋
            board = observation['board']
            description = f"棋盘大小: {board.shape[0]}x{board.shape[1]}\n"
            
            # 描述棋盘状态
            description += "当前棋盘状态:\n"
            for i in range(board.shape[0]):
                row_desc = ""
                for j in range(board.shape[1]):
                    if board[i, j] == 0:
                        row_desc += "·"
                    elif board[i, j] == 1:
                        row_desc += "●"
                    else:
                        row_desc += "○"
                description += f"{i:2d}|{row_desc}|\n"
            
            # 添加列标号
            col_numbers = "  " + "".join([str(i%10) for i in range(board.shape[1])])
            description += col_numbers + "\n"
            
            return description
            
        else:  # 贪吃蛇
            snake1 = observation.get('snake1', [])
            snake2 = observation.get('snake2', [])
            foods = observation.get('foods', [])
            
            description = f"贪吃蛇游戏状态:\n"
            description += f"你的蛇(玩家{self.player_id}): {snake1 if self.player_id == 1 else snake2}\n"
            description += f"对手的蛇: {snake2 if self.player_id == 1 else snake1}\n"
            description += f"食物位置: {foods}\n"
            
            return description
    
    def build_prompt(self, game_description, env):
        """构建给大模型的提示词"""
        game_name = env.__class__.__name__.replace('Env', '')
        valid_actions = env.get_valid_actions()
        
        prompt = f"""你是一个专业的{game_name}游戏AI。

{game_description}

你是玩家{self.player_id}。请分析当前局势并选择最佳动作。

可选动作: {valid_actions}

请按以下格式回复:
分析: [你的分析]
动作: (row, col)

注意:
1. 只能从可选动作中选择
2. 动作格式必须是(row, col)的形式
3. 优先考虑获胜机会
4. 其次考虑阻止对手获胜
5. 选择战略位置
"""
        return prompt
    
    def call_llm(self, prompt):
        """调用大语言模型"""
        if self.model_type == "openai":
            return self.call_openai(prompt)
        elif self.model_type == "ollama":
            return self.call_ollama(prompt)
        else:
            # 简化版：使用规则模拟大模型回复
            return self.simulate_llm_response(prompt)
    
    def call_openai(self, prompt):
        """调用OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        return response.json()["choices"][0]["message"]["content"]
    
    def call_ollama(self, prompt):
        """调用本地Ollama模型"""
        data = {
            "model": "llama2",  # 或其他模型
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        return response.json()["response"]
    
    def simulate_llm_response(self, prompt):
        """模拟大模型回复（用于测试）"""
        # 这里可以实现一个简单的规则来模拟大模型的回复
        return "分析: 选择中心位置\n动作: (4, 4)"
    
    def parse_action(self, response, env):
        """从大模型回复中解析动作"""
        import re
        
        # 使用正则表达式提取动作
        pattern = r'动作[:：]\s*\((\d+),\s*(\d+)\)'
        match = re.search(pattern, response)
        
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            return (row, col)
        
        # 尝试其他格式
        pattern2 = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(pattern2, response)
        if matches:
            row, col = int(matches[-1][0]), int(matches[-1][1])
            return (row, col)
        
        return None
    
    def fallback_strategy(self, observation, env):
        """降级策略"""
        # 如果LLM失败，使用简单规则
        valid_actions = env.get_valid_actions()
        if valid_actions:
            # 优先选择中心位置
            if hasattr(env, 'board_size'):
                center = env.board_size // 2
                for action in valid_actions:
                    row, col = action
                    if abs(row - center) <= 1 and abs(col - center) <= 1:
                        return action
            return random.choice(valid_actions)
        return None
```

### 使用示例

```python
def test_llm_bot():
    """测试LLM Bot"""
    from games.gomoku import GomokuEnv
    
    # 创建环境和AI
    env = GomokuEnv(board_size=9, win_length=5)
    
    # 使用模拟模式（无需API密钥）
    llm_bot = LLMBot("LLM Bot", 1, model_type="simulate")
    
    # 如果要使用真实API，需要设置密钥
    # llm_bot.api_key = "your-openai-api-key"
    # llm_bot.model_type = "openai"
    
    observation, _ = env.reset()
    action = llm_bot.get_action(observation, env)
    print(f"LLM选择的动作: {action}")
```

### 使用建议

1. **API费用控制**: 设置请求限制，避免过度调用
2. **提示词优化**: 多测试不同的提示词格式
3. **错误处理**: 必须有降级策略
4. **本地模型**: 可以使用Ollama等免费本地模型

## 📈 进阶方向

完成基础AI后，可以尝试：

1. **组合策略**: 结合多种算法的优势
2. **自适应AI**: 根据对手风格调整策略
3. **强化学习**: Q-learning、Actor-Critic等
4. **神经网络**: 深度学习方法
5. **多智能体学习**: 智能体之间的协作与竞争

记住：好的AI不是最复杂的算法，而是最适合特定游戏的策略！ 