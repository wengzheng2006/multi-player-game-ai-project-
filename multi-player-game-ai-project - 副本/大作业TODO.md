# 多人游戏AI框架 - 大作业说明

## 📝 项目概述

这是一个基于OpenAI Gym风格的多人游戏AI对战框架项目，由Cursor AI工具自动生成。项目目前支持五子棋和贪吃蛇游戏，提供图形界面和命令行两种模式，包含多种AI算法实现。

**注意**：此项目是教学用途，内部包含一些故意设置的bug和不完善的功能，需要学生修复和改进。

## 🎯 作业目标

1. **发现并修复现有项目中的bug**
2. **完善AI Bot实现**
3. **实现至少一个新游戏**
4. **添加新的AI算法**
5. **体验AI辅助编程工具的使用**

## 📋 作业要求

### 基础要求

#### 1. Bug修复与项目完善

**已知可能存在的问题区域**：
- [ ] 导入模块错误和路径问题
- [ ] 游戏逻辑bug（胜负判断、边界检查等）
- [ ] AI算法实现不完整或有逻辑错误
- [ ] 图形界面显示问题
- [ ] 内存泄漏或性能问题

**具体检查项目**：
1. **模块导入检查**
   - 检查所有import语句是否正确
   - 确保包结构和__init__.py文件完整
   - 修复相对导入和绝对导入问题

2. **游戏逻辑检查**
   - 五子棋胜负判断逻辑
   - 贪吃蛇碰撞检测
   - 边界条件处理
   - 游戏状态转换

3. **AI算法检查**
   - Minimax算法的alpha-beta剪枝
   - MCTS算法的选择、扩展、模拟、回传
   - 贪吃蛇AI的路径规划
   - 随机选择的均匀性

#### 2. AI Bot完善

**需要完善的AI**：
- [ ] **MinimaxBot**：实现完整的alpha-beta剪枝
- [ ] **MCTSBot**：完善蒙特卡洛树搜索算法
- [ ] **SnakeAI**：改进贪吃蛇专用AI算法

**具体任务**：
1. **MinimaxBot改进**
   ```python
   # 需要实现的功能
   - 完整的alpha-beta剪枝
   - 动态深度调整
   - 启发式评估函数
   - 时间控制机制
   ```

2. **MCTSBot改进**
   ```python
   - UCB1选择策略
   - 节点扩展策略
   - 随机模拟策略
   - 结果回传机制
   ```

3. **SnakeAI改进**
   ```python
   - A*寻路算法
   - 安全性评估
   - 对手预测
   - 策略优化
   ```

#### 3. 测试与验证

- [ ] 所有测试用例通过
- [ ] AI对战功能正常
- [ ] 人机对战功能正常
- [ ] 图形界面正常显示
- [ ] 性能基准测试
- [ ] 边界情况测试

### 扩展要求 

#### 1. 实现新游戏

**推荐游戏选择**（选择其中一个或多个，或自选其他的游戏都可以）：

**A. 推箱子 (Sokoban)**
```python
# 实现要求
- 单人益智游戏改造为双人对战
- 目标：比对手更快完成关卡或推更多箱子
- 支持关卡编辑器
- 实现AI求解算法

# 文件结构
games/sokoban/
├── __init__.py
├── sokoban_game.py     # 游戏逻辑
├── sokoban_env.py      # 环境包装
└── levels.json         # 关卡数据
```

**B. 双人乒乓球 (Pong)**
```python
# 实现要求
- 经典乒乓球游戏
- 物理引擎（球的运动、碰撞）
- 双人实时对战
- AI预测球的轨迹

# 文件结构
games/pong/
├── __init__.py
├── pong_game.py        # 游戏逻辑
├── pong_env.py         # 环境包装
└── physics.py          # 物理引擎
```

**C. 分手厨房 (Cooking)**
```python
# 实现要求
- 双人合作/竞争做菜游戏
- 资源管理和时间压力
- 复杂状态空间
- 合作与竞争策略

# 文件结构
games/cooking/
├── __init__.py
├── cooking_game.py     # 游戏逻辑
├── cooking_env.py      # 环境包装
├── recipes.py          # 食谱系统
└── kitchen.py          # 厨房布局
```

**D. 双人吃豆人 (Pacman)**
```python
# 实现要求
- 一个玩家控制吃豆人，一个控制幽灵
- 地图导航和路径规划
- 实时追逐策略
- 道具系统

# 文件结构
games/pacman/
├── __init__.py
├── pacman_game.py      # 游戏逻辑
├── pacman_env.py       # 环境包装
├── maze.py             # 地图系统
└── entities.py         # 游戏实体
```

#### 2. 实现支撑新游戏的AI算法

**基础AI算法（至少选择1个）**：

**A. 基于规则的AI**
```python
# agents/ai_bots/rule_based_bot.py
class RuleBasedBot(BaseAgent):
    """基于规则的AI Bot"""
    def __init__(self, rules=None):
        # 实现基于if-else规则的决策
        self.rules = rules or self.default_rules()
    
    def get_action(self, observation, env):
        # 根据游戏状态应用规则
        for rule in self.rules:
            if rule.condition(observation, env):
                return rule.action(observation, env)
        return self.default_action(observation, env)
```

**B. 贪心算法AI**
```python
# agents/ai_bots/greedy_bot.py
class GreedyBot(BaseAgent):
    """贪心算法AI Bot"""
    def __init__(self, evaluation_function=None):
        # 实现贪心策略
        self.evaluate = evaluation_function or self.default_evaluate
    
    def get_action(self, observation, env):
        # 选择当前最优的动作
        valid_actions = env.get_valid_actions()
        best_action = max(valid_actions, 
                         key=lambda a: self.evaluate(a, observation, env))
        return best_action
```

**C. 搜索算法AI（进阶）**
```python
# agents/ai_bots/search_bot.py
class SearchBot(BaseAgent):
    """搜索算法AI Bot（A*、BFS等）"""
    def __init__(self, search_algorithm='bfs'):
        # 实现路径搜索算法
        self.search_type = search_algorithm
    
    def get_action(self, observation, env):
        # 使用搜索算法找到最优路径
        path = self.search_optimal_path(observation, env)
        return path[0] if path else self.random_action(env)
```

**D. 启发式AI（进阶选做）**
```python
# agents/ai_bots/heuristic_bot.py
class HeuristicBot(BaseAgent):
    """启发式AI Bot"""
    def __init__(self, heuristic_functions=None):
        # 结合多种启发式函数
        self.heuristics = heuristic_functions or []
    
    def get_action(self, observation, env):
        # 综合多个启发式函数的结果
        scores = self.calculate_action_scores(observation, env)
        return max(scores, key=scores.get)
```

**E. 强化学习AI（挑战选做）**
```python
# agents/ai_bots/rl_bot.py
class RLBot(BaseAgent):
    """强化学习AI Bot"""
    def __init__(self, algorithm='DQN'):
        # 实现DQN, Q-learning, 或简化的策略梯度算法
        self.algorithm = algorithm
        self.q_table = {}  # 简单Q-learning
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
    
    def get_action(self, observation, env):
        # Q-learning策略选择
        if random.random() < self.epsilon:
            return random.choice(env.get_valid_actions())
        else:
            return self.get_best_action(observation, env)
    
    def update_q_value(self, state, action, reward, next_state):
        # Q值更新
        current_q = self.q_table.get((state, action), 0)
        max_next_q = max([self.q_table.get((next_state, a), 0) 
                         for a in self.get_possible_actions(next_state)])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
```

**F. 大语言模型AI（创新选做）**
```python
# agents/ai_bots/llm_bot.py
class LLMBot(BaseAgent):
    """大语言模型AI Bot"""
    def __init__(self, model_name='gpt-3.5-turbo', api_key=None):
        # 接入OpenAI API或本地模型（如Ollama）
        self.model_name = model_name
        self.api_key = api_key
        # 也可以使用免费的本地模型
    
    def get_action(self, observation, env):
        # 将游戏状态转换为文本描述
        game_description = self.observation_to_text(observation, env)
        
        # 构建提示词
        prompt = f"""
        你是一个{env.__class__.__name__}游戏AI。
        当前游戏状态：{game_description}
        可选动作：{env.get_valid_actions()}
        
        请分析当前局势，选择最佳动作。只返回动作坐标，如(row, col)。
        """
        
        # 调用LLM获取决策
        response = self.call_llm(prompt)
        action = self.parse_action_from_response(response, env)
        
        return action if action else random.choice(env.get_valid_actions())
    
    def call_llm(self, prompt):
        # 实现LLM调用逻辑
        # 可以使用OpenAI API、本地模型或简化的规则模拟
        pass
```


## 🛠️ 开发流程

### 第一阶段：环境搭建与Bug修复

1. **环境配置**
   ```bash
   # Fork项目到个人GitHub
   git clone https://github.com/your-username/multi-player-game-ai-project
   cd multi-player-game-ai-project
   
   # 创建虚拟环境
   python -m venv game_ai_env
   source game_ai_env/bin/activate  # Linux/Mac
   # 或 game_ai_env\Scripts\activate  # Windows
   
   # 安装依赖
   pip install -r requirements.txt
   ```

2. **初始测试**
   ```bash
   # 运行测试套件
   python test_project.py
   
   # 启动图形界面
   python start_games.py
   
   # 命令行测试
   python main.py --game gomoku --player1 human --player2 minimax
   ```

3. **Bug发现与修复**
   - 系统性测试所有功能
   - 记录发现的bug和修复方案
   - 提交修复的Git commit

### 第二阶段：功能完善 

1. **AI算法完善**
   - 深入理解现有AI算法
   - 实现缺失的功能
   - 优化算法性能

2. **代码质量改进**
   - 添加注释和文档
   - 代码格式化和重构
   - 异常处理完善

### 第三阶段：新功能开发

1. **新游戏实现**
   - 选择要实现的游戏
   - 设计游戏规则和状态表示
   - 实现游戏逻辑和环境

2. **新AI开发**
   - 选择AI算法
   - 实现算法逻辑
   - 训练和调优

3. **集成测试**
   - 新功能测试
   - 性能基准测试
   - 用户体验测试

### 第四阶段：评估与展示 
1. **性能评估**
   - AI算法对战评估
   - 运行效率测试
   - 用户体验评估


2. **展示准备**
   - 演示视频制作
   - 项目展示PPT
   - 代码展示

## 📊 评估标准

### AI Bot性能评估

**评估方法**：
1. **对战测试**
   ```python
   # 运行评估脚本
   python evaluate_ai.py --agents minimax mcts random --games 1000
   ```

2. **计算指标**
   - 胜率 (Win Rate)
   - 平均步数 (Average Moves)
   - 平均思考时间 (Average Think Time)
   - 策略稳定性 (Strategy Consistency)

3. **基准对比**
   - 与标准算法对比
   - 与人类玩家对比
   - 与其他团队AI对比

### 代码质量评估

**评估维度**：
- 代码结构和组织
- 注释和文档完整性
- 错误处理和边界情况
- 性能优化

## 📖 报告要求

### 技术报告内容

#### 1. 项目分析与Bug修复 (60%)
- **问题发现**：详细描述发现的bug类型和位置
- **修复方案**：解释修复思路和实现方法
- **验证测试**：展示修复前后的测试结果

#### 2. 新游戏设计与实现 (20%)
- **游戏选择理由**：为什么选择这个游戏
- **规则设计**：详细的游戏规则和状态定义
- **技术实现**：关键算法和数据结构
- **挑战与解决**：遇到的技术难点和解决方案

#### 3. AI算法设计与优化 (10%)
- **算法选择**：选择的AI算法及其优势
- **实现细节**：核心算法的实现逻辑
- **参数调优**：参数选择和调优过程
- **性能分析**：算法复杂度和实际性能

#### 4. AI辅助编程体验 (10%)
- **工具使用**：使用的AI编程工具（Cursor等）
- **效率提升**：AI工具对开发效率的影响
- **问题与限制**：AI工具的局限性和问题
- **经验总结**：使用AI工具的经验和建议

### 代码提交要求

1. **Git提交规范**
   ```bash
   # 提交信息格式
   feat: 添加新功能
   fix: 修复bug
   docs: 更新文档
   style: 代码格式调整
   refactor: 代码重构
   test: 添加测试
   chore: 其他修改
   ```

2. **分支管理**
   ```bash
   main              # 主分支
   ├── feature/bug-fixes     # bug修复分支
   ├── feature/new-game      # 新游戏开发分支
   ├── feature/ai-enhancement # AI改进分支
   └── feature/ui-improvement # 界面改进分支
   ```

3. **提交频率**
   - 每个功能完成后及时提交
   - 重要节点打tag标记
   - 保持提交历史清晰

## 🏆 最终展示

### 展示形式
- **时间**：最后一次课
- **形式**：各组互相展示和试玩
- **评分**：组间互评打分

### 展示内容
1. **项目演示**
   - 新游戏试玩
   - AI对战展示
   - 特色功能介绍

2. **技术分享** 
   - 关键技术点
   - 创新亮点
   - 开发经验

3. **互动试玩** 
   - 其他组试玩体验
   - 实时反馈和建议
   - 技术交流


### 奖励机制
- **排名前4的组**：分别获得4分、3分、2分、1分的加分

## 📚 参考资源

### 学习材料
- [Minimax算法详解](https://en.wikipedia.org/wiki/Minimax)
- [蒙特卡洛树搜索](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [强化学习入门](https://spinningup.openai.com/)

### 开发工具
- **IDE**: VSCode, PyCharm, Cursor
- **版本控制**: Git, GitHub
- **调试工具**: Python debugger, print调试
- **性能分析**: cProfile, line_profiler

### 测试框架
- **单元测试**: pytest
- **性能测试**: timeit
- **集成测试**: 自定义测试框架

## ❓ 常见问题

### Q1: 如何发现项目中的bug？
A: 建议采用以下方法：
1. 系统性运行所有功能
2. 边界条件测试（如满棋盘、空棋盘）
3. 压力测试（大量游戏对战）
4. 异常输入测试
5. 阅读代码查找逻辑错误

### Q2: 新游戏实现的难度评估？
A: 难度从低到高：
- 简单：井字棋、连连看
- 中等：推箱子、乒乓球
- 困难：分手厨房、实时策略游戏

### Q3: AI算法如何选择？
A: 根据游戏特点选择：
- 完美信息博弈：Minimax, MCTS
- 实时游戏：神经网络, 强化学习
- 复杂状态空间：深度学习
- 创新尝试：大语言模型

### Q4: 如何优化AI性能？
A: 优化策略：
1. 算法层面：剪枝、启发式函数
2. 实现层面：缓存、并行计算
3. 参数调优：深度、模拟次数
4. 硬件优化：GPU加速、内存管理


---

**预祝各组在项目开发中取得优异成绩！** 🎉 