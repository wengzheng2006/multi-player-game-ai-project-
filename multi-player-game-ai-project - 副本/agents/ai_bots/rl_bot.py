import random
from agents.base_agent import BaseAgent

class RLBot(BaseAgent):
    def __init__(self, name="RLBot", player_id=1):
        super().__init__(name, player_id)
        # 简化版RL Bot，实际为随机选择
        # 在实际项目中，这里应该有Q-table或神经网络
        
    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        # 简化实现：随机选择
        return random.choice(valid_actions) 