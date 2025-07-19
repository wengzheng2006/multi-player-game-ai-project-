import random
import time
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent


class RandomBot(BaseAgent):
    """随机选择动作的Bot"""
    
    def __init__(self, name: str = "RandomBot", player_id: int = 1, 
                 seed: Optional[int] = None, timeout: float = 1.0):
        super().__init__(name, player_id)
        self.seed = seed
        self.timeout = timeout
        if self.seed is not None:
            random.seed(seed)
    
    def get_action(self, observation: Any, env: Any) -> Any:
        """
        随机选择有效动作
        
        Args:
            observation: 当前观察
            env: 环境对象
            
        Returns:
            随机选择的动作
        """
        start_time = time.time()
        
        # 获取有效动作
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        # 随机选择动作
        action = random.choice(valid_actions)
        
        # 更新统计
        move_time = time.time() - start_time
        self.total_moves += 1
        self.total_time += move_time
        
        return action
    
    def reset(self):
        """重置Bot状态"""
        super().reset()
        if self.seed is not None:
            random.seed(self.seed)
    
    def get_info(self) -> Dict[str, Any]:
        """获取Bot信息"""
        info = super().get_info()
        info.update({
            'type': 'Random',
            'description': '随机选择动作的Bot',
            'strategy': 'Random selection',
            'seed': self.seed,
            'timeout': self.timeout
        })
        return info 