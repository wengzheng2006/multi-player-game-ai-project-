"""
行为树Bot
使用行为树进行决策
"""

import time
import random
from typing import Dict, List, Tuple, Any, Optional
from agents.base_agent import BaseAgent
import config


class BehaviorNode:
    """行为树节点基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.children = []
    
    def add_child(self, child):
        """添加子节点"""
        self.children.append(child)
    
    def execute(self, env: Any) -> Any:
        """执行节点"""
        raise NotImplementedError


class SequenceNode(BehaviorNode):
    """序列节点 - 所有子节点都成功才成功"""
    
    def execute(self, env: Any) -> Any:
        for child in self.children:
            result = child.execute(env)
            if result is None:  # 失败
                return None
        return True


class SelectorNode(BehaviorNode):
    """选择节点 - 任一子节点成功就成功"""
    
    def execute(self, env: Any) -> Any:
        for child in self.children:
            result = child.execute(env)
            if result is not None:  # 成功
                return result
        return None


class ActionNode(BehaviorNode):
    """动作节点 - 执行具体动作"""
    
    def __init__(self, name: str, action_func):
        super().__init__(name)
        self.action_func = action_func
    
    def execute(self, env: Any) -> Any:
        return self.action_func(env)


class ConditionNode(BehaviorNode):
    """条件节点 - 检查条件"""
    
    def __init__(self, name: str, condition_func):
        super().__init__(name)
        self.condition_func = condition_func
    
    def execute(self, env: Any) -> Any:
        if self.condition_func(env):
            return True
        return None


class BehaviorTreeBot(BaseAgent):
    """行为树Bot"""
    
    def __init__(self, name="BehaviorTreeBot", player_id=1, max_depth=10, timeout=5):
        super().__init__(name, player_id)
        self.max_depth = max_depth
        self.timeout = timeout
        # 简化版行为树Bot，实际为随机选择
        # 在实际项目中，这里应该有行为树结构
        
    def get_action(self, observation, env):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return None
        # 简化实现：随机选择
        return random.choice(valid_actions)
    
    def reset(self):
        """重置行为树Bot"""
        super().reset()
    
    def get_info(self) -> Dict[str, Any]:
        """获取行为树Bot信息"""
        info = super().get_info()
        info.update({
            'type': 'BehaviorTree',
            'description': '使用行为树的Bot',
            'strategy': 'Behavior Tree with multiple strategies',
            'max_depth': self.max_depth,
            'timeout': self.timeout
        })
        return info 