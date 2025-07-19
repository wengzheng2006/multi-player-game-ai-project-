#!/usr/bin/env python3
"""
简单的幽灵AI测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.pacman.pacman_env import PacmanEnv
from games.pacman.entities import Direction

def main():
    print("开始简单幽灵AI测试...")
    
    # 创建游戏环境
    game_config = {
        'ghost_ai_type': 'aggressive',
        'ghost_aggression_level': 2.0
    }
    
    try:
        env = PacmanEnv(game_config)
        print("✅ 游戏环境创建成功")
        
        # 检查幽灵AI
        print(f"幽灵数量: {len(env.game.ghosts)}")
        print(f"幽灵AI数量: {len(env.game.ghost_ais)}")
        
        if env.game.ghost_ais:
            print("✅ 幽灵AI已初始化")
            for i, ghost_ai in enumerate(env.game.ghost_ais):
                if ghost_ai:
                    print(f"幽灵 {i} AI类型: {type(ghost_ai).__name__}")
        else:
            print("❌ 幽灵AI未初始化")
        
        # 测试幽灵移动
        print("\n测试幽灵移动...")
        initial_positions = []
        for ghost in env.game.ghosts:
            if ghost:
                initial_positions.append(ghost.get_position())
        
        print(f"初始位置: {initial_positions}")
        
        # 运行几步
        for step in range(10):
            # 玩家不移动
            observation, reward, done, truncated, info = env.step_pacman(Direction.NONE)
            
            # 检查位置变化
            current_positions = []
            for ghost in env.game.ghosts:
                if ghost:
                    current_positions.append(ghost.get_position())
            
            print(f"步骤 {step}: 位置 {current_positions}")
            
            # 检查是否有移动
            for i, (initial, current) in enumerate(zip(initial_positions, current_positions)):
                if initial != current:
                    print(f"✅ 幽灵 {i} 从 {initial} 移动到 {current}")
            
            initial_positions = current_positions.copy()
        
        print("\n🎉 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 