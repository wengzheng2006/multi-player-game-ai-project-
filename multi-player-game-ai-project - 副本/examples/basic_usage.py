"""
基础使用示例
演示如何使用双人游戏AI框架
"""

from games.gomoku import GomokuGame, GomokuEnv
from agents import RandomBot, MinimaxBot, HumanAgent
from utils.game_utils import evaluate_agents, play_human_vs_ai

def basic_game_example():
    """基础游戏示例"""
    print("=== 基础游戏示例 ===")
    
    # 创建游戏
    game = GomokuGame(board_size=9, win_length=5)
    print("游戏创建成功")
    
    # 重置游戏
    state = game.reset()
    print("游戏重置成功")
    
    # 获取有效动作
    valid_actions = game.get_valid_actions()
    print(f"有效动作数量: {len(valid_actions)}")
    
    # 执行几步动作
    for i in range(3):
        if valid_actions:
            action = valid_actions[0]
            state, reward, done, info = game.step(action)
            print(f"第{i+1}步: 动作={action}, 奖励={reward}, 结束={done}")
            
            if done:
                break
                
            valid_actions = game.get_valid_actions()
    
    # 显示游戏状态
    print("当前游戏状态:")
    game.render()

def basic_env_example():
    """基础环境示例"""
    print("\n=== 基础环境示例 ===")
    
    # 创建环境
    env = GomokuEnv(board_size=9, win_length=5)
    print("环境创建成功")
    
    # 重置环境
    observation, info = env.reset()
    print("环境重置成功")
    
    # 获取有效动作
    valid_actions = env.get_valid_actions()
    print(f"有效动作数量: {len(valid_actions)}")
    
    # 执行几步动作
    for i in range(3):
        if valid_actions:
            action = valid_actions[0]
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"第{i+1}步: 动作={action}, 奖励={reward}, 结束={terminated}")
            
            if terminated or truncated:
                break
                
            valid_actions = env.get_valid_actions()
    
    # 显示环境状态
    print("当前环境状态:")
    env.render()

def agent_vs_agent_example():
    """智能体对战示例"""
    print("\n=== 智能体对战示例 ===")
    
    # 创建环境和智能体
    env = GomokuEnv(board_size=9, win_length=5)
    agent1 = RandomBot(name="随机Bot1", player_id=1)
    agent2 = MinimaxBot(name="MinimaxBot", player_id=2, max_depth=2)
    
    print(f"智能体1: {agent1.name}")
    print(f"智能体2: {agent2.name}")
    
    # 评估智能体
    results = evaluate_agents(env, agent1, agent2, num_games=5, save_results=False)
    
    print(f"评估结果:")
    print(f"  {agent1.name} 胜率: {results['summary']['agent1_win_rate']:.2%}")
    print(f"  {agent2.name} 胜率: {results['summary']['agent2_win_rate']:.2%}")
    print(f"  平局率: {results['summary']['draw_rate']:.2%}")

def human_vs_ai_example():
    """人机对战示例"""
    print("\n=== 人机对战示例 ===")
    
    # 创建环境和智能体
    env = GomokuEnv(board_size=9, win_length=5)
    human = HumanAgent(name="人类玩家", player_id=1)
    ai = RandomBot(name="随机AI", player_id=2)
    
    print("准备开始人机对战...")
    print("输入格式: 行,列 (例如: 0,0)")
    print("按 Ctrl+C 可以退出游戏")
    
    try:
        play_human_vs_ai(env, human, ai)
    except KeyboardInterrupt:
        print("\n游戏被用户中断")

def main():
    """主函数"""
    print("双人游戏AI框架 - 基础使用示例")
    print("=" * 50)
    
    # 运行各种示例
    basic_game_example()
    basic_env_example()
    agent_vs_agent_example()
    
    # 询问是否进行人机对战
    choice = input("\n是否进行人机对战? (y/n): ").lower().strip()
    if choice in ['y', 'yes', '是']:
        human_vs_ai_example()
    
    print("\n示例运行完成!")

if __name__ == "__main__":
    main() 