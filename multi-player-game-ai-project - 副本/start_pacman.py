#!/usr/bin/env python3
"""
吃豆人游戏启动脚本
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pacman_gui import PacmanGUI

def main():
    """主函数"""
    print("🎮 吃豆人游戏启动中...")
    print("=" * 50)
    print("控制说明:")
    print("• ESC: 退出游戏")
    print("=" * 50)
    
    try:
        # 创建并运行GUI
        gui = PacmanGUI()
        print("✓ 游戏界面创建成功")
        print("🎮 开始游戏...")
        gui.run()
    except Exception as e:
        print(f"❌ 游戏启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  