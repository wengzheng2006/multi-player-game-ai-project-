#!/usr/bin/env python3
"""
游戏启动脚本
让用户选择不同的游戏模式
"""

import sys
import os
import subprocess


def main():
    print("=" * 50)
    print("🎮 多游戏AI对战平台")
    print("=" * 50)
    print()
    print("请选择游戏模式:")
    print("1. 多游戏GUI - 五子棋和贪吃蛇 (推荐)")
    print("2. 贪吃蛇专用GUI - 更好的贪吃蛇体验")
    print("3. 五子棋命令行版本")
    print("4. 贪吃蛇命令行版本")
    print("5. 运行测试")
    print("6. 退出")
    print()

    while True:
        try:
            choice = input("请输入选择 (1-6): ").strip()

            if choice == "1":
                print("\n🎯 启动多游戏图形界面...")
                print("支持:")
                print("- 五子棋: 鼠标点击落子")
                print("- 贪吃蛇: 方向键/WASD控制")
                print("- 多种AI难度选择")
                print("- 暂停/继续功能")
                print()

                # 检查GUI文件是否存在
                if os.path.exists("gui_game.py"):
                    subprocess.run([sys.executable, "gui_game.py"])
                else:
                    print("❌ GUI文件未找到，请检查项目文件")
                break

            elif choice == "2":
                print("\n🐍 启动贪吃蛇专用图形界面...")
                print("特性:")
                print("- 专为贪吃蛇优化的界面")
                print("- 更流畅的游戏体验")
                print("- 多种贪吃蛇AI算法")
                print("- 实时状态显示")
                print()

                if os.path.exists("snake_gui.py"):
                    subprocess.run([sys.executable, "snake_gui.py"])
                else:
                    print("❌ 贪吃蛇GUI文件未找到")
                break

            elif choice == "3":
                print("\n♟️  启动五子棋命令行版本...")
                subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--game",
                        "gomoku",
                        "--player1",
                        "human",
                        "--player2",
                        "random",
                    ]
                )
                break

            elif choice == "4":
                print("\n🐍 启动贪吃蛇命令行版本...")
                subprocess.run(
                    [
                        sys.executable,
                        "main.py",
                        "--game",
                        "snake",
                        "--player1",
                        "human",
                        "--player2",
                        "snake_ai",
                    ]
                )
                break

            elif choice == "5":
                print("\n🧪 运行项目测试...")
                subprocess.run([sys.executable, "test_project.py"])
                break

            elif choice == "6":
                print("\n👋 再见！")
                sys.exit(0)

            else:
                print("❌ 无效选择，请输入 1-6")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 再见！")
            sys.exit(0)


if __name__ == "__main__":
    main()
