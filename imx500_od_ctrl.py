#!/usr/bin/env python3
"""
IMX500 Object Detection Demo - コントローラー
別ターミナルで起動し、threshold をリアルタイムに変更する。

キー操作:
  +/=  → threshold +0.05
  -/_  → threshold -0.05
  1〜9 → threshold を 0.1〜0.9 に直接セット
  q    → 終了
"""

import sys
import tty
import termios
import os

PIPE_PATH = "/tmp/imx500_ctrl"

def send(cmd: str):
    with open(PIPE_PATH, 'w') as pipe:
        pipe.write(cmd + "\n")

def main():
    if not os.path.exists(PIPE_PATH):
        print(f"Error: {PIPE_PATH} が見つかりません。")
        print("先に imx500_od_demo.py を起動してください。")
        sys.exit(1)

    print("IMX500 Controller")
    print("  +/=  : threshold +0.05")
    print("  -/_  : threshold -0.05")
    print("  1〜9 : threshold を 0.1〜0.9 に直接セット")
    print("  q    : 終了")
    print()

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('+', '='):
                send("inc")
                print(f"\r  → inc (+0.05)      ", end='', flush=True)
            elif ch in ('-', '_'):
                send("dec")
                print(f"\r  → dec (-0.05)      ", end='', flush=True)
            elif ch.isdigit() and ch != '0':
                val = int(ch) * 0.1
                send(f"set {val:.2f}")
                print(f"\r  → set {val:.2f}          ", end='', flush=True)
            elif ch in ('q', 'Q', '\x03'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\n終了")

if __name__ == "__main__":
    main()
