#!/usr/bin/env python3
"""
IMX500 Object Detection Demo - Controller
Run in a separate terminal to change threshold in real-time.

Keys:
  +/=  -> threshold +0.01
  -/_  -> threshold -0.01
  1~9  -> set threshold directly to 0.1~0.9
  q    -> quit
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
        print(f"Error: {PIPE_PATH} not found.")
        print("Please start imx500_od_demo.py first.")
        sys.exit(1)

    print("IMX500 Controller")
    print("  +/=  : threshold +0.01")
    print("  -/_  : threshold -0.01")
    print("  1~9  : set threshold to 0.1~0.9")
    print("  q    : quit")
    print()

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ('+', '='):
                send("inc")
                print(f"\r  -> inc (+0.01)      ", end='', flush=True)
            elif ch in ('-', '_'):
                send("dec")
                print(f"\r  -> dec (-0.01)      ", end='', flush=True)
            elif ch.isdigit() and ch != '0':
                val = int(ch) * 0.1
                send(f"set {val:.2f}")
                print(f"\r  -> set {val:.2f}         ", end='', flush=True)
            elif ch in ('q', 'Q', '\x03'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\nQuit.")

if __name__ == "__main__":
    main()
