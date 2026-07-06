"""薄入口：把 src/ 加入路径后委托给 livetrans.server.run()。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from livetrans.server import run

if __name__ == "__main__":
    run()
