"""pytest 配置：把 src/ 加入导入路径，使 `import livetrans.xxx` 可用。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
