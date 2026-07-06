"""转写记录自动落盘（仅依赖标准库）。

每条 (原文, 译文, 时间戳) 追加到按日期命名的 CSV 与 JSON 文件，供事后查阅/导出。
"""

from __future__ import annotations

import csv
import json
import os
import time

__all__ = ["auto_save_record"]


def auto_save_record(save_dir: str, orig: str, tran: str, ts: float) -> None:
    """追加保存一条记录到 {save_dir}/{date}.csv 与 .json。出错静默（不影响主流程）。"""
    try:
        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

        csv_path = os.path.join(save_dir, f"{date_str}.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["时间", "原文", "译文"])
            writer.writerow([time_str, orig, tran])

        json_path = os.path.join(save_dir, f"{date_str}.json")
        records = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, ValueError):
                records = []
        records.append({"time": time_str, "ts": ts, "orig": orig, "tran": tran})
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
