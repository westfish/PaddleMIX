#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage:
    python avg_time_taken.py logfile1.txt logfile2.txt ...
    # 若不提供文件名，则可用管道：
    cat logfile.txt | python avg_time_taken.py
"""

import re
import fileinput

pattern = re.compile(r'Time taken:\s*([0-9.]+)\s*seconds')
times = []

for line in fileinput.input():          # 可同时遍历多个文件或 stdin
    m = pattern.search(line)
    if m:
        times.append(float(m.group(1)))

if times:
    avg = sum(times) / len(times)
    print(f'共找到 {len(times)} 条记录；平均 Time taken: {avg:.6f} 秒')
else:
    print('未找到任何 “Time taken:” 记录。')