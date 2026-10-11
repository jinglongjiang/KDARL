# 当前代码token字段对照

当前KDA-VL代码不等于第一篇Mamba-VL投稿版本。本表只陈述当前代码事实，不作为第一篇勘误依据。论文符号未提供，全部留空。

| 行 | 列 | 代码字段 | 论文符号 | 恒零 |
|---|---|---|---|---|
| 0 | 0 | px |  | False |
| 0 | 1 | py |  | False |
| 0 | 2 | vx |  | False |
| 0 | 3 | vy |  | False |
| 0 | 4 | radius |  | False |
| 0 | 5 | gx |  | False |
| 0 | 6 | gy |  | False |
| 0 | 7 | v_pref (raw) |  | False |
| 0 | 8 | theta |  | False |
| 0 | 9 | sqrt((gx-px)^2+(gy-py)^2) |  | False |
| 0 | 10 | sqrt(vx^2+vy^2) |  | False |
| 0 | 11 | clip(v_pref,0,3) |  | False |
| 0 | 12 | 1 (robot flag) |  | False |
| 1 | 0 | gx-px |  | False |
| 1 | 1 | gy-py |  | False |
| 1 | 2 | sqrt(dx^2+dy^2) |  | False |
| 1 | 3 | atan2(dy,dx) |  | False |
| 1 | 4 | cos(goal angle) |  | False |
| 1 | 5 | sin(goal angle) |  | False |
| 1 | 6 | 0 |  | True |
| 1 | 7 | 0 |  | True |
| 1 | 8 | 0 |  | True |
| 1 | 9 | 0 |  | True |
| 1 | 10 | 0 |  | True |
| 1 | 11 | 0 |  | True |
| 1 | 12 | 0 |  | True |
| 2 | 0 | speed if speed>1e-6 else 0 |  | False |
| 2 | 1 | atan2(vy,vx) if speed>1e-6 else 0 |  | False |
| 2 | 2 | cos(motion angle) if speed>1e-6 else 0 |  | False |
| 2 | 3 | sin(motion angle) if speed>1e-6 else 0 |  | False |
| 2 | 4 | 0 |  | True |
| 2 | 5 | 0 |  | True |
| 2 | 6 | 0 |  | True |
| 2 | 7 | 0 |  | True |
| 2 | 8 | 0 |  | True |
| 2 | 9 | 0 |  | True |
| 2 | 10 | 0 |  | True |
| 2 | 11 | 0 |  | True |
| 2 | 12 | 0 |  | True |
| 3 | 0 | human.px-robot.px |  | False |
| 3 | 1 | human.py-robot.py |  | False |
| 3 | 2 | sqrt(rel_x^2+rel_y^2+1e-6) |  | False |
| 3 | 3 | human.vx (absolute) |  | False |
| 3 | 4 | human.vy (absolute) |  | False |
| 3 | 5 | sqrt(human.vx^2+human.vy^2) |  | False |
| 3 | 6 | human.radius |  | False |
| 3 | 7 | inverse TTC: dist/closing; clamp TTC>=0.1; receding=0 |  | False |
| 3 | 8 | 1 if dist<2 else 0 |  | False |
| 3 | 9 | 0 |  | True |
| 3 | 10 | 0 |  | True |
| 3 | 11 | 0 |  | True |
| 3 | 12 | 0 |  | True |
| 4 | 0 | human.px-robot.px |  | False |
| 4 | 1 | human.py-robot.py |  | False |
| 4 | 2 | sqrt(rel_x^2+rel_y^2+1e-6) |  | False |
| 4 | 3 | human.vx (absolute) |  | False |
| 4 | 4 | human.vy (absolute) |  | False |
| 4 | 5 | sqrt(human.vx^2+human.vy^2) |  | False |
| 4 | 6 | human.radius |  | False |
| 4 | 7 | inverse TTC: dist/closing; clamp TTC>=0.1; receding=0 |  | False |
| 4 | 8 | 1 if dist<2 else 0 |  | False |
| 4 | 9 | 0 |  | True |
| 4 | 10 | 0 |  | True |
| 4 | 11 | 0 |  | True |
| 4 | 12 | 0 |  | True |
| 5 | 0 | human.px-robot.px |  | False |
| 5 | 1 | human.py-robot.py |  | False |
| 5 | 2 | sqrt(rel_x^2+rel_y^2+1e-6) |  | False |
| 5 | 3 | human.vx (absolute) |  | False |
| 5 | 4 | human.vy (absolute) |  | False |
| 5 | 5 | sqrt(human.vx^2+human.vy^2) |  | False |
| 5 | 6 | human.radius |  | False |
| 5 | 7 | inverse TTC: dist/closing; clamp TTC>=0.1; receding=0 |  | False |
| 5 | 8 | 1 if dist<2 else 0 |  | False |
| 5 | 9 | 0 |  | True |
| 5 | 10 | 0 |  | True |
| 5 | 11 | 0 |  | True |
| 5 | 12 | 0 |  | True |
| 6 | 0 | human.px-robot.px |  | False |
| 6 | 1 | human.py-robot.py |  | False |
| 6 | 2 | sqrt(rel_x^2+rel_y^2+1e-6) |  | False |
| 6 | 3 | human.vx (absolute) |  | False |
| 6 | 4 | human.vy (absolute) |  | False |
| 6 | 5 | sqrt(human.vx^2+human.vy^2) |  | False |
| 6 | 6 | human.radius |  | False |
| 6 | 7 | inverse TTC: dist/closing; clamp TTC>=0.1; receding=0 |  | False |
| 6 | 8 | 1 if dist<2 else 0 |  | False |
| 6 | 9 | 0 |  | True |
| 6 | 10 | 0 |  | True |
| 6 | 11 | 0 |  | True |
| 6 | 12 | 0 |  | True |
| 7 | 0 | human.px-robot.px |  | False |
| 7 | 1 | human.py-robot.py |  | False |
| 7 | 2 | sqrt(rel_x^2+rel_y^2+1e-6) |  | False |
| 7 | 3 | human.vx (absolute) |  | False |
| 7 | 4 | human.vy (absolute) |  | False |
| 7 | 5 | sqrt(human.vx^2+human.vy^2) |  | False |
| 7 | 6 | human.radius |  | False |
| 7 | 7 | inverse TTC: dist/closing; clamp TTC>=0.1; receding=0 |  | False |
| 7 | 8 | 1 if dist<2 else 0 |  | False |
| 7 | 9 | 0 |  | True |
| 7 | 10 | 0 |  | True |
| 7 | 11 | 0 |  | True |
| 7 | 12 | 0 |  | True |
