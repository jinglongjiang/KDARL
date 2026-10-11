# B第一轮人数、密度与任务难度因果隔离

状态：第一轮授权执行范围完成。原正式960例、原40续跑shard、N10全部36smoke/192正式/8续跑shard及追加Mamba-v3的36smoke/192正式均完成独立CPU raw验收PASS。B5科学归因仍INCOMPLETE（严格跨N机会为0）。A1有限root机会门PASS，excess_transfer必要baseline门FAIL，主研究决策KILL额外Value迁移候选；两道门不混淆。零完整IL/RL，本轮没有方法级进展。

## 【研究问题】

解释固定5人训练checkpoint在20人条件下的损失，分别识别全局人数、局部拥挤、任务难度、可观测信息与训练覆盖。不同场景SR差不是“泛化损失”的定义；circle/square、两个训练seed分开统计。

原始采样合同于结果采集前冻结：每几何、每E1–E5固定case90000–90015，smoke取其中最前3例。640个模型正式episode＋320个参考episode；smoke180例不并入分母。92000块未读未用。原工程、a和engine不修改，所有执行在独立B目录。

| 条件 | 干预 | 保持不变 | 可解释范围 |
|---|---|---|---|
| E1 | 原始5人 | R4/width10；机器人8m | 原任务基线 |
| E2 | 原始20人 | 同case原场地、机器人8m | 人数与交互联合变化，不是纯N效应 |
| E3 | 原5人＋15远人 | 原5人初值与真实ORCA；同命令逐步核验 | 通过所有闸门时才能隔离全局N |
| E4 | 20人中新增15局部重布 | 原5人及20人总数、场地、机器人8m | 特定固定局部拥挤干预，不是密度曲线 |
| E5 | E2行人位置/目标×2 | 行人速度/半径、机器人起终点均不变 | 名义面积密度相同，不等于局部遇见密度相同 |

E3远人x=10000+20i、y=0，目标同x、y=4。无强制轨迹、无移除行人ORCA相互作用。极远布局是人工负对照，不属于原始测试分布。E4固定0.85m格点顺序，新增起点与已有主体保持半径和＋0.2m净间隔，不按终局调参；原5人的继承重叠不修复。E5人类circle名义R4→R8、square width10→20，绝不把机器人起终点扩成16m。

## 【已经确认的直接机制】

**SUPPORTED，仅限所执行CPU同命令轨迹：E3全局人数由5到20，没有改变局部动力学及当前决策接口。**

原始证据：b/work/cpu-smoke.json与b/work/cpu-main.json。smoke30场景、6条配对；正式160场景、32条配对，每条201步，共6432步。原5人位置、速度、目标、半径、偏好速度、角度、实际ORCA参数及实际RVO2邻接集合全程一致；TTC5、当前token、24帧可见history、80个平滑command、即时r与clearance逐位相同，最大误差0。新增远人全过程互不相撞，最小远人相互净距19.4m，x始终>9000，原5人邻接没有远人。

环境人类ORCA有全体TTC制动，故这里核验真实计算而非只看10m邻域或初始嵌套。CPU资格检查在终止后继续物理更新至50.25s，只用于分离性，不把终止后轨迹计为导航成绩。正式E1每个模型/参考闭环还会在E3 shadow逐步验相同命令下的人群与即时量；独立E3闭环另执行，不用shadow替代完整episode。

**正式闭环补充证据：SUPPORTED，限于全部已归档字段。** 独立verify-b-raw.py不import驱动/模型、不运行rollout，以NumPy/标准库读取实际gzip raw及done/start。60个SHA全部通过，16case顺序/唯一性、四权重/config manifest SHA、公共acceptance、80命令/即时r/clearance/评分、TTC5/token/history hash、转移、终止、原Return和路径均复核；共52667个raw步。E1/E3按12个控制器/seed/几何组逐case比较192对、8217对步，除附加远人和延迟外，所有归档字段精确相同；远人x>9000，最小相互净距19.4m。不是仅比较summary或终局。

存档限制必须保留：GPU raw每步记录更新前行人全状态，未另存最后一次更新后的完整行人状态，也未逐步存储ORCA参数/邻接ID数组；最终人群更新及邻接一致性由已冻结源码中的shadow执行断言核验，CPU资格文件保存了参数/邻接比较结果。不能把执行断言称为可从GPU raw独立复算的最终全状态/全邻接证据。80候选windows未逐项归档，当前token与真实history hash可独立重建，不声称重跑神经网络获得了同一windows/value。

这排除的是“这些已验轨迹中，只因后台N变大就改变TTC5/局部评分”的解释，不证明所有20人任务观测充分，不证明远人对任意策略都无影响。完整history相同且即时量相同只是负对照；它本身不建立观测混叠导致的实际损失。

公共修复验收由主Agent执行，原始acceptance.json可查：184f6f0修探索漏记真实history；9f06223只修已预测终止/已知超时的bootstrap。B工作树对应cherry-pick为3e30f19、1a44879，policy内容SHA与accepted原commit一致。冻结checkpoint不因history修复自动获得重训收益；终止修复后的评分另立基线，不与旧分支混算。

模型endpoint碰撞/goal0.25与环境旧速度swept判碰/goal0.3仍不同；终止修复未统一这两套物理/Reward合同。原env Reward不修改，actual ORCA扫掠碰撞另报。超时在旧env第201步出现，因此time可为50.25s；不可悄悄裁到200步。

## 【已执行的因果干预】

| 项目 | 实际执行 | 状态 |
|---|---|---|
| E1–E5固定场景生成 | circle/square×16case×5条件 | COMPLETE |
| E3全过程隔离 | 32配对×201步；无轨迹强制 | PASS |
| 入口/接口单元验收 | 5项；50个原score即时r/clearance/token/command/filter逐位parity；ORCA和CV-MPC重放及一次覆盖；E4/E5；公共源码验收；禁92000 | PASS |
| 四checkpoint完整闭环 | 640正式episode，逐raw核验 | PASS |
| ORCA-grid与CV-MPC参考 | 320正式episode，逐raw核验 | PASS执行；CV-MPC本身不充当成熟第二参考，另见追加Mamba |
| 全控制器E1/E3独立闭环等价 | 192配对、8217对步，所有归档字段精确相同 | PASS，保留最终全状态存档限制 |
| 冻结A0/A1及成功对照 | 原40shard、61root（36失败＋25成功）、122个A0/A1 episode，全raw重放/覆盖/返回值复核 | 执行PASS；未匹配11失败root不得称受保护 |
| N10独立扩展 | CPU32组native/160组旧转发；36smoke、192正式（128模型＋64参考）、8branch/7root/14个A0/A1 episode | 执行PASS；唯一救援无成功匹配 |
| B5三人数相似机会 | 独立重建全部8组5/10/20匹配，与存档逐项一致 | INCOMPLETE；三类strict pair与triple均0 |
| 已有Mamba-VL匹配接口参考 | v3独立预注册、36smoke＋192正式、全部raw独立复核；E1/E3 32正式配对1536步 | PASS；已有本地控制器层级，非完整论文复现 |
| B4受控训练 | 未授权，零训练 | NOT EXECUTED |

实现入口（调度命令已合并到下方复现节，不另交付helper报告）：

/home/abc/workspace/ab-causal-isolation-20261011/b/run-b.py

原始合同：

/home/abc/workspace/ab-causal-isolation-20261011/b/b-contract.json

主Agent独占GPU；remote project为b/runtime-project，不复制本地worktree的.git绝对路径。run-b.py的jobs只输出argv，绝不启动GPU或训练。smoke全部完成并通过smoke-gate后才允许正式16case；预算停机保留INCOMPLETE，不自动缩样或换case。未完成shard无法进汇总。

独立基线验收原始小文件已回本地，SHA256=a2dc0250cebfc5edc80a740c61105e6113c637b640b3036092f3d6e86b07d8da：

/home/abc/workspace/ab-causal-isolation-20261011/b/work/verified-baseline.json

完整60组含时间、路程、actual swept、推理代价及配对bootstrap保存在同一结构化验收文件；表格内容已并入本报告，不另交付重复helper报告：

/home/abc/workspace/ab-causal-isolation-20261011/b/work/verified-baseline.json

基线核验器SHA256=1b96504b682452f78e673633a5036500bccd25edc3d94bec8a7b6788a7972eec；原driver仍为8b4ee9d18c5014e43d30fb7f8944625dae4104b23c0713c4bdddaee19099dece，原contract仍为c89f0ca680e17d77470ee572f8608567aa00e4240cb20be1c08e47b979ed6ac2。验收在远端CUDA_VISIBLE_DEVICES为空、BLAS单线程下完成，未占GPU，未修改raw。

原续跑与N10独立原始数据核验：verify-b-continuations.py SHA256=253a0fef7f277c526b2c2ba179f88df16790e8c72749f780f4e6d6ff69826de5；其NumPy raw helper现为e98384fe0ec4e66f26964542d688a302642917ac4bbfa75ad081b843cfc905aa（仅增加原生E10人数校验，不改驱动）。未加载模型、未rollout、未改阈值，6项核验器单元测试通过。原960例52667步，N10正式192例9604步；48个branch shard、68root/136个A0/A1 episode。全部A0物理/history/score/terminal/Return精确重放，A1仅规定时刻覆盖一次真实平滑命令及last_action，后续恢复原argmax策略。

完整验收文件远端SHA256=5995024f62ad85231f3f632c46b3b1b9a39651f33e31e42a415826780feb4b0d：

/root/ab-causal-isolation-20261011/b/work/verified-complete.json

本地已回传小文件仅省略每root的all80重复数组，保留实际两动作/score/G/outcome/成功保护与完整验收SHA；原61root完整all80还在本地verified-branches.json。小文件可直接读：

/home/abc/workspace/ab-causal-isolation-20261011/b/work/verified-complete-brief.json

/home/abc/workspace/ab-causal-isolation-20261011/b/work/verified-gates.json

追加Mamba验收小文件SHA256=d3b4541d2631be8ab66b10ec5010a0cd937228fabaa42b62c94cbf7060304153，已拉本地：

/home/abc/workspace/ab-causal-isolation-20261011/b/work/verified-mamba-reference.json

### 复现命令与边界

以下合并原RUN-COMMANDS.md及RUN-B10-COMMANDS.md。仅主Agent调度；不是要求现在重跑。已有产物拒绝覆盖，GPU仍由主Agent独占。remote必须用已accepted的runtime-project；local仍为b/project，不能复制其含绝对路径的.git文本。

```bash
PY=/root/miniconda3/envs/mamba/bin/python
B=/root/ab-causal-isolation-20261011/b
C=/root/ttc5-causal-20261011
PROJECT=$B/runtime-project
ACCEPT=/root/ab-causal-isolation-20261011/acceptance.json
export PYTHONPATH=/root/kda-vl-deps/flash-linear-attention
CUDA_VISIBLE_DEVICES= "$PY" "$B/run-b.py" cpu --phase smoke --project "$PROJECT" --config "$C/inputs/kda-s419-t24/config.ini"
CUDA_VISIBLE_DEVICES= "$PY" "$B/run-b.py" cpu --phase main --project "$PROJECT" --config "$C/inputs/kda-s419-t24/config.ini"
"$PY" "$B/run-b.py" jobs --phase smoke --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT"
```

jobs只打印JSON argv，绝不启动任务。主Agent依次执行60个smoke argv，每个固定3case，180条(120模型＋60参考)。参考只各执行一次，不按四checkpoint重复。单任务例（已在该batch内，勿重复）：

```bash
"$PY" "$B/run-b.py" baseline --phase smoke --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT" --model kda-s419-t24 --controller model --geometry circle --condition E1
"$PY" "$B/run-b.py" smoke-gate --project "$PROJECT"
"$PY" "$B/run-b.py" jobs --phase main --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT"
```

主Agent在gate PASS后依次执行60个main argv，每个16case，共960条(640模型＋320参考)。case90000–90015固定顺序，circle/square分别执行，smoke重复前3不加正式分母，92000禁读。每个模型全部10组基线完成后冻结root，再执行其续跑；无成功对照不得称受保护。以下循环也是原调度合同，不在B自行启动：

```bash
for M in kda-s419-t24 kda-s443-t24 gru-s419-t24 gru-s443-t24; do
  CUDA_VISIBLE_DEVICES= "$PY" "$B/run-b.py" roots --project "$PROJECT" --model "$M"
done
for M in kda-s419-t24 kda-s443-t24 gru-s419-t24 gru-s443-t24; do
  for G in circle square; do
    for E in E1 E2 E3 E4 E5; do
      "$PY" "$B/run-b.py" counterfactual --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT" --model "$M" --geometry "$G" --condition "$E"
    done
  done
done
CUDA_VISIBLE_DEVICES= "$PY" "$B/run-b.py" summarize --project "$PROJECT"
```

所有空root shard也必须产出done。A0全物理/history/评分/终止/Return精确重放，延迟除外；A1不读后验结果。原cap上限160root、320个A0/A1 episode，不把root当独立采样episode。原正式最大128640模拟步；冻结前50/100ms每步的107/214min仅为上界占位算术，实际基线成本已另报。CPU smoke3.79s、formal20.66s；不得用FLOPs替代实测速。入口在原8h绝对截止前10min停止新工作，保留INCOMPLETE，不自动缩/扩样。

N10使用独立b10-protocol.json、b10.py，输出b/n10/work，不改原driver/contract、不重启8h计时。两几何×6控制器×16case=192正式(128模型＋64参考)；12smoke shard×3=36。

```bash
CUDA_VISIBLE_DEVICES= "$PY" -B "$B/b10.py" cpu --project "$PROJECT" --causal "$C" --config "$C/inputs/kda-s419-t24/config.ini"
"$PY" -B "$B/b10.py" jobs --phase smoke --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT"
```

CPU构造32个native N10与原reference.environment reset配对，不load权重、不执行episode。主Agent串行执行12个N10 smoke argv后：

```bash
CUDA_VISIBLE_DEVICES= "$PY" -B "$B/b10.py" smoke-gate --project "$PROJECT" --causal "$C"
"$PY" -B "$B/b10.py" jobs --phase main --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT"
```

gate PASS后串行执行12个正式argv。每模型的两几何正式完成后，先冻结root，再运行两几何续跑；同cap、同表型/caliper、同A1，不另选样：

```bash
for M in kda-s419-t24 kda-s443-t24 gru-s419-t24 gru-s443-t24; do
  CUDA_VISIBLE_DEVICES= "$PY" -B "$B/b10.py" roots --project "$PROJECT" --causal "$C" --model "$M"
  for G in circle square; do
    "$PY" -B "$B/b10.py" counterfactual --project "$PROJECT" --causal "$C" --acceptance "$ACCEPT" --model "$M" --geometry "$G"
  done
done
CUDA_VISIBLE_DEVICES= "$PY" -B "$B/match-b10.py" --work "$B/work" --n10-work "$B/n10/work"
```

matcher只读冻结N5/E1、N10/E10、N20/E2 root，输出严格pair/triple及分母，既不load模型也不以无匹配否定泛化损失。旧10S数据不能顶替N10。量化ORCA/自制CV-MPC不是最优界或官方完整baseline；B4仍零训练。

独立CPU验收（只读NumPy/标准库、输出JSON、不import模型/rollout），原remaining/done PASS后执行；N10全部8branch done与cross-N-matches存在后可执行all，不必等其他supplemental任务。all复用SHA固定的原验收避免重复模型/轨迹计算：

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -B "$B/verify-b-continuations.py" --scope original --work "$B/work" --manifest "$C/manifest.json" > "$B/work/verified-branches.json"
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -B "$B/verify-b-continuations.py" --scope all --work "$B/work" --n10-work "$B/n10/work" --manifest "$C/manifest.json" --original-verified "$B/work/verified-branches.json" > "$B/work/verified-complete.json"
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -B "$B/verify-b-gates.py" "$B/work/verified-complete.json" > "$B/work/verified-gates.json"
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -B "$B/verify-mamba-reference.py" --reference /root/ab-causal-isolation-20261011/reference-mamba-v3 > "$B/work/verified-mamba-reference.json"
```

Mamba原始复现入口为BASE/reference-mamba.py，--project "$PROJECT" --causal "$C" --acceptance "$ACCEPT"；只由主Agent执行，已有v3 protocol时拒绝覆盖，不能对已有目录再次启动。B只运行上述CPU核验。核验器源及分门判据固定于b commit c849a6a，7项新增自测PASS；核验Mamba时不import wrapper/模型/环境，不调用FLA或torch。

## 【各因素的实际导航效果】

**以下来自960条正式raw的独立复算，非smoke/旧数据。** 每格n=16，单元为成功/碰撞/超时计数；分号后为原gamma0.99折扣Return均值。E3每格与E1完全相同，省略重复数值，不省略其192对轨迹验收。

| 几何 | 控制器/seed | E1 | E2 | E4 | E5 |
|---|---|---|---|---|---|
| circle | KDA419 | 16/0/0; .64580 | 14/0/2; .40170 | 15/1/0; .42946 | 7/9/0; .06656 |
| circle | KDA443 | 15/0/1; .59871 | 15/1/0; .45142 | 14/0/2; .35186 | 8/8/0; .09946 |
| circle | GRU419 | 15/1/0; .58893 | 14/1/1; .42861 | 10/0/6; .36743 | 10/6/0; .21449 |
| circle | GRU443 | 16/0/0; .64341 | 13/2/1; .37379 | 13/0/3; .45732 | 10/6/0; .23787 |
| circle | ORCA-grid | 14/2/0; .47090 | 13/3/0; .29878 | 13/1/2; .33389 | 7/9/0; .08121 |
| circle | CV-MPC | 5/11/0; -.11879 | 0/16/0; -.46371 | 13/3/0; .28227 | 10/6/0; .28250 |
| square | KDA419 | 16/0/0; .66294 | 13/1/2; .36596 | 15/0/1; .47282 | 15/1/0; .51500 |
| square | KDA443 | 16/0/0; .64687 | 12/2/2; .27877 | 15/0/1; .43396 | 15/1/0; .53246 |
| square | GRU419 | 16/0/0; .63127 | 14/2/0; .37487 | 15/0/1; .53983 | 14/2/0; .49434 |
| square | GRU443 | 16/0/0; .62036 | 13/1/2; .32649 | 15/0/1; .50156 | 14/2/0; .44240 |
| square | ORCA-grid | 16/0/0; .67117 | 12/3/1; .23768 | 16/0/0; .51888 | 15/1/0; .52564 |
| square | CV-MPC | 15/1/0; .63234 | 8/8/0; .06276 | 6/10/0; -.14950 | 11/5/0; .31706 |

E2平均时间(s)/路程(m)：circle KDA419=22.6406/17.8486、KDA443=17.4375/13.1645、GRU419=17.4844/13.4722、GRU443=17.8750/13.3526、ORCA=15.8750/9.9185、CV-MPC=4.7969/4.3362；square依次20.6719/14.5707、21.2656/12.9182、16.2969/12.7363、22.4063/15.4788、18.3906/10.4205、7.5156/6.9905。含失败episode，CV-MPC短时间不等于高导航效率。

模型每episode p50延迟的组均值约KDA15.2–15.6ms、GRU13.0–13.3ms；ORCA0.63–1.06ms、CV-MPC5.88–6.57ms。每组p50/p95完整列于baseline-tables.md，这不是所有step合并后的分位数。60个shard内墙钟合计944.25s；主调度总墙钟1263.99s(21m03.99s)，后者包含进程/调度等开销，不与前者混同。

legacy env CR与actual swept不统一的实际反例保留：circle E4 KDA419有1个env collision但actual物理碰撞episode为0；circle E5 KDA419有1个env success却actual swept<0。不得把这些记为公共terminal修复已统一物理安全。

Return沿用旧audit的gamma0.99折扣定义，undiscounted_return单列。同case配对E1→E2、E1→E3、E2→E4、E1→E5、E2→E5；case为统计单位，固定seed41995、2000次case bootstrap；训练seed分别报，不从两个seed估总体架构方差。安全候选数、crowd2m是实际经过轨迹的描述量，不直接充当根因。

ORCA参考用已有ORCA源代码，但连续输出量化到原80个alpha0.3平滑command，再施加同0.2安全filter、0.8风险惩罚和index0微小偏置；不能称为未约束ORCA性能。CV-MPC是CV行人、4步有限时域、beam4的已知规划构造，使用环境Robot kinematics；机器人目标进度是内部成本，不改env Reward。参考读取同一合法全人ObservableState，不能读人类隐藏目标或真实后继。原模型的安全评分本就读取全体可见行人，TTC5截断仅作用于Value输入。

两参考都不是数学最优或不可避免失败率上界；未执行完整RGL复现。CV-MPC仅为自制标准构造，不是成熟已知方法的官方代码/论文基线复现；尤其circle E1仅5/16、E2为0/16，**不能据其失败证明任务不可行，也不满足用户B3成熟第二参考要求**。其beam截断、短时域和无制动命令差异及负结果原样保留。

主Agent另行预注册追加已有Mamba-VL optimized checkpoint(f23e411effac2f1959eead081fa49c908a85532642172c50a34427fe40bff2b9)，零训练，同T24/TTC5/80/alpha.3/公共terminal基线。层级为“已有本地Mamba-VL checkpoint的匹配接口控制”，不是论文完整复现、不是fresh-history公平架构比较、不是新骨干研发。追加只补B3难度证据，不事后替换原CV-MPC或修改原candidate gate。初始wrapper缺case属性，首episode前失败；v2漏写驱动要求的work/smoke-accepted.json，35/36smoke时主动SIGTERM143、零formal；旧protocol/log/source snapshots保留。v3仅修接口，独立protocol及目录，36smoke全部重跑，前两版结果不并入正式分母。最终v3 summary PASS后独立核验全部24shard的gzip SHA、原权重/config/accepted project source、固定case/初值、TTC5/history/80command/即时量/评分/转移/终止/Return，全部PASS；正式13016步，加smoke共15713步。正式E1/E3 32对/1536步全归档字段精确一致，另有6smoke配对260步。

v3 protocol SHA256=6859f125ee3d0598ba972c32e952f256770f86b3da893005001dcc6f1050598a，入口SHA256=b793ad9b57b51f1760963a0bb11ae0dd01503b9a2850a4fd2cb6d905cf4589d0，config SHA256=cc2a64237e77404ee8d9808af9640b44c415527ddfdde6d5c82839ae8fd13a25；CPU核验器SHA256=58246bea30b0627fd69b0f065ea8f0a475deb3e7d76771d34fe391cb50c0f959。执行墙钟309.522s包含36smoke＋192formal及进程开销，不是仅formal或每episode推理成本。这里只完成已有已知价值方法的本地匹配接口参考，不能提高标成官方完整论文baseline复现。

每格n=16，Mamba-v3真实导航结果：

| 几何 | 条件 | 成功/碰撞/超时 | mean G | time(s)/path(m) | actual碰撞episode |
|---|---|---|---|---|---|
| circle | E1 | 16/0/0 | .644296 | 11.1875/9.2448 | 0 |
| circle | E2 | 13/1/2 | .375858 | 20.8438/14.8271 | 1 |
| circle | E3 | 16/0/0 | .644296 | 11.1875/9.2448 | 0 |
| circle | E4 | 13/0/3 | .394058 | 24.5469/17.3502 | 0 |
| circle | E5 | 5/11/0 | −.107278 | 10.2813/7.7764 | 10 |
| circle | E10 | 16/0/0 | .556336 | 14.8281/10.8973 | 0 |
| square | E1 | 16/0/0 | .602351 | 12.8125/10.2324 | 0 |
| square | E2 | 8/2/6 | .120209 | 31.2656/17.9321 | 2 |
| square | E3 | 16/0/0 | .602351 | 12.8125/10.2324 | 0 |
| square | E4 | 15/0/1 | .445694 | 20.9688/15.2415 | 0 |
| square | E5 | 14/1/1 | .472059 | 14.7656/11.3794 | 1 |
| square | E10 | 13/1/2 | .421738 | 17.8750/12.4080 | 0 |

Mamba组均值推理episode p50为12.34–12.71ms、p95为12.82–13.30ms，完整12组在验收JSON。circle E5的11个env collision仅10个actual碰撞、square E10的1个env collision但actual为0，再次证实公共修复scope不包含统一两套物理终止。

Mamba E1→E2 ΔSR/ΔG：circle −3/16/−.268438，ΔG case-bootstrap95=[−.411435,−.147802]；square −8/16/−.482141，CI=[−.644552,−.327375]。与ORCA对应−1/16/−.172123、−4/16/−.433494共同支持这两类已知控制器在原20人任务受损（circle ORCA靠Return门而非SR门），不证明失败不可避免。E2→E4 Mamba为circle 0/+.018201、square +7/16/+.325485，与原结论一致，不支持“固定局部重布必然加难”。Mamba E1→E10 ΔG为circle −.087960、square −.180612；E10→E2为−.180478/−.301529，仍不是同机会因果放大。

仅作追加参考的描述性相对差距：KDA419/443的E1→E2 Return loss减Mamba loss，circle为−.024343/−.121140，square为−.185162/−.114041；SR excess分别−1/16/−3/16及−5/16/−4/16。没有相对该追加参考的更大下降，但不据此做架构排名，也不将其重新塞入原门。本轮B3已有ORCA＋已知价值方法的合法匹配控制器实证；原CV-MPC不够成熟的负结果不撤回，完整论文复现并未获得。

**冻结参考难度对比**，以下均为B−A；CI为2000次case配对bootstrap、seed41995，不作架构seed总体CI：

| 几何/参考 | E1→E2 ΔSR | E1→E2 ΔReturn [95%CI] | E2→E4 ΔSR | E2→E4 ΔReturn |
|---|---|---|---|---|
| circle ORCA | -.0625 | -.172123 [-.382718,.022388] | 0 | +.035110 |
| circle CV-MPC | -.3125 | -.344917 [-.586648,-.145348] | +.8125 | +.745982 |
| square ORCA | -.2500 | -.433494 [-.626531,-.260677] | +.2500 | +.281197 |
| square CV-MPC | -.4375 | -.569578 [-.840402,-.310233] | -.1250 | -.212264 |

E1→E2在两几何满足原“两构造参考”数值难度门，追加已有Mamba也与ORCA同向下降，获得本地已知价值参考的独立补充，不冒称完整论文复现。它支持这些控制器在原20人任务受损，不证明“难度单独足够”。E2→E4两几何均不满足两原参考共同下降门，按原合同KILL本轮“该固定局部重布必然加难”的强解释；实际布局更拥挤不蕴含任务更难。

**额外损失的必要baseline门**：定义excess=(KDA E1−E2损失)−(参考E1−E2损失)，正值才表示KDA额外下降。Return excess(circle/square)：KDA419相对ORCA为+.071972/−.136514，相对CV-MPC为−.100822/−.272598；KDA443相对ORCA为−.024825/−.065393，相对CV-MPC为−.197619/−.201478。四个seed×几何格均未同时超过两参考的原SR≥2/16或Return>.05门。故原额外Value迁移候选的必要门未满足，保留该否定结果；A1有限root门通过不填补这一必要门，追加Mamba也不追溯替换此闸门。科学上仍不能反证所有Value失效，更不能据此指认覆盖/表征根因。

| 预设判断 | GO条件 | 未满足时 |
|---|---|---|
| 任务变难 | 两参考同几何E2−E1或E4−E2均SR下降≥2/16或平均Return下降>0.05 | KILL本轮“难度单独足够”强解释，不声称理论反证 |
| 额外迁移损失候选 | KDA相对两参考的高人数额外下降≥2/16或>0.05Return，两seed重复；A1净挽救≥2且成功破坏0 | KILL该候选，不把原SR差指定为价值泛化失败 |
| A1受保护改进 | 两KDAseed合计至少2失败→成功、配对Return均值>0、匹配成功破坏0、所有验收有效 | KILL受保护候选；不补选更好动作 |

续跑只用冻结预设A1：安全且progress≥0.02中的最高raw-score，最小index破并列。每模型所有基线完成后按case序冻结root，续跑前A0必须完整状态/history/score/terminal/Return精确重放；改变一次平滑后命令并同步last_action，随后恢复同策略。失败root、无root、超出cap、缺成功匹配均保留分母。成功对照用相同时间/目标距离/安全候选/crowd2m caliper和机会表型，无匹配不得称“受保护”。

### 实际续跑与A1独立裁决

以下原40个branch shard，均为原合同cap内；无root、cap外不补采。ΔG是失败root的A1−A0均值，成功对照另报，不把root当独立整episode或架构总体统计：

| 模型 | 全失败 | 无合格root | 合格但cap外 | 失败root | 成功对照 | 未匹配失败root | ΔG失败均值 | 救援/匹配保护救援 | 成功破坏 |
|---|---|---|---|---|---|---|---|---|---|
| KDA419 | 17 | 7 | 2 | 8 | 5 | 3 | +.342922 | 4/3 | 0 |
| KDA443 | 19 | 3 | 6 | 10 | 9 | 1 | +.173045 | 3/3 | 0 |
| GRU419 | 21 | 5 | 7 | 9 | 5 | 4 | +.353282 | 2/1 | 0 |
| GRU443 | 18 | 4 | 5 | 9 | 6 | 3 | +.221786 | 3/3 | 0 |
| 总计（含E1/E3重复） | 75 | 19 | 20 | 36 | 25 | 11 | 不跨seed给总体CI | 12/10 | 0 |

25个成功对照全部维持env success，所有救援及对照的A1归档actual swept均非负；这不等于效率/Return也不损失。成功对照平均ΔG按上述模型为+.052061、−.046085、+.051889、−.012477。KDA443 square E4的成功对照ΔG=−.308805，不能以零成功破坏掩盖收益退化。KDA443 circle E1/E3的失败case90005和对应成功对照是同一物理轨迹的重复，不是跨任务独立证据。

**A1门：PASS，仅有限且具完整成功匹配的root机会。** 原合同缺control按model×geometry×condition整格失格，不按救援效果挑root。失格为KDA419 circle E2（2失败root、1缺control）、KDA419 circle E5（2缺）、KDA443 circle E5（1缺）。其余有root的KDA格保留：419 circle E4、square E2/E4；443 circle E1/E2/E3/E4、square E2/E4。两seed均在保留集合中且各自有救援；总12失败root＋12成功control、5救援、平均ΔG=+.247694、成功破坏0。circle为6root/2救援/ΔG+.237197，square为6root/3救援/ΔG+.258192，分别满足原数值门。circle的两次救援均来自443，不声称其救援在两seed独立重复。去掉重复E3后11失败root/11control、仍5救援、ΔG+.270212，结论不靠重复达门。该通过是局部因果动作收益，不是表征或泛化根因，也不是可部署算法：root由事后失败轨迹定位，尚无合法在线触发合同、全分布保护及独立技术差分，METHOD_SPEC_FREEZE仍NO。

**excess_transfer门：FAIL，研究决策KILL该额外迁移候选。** 两seed×两几何均不满足相对两原参考的必要baseline门；A1 PASS不覆盖这一FAIL。verify-b-gates.py读取已验收raw统计分别复算这两门，输出verified-gates.json；不更新contract、不将追加参考替换原候选参考。

Return保护补充：上述A1合格条件的12失败root＋12成功对照共同计入，平均ΔrootG=+.107440、平均整episode ΔReturn=+.084495；仅失败root平均整episode ΔReturn=+.195560。不是隐藏成功对照负ΔG后才得到正均值。但这仍是冻结有限机会的配对描述，不是随机全分布平均收益或置信保证。

部分可定位raw例（score列为原raw Q；所列候选均安全，filtered Q相同；G从root时刻折扣，非整episode Return）：

| 模型/几何/条件/case/step | A0→A1 index | Q0→Q1 | G0→G1 | 实际结果及保护 |
|---|---|---|---|---|
| KDA419/circle/E2/90007/35 | 49→29 | .770139→.768996 | −.095231→.617290 | timeout→success，缺control，整格失格 |
| KDA419/circle/E2/90008/23 | 74→34 | .710971→.709723 | −.261841→.609169 | timeout→success，control90010/23保持；因同格缺control仍不入A1门 |
| KDA419/square/E2/90001/28 | 67→5 | .603650→.585368 | −.088818→.613577 | timeout→success，有保持成功的对照 |
| KDA443/circle/E2/90010/28 | 74→14 | .620928→.615643 | −.458020→.237543 | collision→success，有保持成功的对照 |
| KDA419/circle/E4/90012/20 | 3→9 | .686428→.685987 | −.445245→−.482199 | 仍collision，ΔG负，有保持成功的对照 |
| KDA443/circle/E1及E3/90005/23 | 9→14 | .712031→.710931 | −.084411→−.084411 | 均仍timeout，重复负结果保留 |
| KDA443/circle/E10/90002/20 | 69→34 | .282715→.251976 | −.081904→.475340 | timeout→success，无成功对照，不受保护 |

真实执行平滑命令不是网格index对应的未平滑速度。circle E2 KDA419/90008/23为[.350592,−.237922]→[−.639357,.752027]，匹配成功control90010/23为[.342778,−.454934]→[.151575,.376891]，control Q=.706536→.703187，G=.257722→.496399；N10 KDA443/90002/20为[.372349,−.925934]→[−.390504,.215756]。全部68root的实际两命令、raw/filtered score、即时r/clearance/progress、G0/G1、time/path/终局及control关联均在verified-complete-brief.json的root_details，完整all80和原raw路径由其SHA锚定；不以这些例代替全量验收。

### N10实际正式结果与匹配边界

N10理由仅补用户原5/10/20要求，在采集前冻结b10-protocol.json，不依据B-main效果。b10.py包装scene，native CrowdSim N10原场地、robot8m；原driver/contract完全未改，继承A1/cap2＋2/caliper/A0及绝对截止时间。CPU32组native初值/ORCA和160组旧条件转发精确通过；正式36smoke、192main及8branch已由主调度完成并独立验收。首次CPU相对脚本路径保存错误修复后重跑，无模型结果丢弃。旧10S从未用于新臂。

每格n=16，延迟为episode p50/p95的均值（ms），非全step合并分位数：

| 几何 | 控制器 | 成功/碰撞/超时 | mean G | time(s)/path(m) | actual碰撞episode | p50/p95(ms) |
|---|---|---|---|---|---|---|
| circle | KDA419 | 15/0/1 | .550382 | 15.6719/12.9101 | 0 | 15.336/16.181 |
| circle | KDA443 | 14/0/2 | .522663 | 17.2969/12.4046 | 0 | 15.102/15.772 |
| circle | GRU419 | 15/0/1 | .562095 | 15.1250/12.8307 | 0 | 13.138/13.822 |
| circle | GRU443 | 16/0/0 | .594444 | 13.2188/10.4088 | 0 | 13.152/13.668 |
| circle | ORCA-grid | 13/3/0 | .387619 | 12.5313/8.9023 | 3 | .846/.948 |
| circle | CV-MPC | 0/16/0 | −.442029 | 5.7188/5.1092 | 16 | 6.093/6.698 |
| square | KDA419 | 15/1/0 | .521094 | 13.0156/10.3162 | 1 | 15.300/15.965 |
| square | KDA443 | 15/1/0 | .535386 | 11.9844/9.8703 | 1 | 15.282/15.780 |
| square | GRU419 | 16/0/0 | .607714 | 12.5000/10.4359 | 0 | 13.153/13.682 |
| square | GRU443 | 15/1/0 | .512051 | 12.7031/10.2637 | 1 | 13.195/13.824 |
| square | ORCA-grid | 14/2/0 | .430908 | 12.3125/9.1687 | 2 | .817/.918 |
| square | CV-MPC | 11/5/0 | .304101 | 7.9844/7.5841 | 6 | 6.125/6.307 |

N10 square CV-MPC另有1个env success带actual swept碰撞，保留旧物理合同缺口。N10共7失败，2无合格root、0 cap外、5失败root＋2control，其中3失败root无control。KDA419两失败root零救援、平均ΔG+.003585，1control ΔG−.005311；KDA443两失败root有1救援、平均ΔG+.278622，但救援90002无control，另一root90007的control90005保持成功且ΔG+.016944；GRU419一个失败root ΔG0/无control，GRU443失败无合格root。N10没有可称受保护的救援，不能并入原A1门凑救援数。

以下数组顺序统一N5/E1、N10/E10、N20/E2；分母只含这三个原场地臂，不含E3/E4/E5。match-b10.py与独立实现逐组重建同表型/同caliper、优先同case、不放回匹配；triple必须三条边均合格：

| 模型/几何 | 全失败 | 合格失败 | cap外 | selected失败root | 成功control | strict pairs(5–10/10–20/5–20) | triples |
|---|---|---|---|---|---|---|---|
| KDA419 circle | 0/1/2 | 0/1/2 | 0/0/0 | 0/1/2 | 0/0/1 | 0/0/0 | 0 |
| KDA419 square | 0/1/3 | 0/1/3 | 0/0/1 | 0/1/2 | 0/1/2 | 0/0/0 | 0 |
| KDA443 circle | 1/2/1 | 1/2/1 | 0/0/0 | 1/2/1 | 1/1/1 | 0/0/0 | 0 |
| KDA443 square | 0/1/4 | 0/0/3 | 0/0/1 | 0/0/2 | 0/0/2 | 0/0/0 | 0 |
| GRU419 circle | 1/1/2 | 0/1/2 | 0/0/0 | 0/1/2 | 0/0/1 | 0/0/0 | 0 |
| GRU419 square | 0/0/2 | 0/0/1 | 0/0/0 | 0/0/1 | 0/0/1 | 0/0/0 | 0 |
| GRU443 circle | 0/0/3 | 0/0/3 | 0/0/1 | 0/0/2 | 0/0/1 | 0/0/0 | 0 |
| GRU443 square | 0/1/3 | 0/0/3 | 0/0/1 | 0/0/2 | 0/0/2 | 0/0/0 | 0 |

三臂全失败合计2/7/20，合格失败1/5/18，但它们是失败条件化且capped的机会集合；不能拿错排率相除或将无低N失败匹配解释为“没有泛化损失”。严格三类pair和triple合计均0，**B5跨人数放大命题INCONCLUSIVE，交付科学状态INCOMPLETE_NO_STRICT_TRIPLES**。不改变caliper、cap、cases补显著性；一般动作错误与额外迁移损失不是同一命题，不预设A=B。

## 【被排除的竞争解释】

1. 对固定CPU资格轨迹，E3不是通过改变原5人的ORCA或强制父轨迹来“保持局部”。实际邻接、状态和评分输入均精确核验。
2. E5没有同时增加机器人任务长度。机器人起终点一直(0,−4)→(0,4)，只有行人位置/目标映射。
3. 参考的目标进度没有写入环境Reward；即时量逐位比对原score，所有controller消费同80命令和filter。
4. 不把CPU smoke成功执行、原5人初值嵌套、分数翻转或安全候选数本身作为导航收益。

正式192对E1/E3 raw进一步排除了本次已执行闭环中“仅增加这些远人的N标签就改变当前模型输入、全80即时评分或导航结果”的解释；不推广到任意20人配置。E5相对E2，circle的四模型与ORCA成功下降，square所有控制器的Return上升；这一几何差异不支持仅由名义面积密度决定损失。E5相对E1也未统一恢复原任务成绩；这是干预反例，不是唯一根因。

尚未排除：局部任务难度足以解释损失；TTC5外观测产生实际代价；原Value读出/监督不足；混合人数覆盖能够修复；安全filter屏蔽了必要动作；模型/环境运动合同差异。未采集的相反解释均不以“已排除”呈现。

工程无效运行保留：初次CPU入口用了不存在的grid helper，未生成实验结果；改为真实contracts.discrete_index_to_action。一次汇总缩进错误在正式调度前被语法测试发现，b25de81→29a0b18保留修复历史；它不是科学负结果，未改采样或判据。

## 【无法单独归因的交互作用】

E2自然增加N，同时改变ORCA邻接、局部占用和遇见时序；E4改新增行人局部位置也会影响原5人后续ORCA，不能称为“只改网络输入”。E5名义密度相同，局部遇见时间及路径分布仍不同。这些正是各自场景干预的真实效果，不能互相相减后声称唯一机制百分比。

冻结权重的R2全人接口测试属于观测消费干预，不等于R2训练成功；20人诊断标签、D2混合人数训练都不构成5人训练的OOD贡献。输入×训练覆盖的因果交互必须在第二轮相同合同下识别。

### 第二轮训练合同（未授权，禁止启动）

**关键冗余：N=5时R1=TTC5与R2=全合法人群若逐值完全一致，D1两格不是两个独立训练因素。**

| 评测单元 | 训练数据 | 训练实现 | 评测接口 | 独立训练任务 |
|---|---|---|---|---|
| R1+D1 | 仅N5 | 共同N5实现 | TTC5 | 每seed1个D1 |
| R2+D1 | 同一N5数据/轨迹 | 与上一格共享同一权重 | 全人群 | 0，冗余训练格 |
| R1+D2 | N5/10/20配额相等 | TTC5 Value＋合法全人安全 | TTC5 | 每seed1个 |
| R2+D2 | 同D2场景/预算配额 | 只作变N兼容、原256维关系/attention/pooling | 全人群 | 每seed1个 |

因此完整表为4格×2匹配seed=8个评测单元，**在N5全训练路径等价验收成立的前提下，最少3种训练×2seed=6个GRU训练任务**，而不是8个独立训练。R1D1与R2D1的bootstrap/历史/命令/Reward/监督若确实相同，可共享每个真实采样轨迹、IL minibatch顺序、初始化和最终权重；它们高度相关，不可当独立重复。

共享前必须验：N5 token和窗口bitwise；连续前向、loss、全部参数梯度、optimizer更新及随机探索动作相同。仅参数量相同、输出均值接近或单帧前向相同不够。若GPU kernel数值或目标消费不同，先FIX这个阻塞，不借数值噪声制造“四臂不同训练”。不能只因两个checkpoint最终相近便事后合并。

每seed的D1 corrected模型要新训练：本轮旧checkpoint没有接受历史修复后的采样，不能拿旧模型冒充修复后D1。两seed419/443是匹配重复，不仅用一个seed四臂。先GRU；只有实际导航效果达到预设门槛且用户另授权，才对KDA进行同样6任务确认。

**真实实现阻塞：第二轮目前不是可启动的训练计划，不能只改spatial就跑R2+D2。** 已亲读本地独立accepted工作树：train.py:735–739将IL窗口固定reshape为8×13；train.py:753–784将输入pad/truncate到34维，:848–849继续用该表示。utils/ppo_buffer.py:25、:43按固定obs_shape=(8,13)预分配完整窗口；ReplayBufferMC:343–354默认经过joint34_to_tokens，sample:414附近返回同shape窗口。当前缓存或replay消费路径没有保存R2所需的完整N20观测。

第二轮授权前必须完成以下实现合同及验收；当前仅写合同，未改源码、未采集IL、未训练：

- 保存每个真实时刻的机器人合法FullState与全部合法人群ObservableState(px,py,vx,vy,radius)，及terminal/reward/action/index/history边界/实际N。输入不得包含人类隐藏goal、未来真实轨迹或intent标签。R1/R2从同一原始合法帧派生；R1截TTC5，R2保留N人，维度为3+N个13维token。全N排序规则、真实history和候选窗口必须一致冻结。
- 旧34维IL缓存不能恢复N20其余15人，任何reshape/padding都不补回信息。D2必须新采集合法ORCA N5/10/20混合专家数据并保存未截断观测；D1也要证明IL原始N确为5。计入采集真实交互、ORCA控制、文件写入、token/窗口构建、丢弃片段及墙钟；不得把旧merged cache改人数标签后当全N IL。
- Replay/IL按实际N分桶，每个minibatch同shape，无跨N未mask padding，避免attention和maxpool被填充值污染。时序不足的左补帧沿用同一真实帧合同，不能与人群padding混淆。若将来改用padding，attention、aggregation及maxpool全部mask的正确性必须单独验收，本合同首选分桶。
- 所有桶共享**总replay容量200000**，而不是每N各200000。D2建议N5/N10/N20槽数66667/66667/66666，总和200000；固定等比例桶调度和总40000更新，末尾余数事前指定，不给每N各40000更新。D1总容量/总更新/总600000交互保持相同。桶内采样/淘汰、MC target、window边界与权重的预算单位一致，并报告占用和实际采样配额。
- R1D2虽然Value只用5人，也从全合法raw派生，安全评分仍观察全体；其N桶分组可与R2D2对齐。D2两策略闭环不同，不能声称共享真实后续训练轨迹；D1才在端到端等价PASS后共享权重/轨迹。
- 当前预分配方式下，仅float32 state窗口的算术内存为N5约1.860GiB、N20约5.347GiB、上述等额混合桶约3.409GiB，尚未计raw存储、tensor拷贝、IL缓存、optimizer和网络。不是显存实测；可提出frame存储＋采样构窗，但须验MC/窗口parity，不偷偷改变算法或容量单位。
- 授权后先timing门，分别测N5/10/20的采集步/s、构建窗口/s、IL/RL秒/更新、峰值host RSS/显存及推理p50/p95，覆盖真实分桶采样和全部临时拷贝；先做N5 forward/loss/全部梯度/更新bitwise共享验收。timing前不承诺6任务墙钟；超资源先FIX表示/存储实现并重新预算，不启动全量任务、不用旧吞吐代替全N测试。

**预算建议供第二轮授权冻结，当前不执行：**

- 相同原Reward、命令网格80、dt0.25、历史T24、256维表示、参数量、optimizer/lr/损失、MC定义、终止mask、训练/部署history规则；训练平滑/filter合同选一个公共版本并全臂相同，不借B4同时测试它们。
- D1整个IL与RL均仅N5；D2两个阶段均按N5/10/20等比例。原merged IL数据是否含高N必须回原始字段验证，不因token截成5个就把数据叫N5。若改为共用N5 IL，仅RL混合，则因素必须明确命名“RL覆盖”，不能称完整训练分布干预。
- 授权建议每任务IL50epoch，等数量可监督窗口/等IL optimizer steps；原日志为15000 demo轨迹，但长度不同不可只按轨迹数宣称预算匹配。先按固定seed、固定case序、每N配额采集，再截取相同数量合法窗口；全部采集交互和被丢弃帧单列。IL数据生成也需计入预算，不能免费新增混合专家数据。
- RL建议每臂600000真实env步、40000次optimizer更新、batch256、replay200000，复现原10000episode约60万步量级。相同episode数不是相同交互预算；D2每N200000步，更新频率按累计步调度。结束时不完整episode不伪造MC终局标签，保留尾段缺失数量；所有交互都计预算。该预算是新的匹配合同，不冒称逐字复现原4updates/episode。
- D2两输入臂共享scene seeds、人数配额与初始世界，不强制共享后续机器人轨迹：策略已经不同，强行同轨迹就变成off-policy诊断。D1训练两接口只有已证完全等价才共享。
- 参数量须相等；N20注意力/池化计算量可能增加，分别记录推理p50/p95、训练秒/更新、显存。不能把相同256维叫相同FLOPs。
- 固定验证/评测case分组，不以92000调参；与第一轮同一参考、场景、成功保护指标。D2在20人的收益仅诊断覆盖作用，外推结论需另一个事先冻结、未训练的人数分布；不在此自动添加。

**原始成本锚点（seed443，实际train.log，不使用旧5h52m估计）：**

| 模型 | 开始 | IL第50epoch结束 | RL首episode | 第10000episode | 保存完成 | 总墙钟 |
|---|---|---|---|---|---|---|
| GRU24 | 10-10 00:38:13 | 02:01:46 | 02:03:49 | 04:57:29 | 04:58:24 | 4h20m11s |
| KDA24 | 10-10 00:37:13 | 02:12:10 | 02:14:27 | 06:19:36 | 06:20:49 | 5h43m36s |

GRU日志末计599466个reward步（skip0），IL约1h23m33s、RL ep1→10000约2h53m40s。KDA为601048步，IL约1h34m57s、RL约4h05m09s。两者是在原最大4训练worker共享4090的调度下观察到的墙钟，不是独占GPU基准。

6个GRU任务直接串行套用旧墙钟相当26h01m06s，8任务会是34h41m28s；6个KDA相当34h21m36s。**这些仅是旧日志的算术乘法，不是已实现的新计划。** 原调度并发最多4，不能直接把串行和除以4当新计划。R2N20 attention N²、混合episode长度、合法IL新数据采集/构建、分桶重放与验证开销未测，**新全N20吞吐/内存UNVERIFIED**。第二轮不能承诺8h完成；授权后先实现上述阻塞及短timing门，主Agent据实锁定worker与墙钟，不自动全训。

成本原始定位与复算入口：

/home/abc/workspace/kda-history-replication-20261010/runs/gru-t24/train.log

/home/abc/workspace/kda-history-replication-20261010/runs/kda-t24/train.log

/home/abc/workspace/ab-causal-isolation-20261011/b/training-budget.py

GRU开始line1、IL50 line137、RL首line174、RL末line16975、完成line17014；KDA对应line1/136/173/16975/17014。GRU源SHA256为d3d3f13e6c4ed3554a16965a5512095f63065dd6c84a13b568f1c8050a6d6d3d，KDA为c6dcb0064c06e77a7f951442327bcd290b69b884947fcb7b24b4099fe3581128。脚本输出源SHA及原行，不能把成本推算包装成已经训练。

**第二轮要识别的精确对比：**

Δinput(D1)=J(同一N5权重,全人评测)−J(同一N5权重,TTC5评测)，仅评测接口效应；Δcoverage(R1)=J(R1,D2)−J(R1,D1)；Δinput(D2)=J(R2,D2)−J(R1,D2)，包括输入改变诱导的训练/采样变化；交互为Δinput(D2)−Δinput(D1)。差分效果不得误称互斥根因比例。任何D2收益不作五人训练OOD创新。

第二轮GO门建议：两seed都出现相同方向受控导航改善；固定20人case至少净增2/16成功或平均折扣Return>0.05且CR不升，5人成功破坏为0，actual swept碰撞不增。若四评测单元均无改善或只有不受保护收益，KILL输入/覆盖作为当前主要可修复候选，不增无关模块。未执行前科学裁决INCONCLUSIVE，不提前批准训练。

## 【最主要的可修复机制】

实际可修复的是某些冻结机会处的动作选择：按原合同完整成功匹配的KDA条件，A1有5次失败→成功，平均ΔG+.247694、成功对照零破坏，有限root门PASS。不能抹去这一已验证局部收益，也不能指定“输入不足”或“Value泛化”是主要总体根因。成功control的Return仍可能下降，事后失败定位并非在线触发器，N10唯一救援无control、跨N严格pair/triple为0。E3说明这些远人没有直接接口作用，ORCA与追加Mamba在原20人任务均受损，E4/E5说明名义拥挤与真实难度不单调；以上均未建立新的独立算法。本轮没有方法级进展。

科学裁决分别为SUPPORTED“已验E1/E3归档全轨迹/接口精确等价”、SUPPORTED“有限完整匹配root中一次冻结A1可产生受保护救援”；INCONCLUSIVE“20人额外损失主要来自Value迁移”及“同机会动作误差随5/10/20放大”。**主研究决策：KILL原冻结的额外Value迁移候选**，必要baseline门FAIL不被A1 PASS或追加参考覆盖。A1门仅支持合同定义的有限机制后续资格，不自动freeze新方法，也不授权训练。该KILL不是所有Value修复理论不可能的REFUTED命题。

## 【下一步是否具备算法研发依据】

METHOD_SPEC_FREEZE=NO。本轮具有有限root级成功保护收益，但没有合法在线触发、全分布闭环保护、最近方法的具体技术缺口或独立算法差分，不具备据此直接冻结投稿方法的依据。参考实现、公共bug修复和输入模块追加均不是本轮贡献；A1 root门PASS与额外迁移候选KILL可同时成立。

当前主行动是封存额外Value迁移候选，不按A1通过自动追加训练或复活泛化归因。B5因实际严格机会0保持INCOMPLETE，不扩大case/cap/caliper找显著性；已授权第一轮含N10及Mamba的CPU验收全部完成，停止本轮执行。A同世界输入、dup和readout结果未作为本报告证据引用，未继承其他Agent解释。第二轮仅保留上述合法全N数据路径阻塞、timing门及条件式6训练/8评测/至少2seed合同；未经授权不实施，不把旧4h20m11s乘6当可启动预算。
