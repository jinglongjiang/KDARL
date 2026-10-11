# A Direction: First-Round Causal Isolation

【研究结论】A2物理V/reward/terminal离线细分、A3全N及dup负对照、A4读出、A5有限冻结续策采集与读出已完成。完整A5学习因果及整个A根因归因仍INCOMPLETE：未授权匹配IL/RL。全N保护失败，A4失败，A5有限续策中历史修复未改变终局。本轮没有方法级进展。

【核心证据】dup与TTC5的80/80根command/reward/physical-state序列精确相同，G差为0；allN救援27/40失败根，却破坏4/40成功保护根并新增4个环境碰撞。A4门FAIL。旧20Square全部G改善对中actualTTC仍反向为KDA419 107对/75根、443 156/87、GRU419 86/49、443 75/51；不全属原Qpred严格错排，GRU419其中严格错排残余为85对/49根，另1对原预测近似平局。

【科学裁决】REFUTED：一步纠正消除全部原排序错、当前全N替换无成功/安全损害、同5人重复至20可复现全N效果。SUPPORTED：冻结递归输入与续策合同可改变实际终局。INCONCLUSIVE：各因素自然分布归因比例、内容与其他统计变化的独立份额。

【研究决策】KILL：封存当前无条件全N、线性MC/pairwise和filter绕过候选。dup负对照不推翻原保护失败；不启动IL/RL或第二轮训练。

【执行结果或下一动作】A2细分、count逐步核验、训练/部署合同及最终验收表已封存；原collector/旧A2产物不变。本轮A执行停止，无新GPU或训练。

## 【研究问题】

本轮识别四种不同问题，不把它们合并为“时序骨干失败”：一步物理/奖励误差、下一帧 TTC 观测选择误差、同表示读出监督不足、生成动作/标签的训练与部署续策差异。

独立实现边界：A 只修改 a/project 的空间编码硬编码 N 维度；主 Agent 提供 history-only 184f6f0 与 terminal-only 9f06223，A cherry-pick 后本地 HEAD 为 0b17908。原工程和 b/engine 未修改。远端由 main 使用 runtime-project 的真实 git worktree。

每项实验的 GO/KILL 问题、预算、保护对照已在实际运行前的 protocol 固定：[A2 protocol](a/artifacts/a2-protocol.json)、[A4 protocol](a/artifacts/readout-complete/readout-summary.json)、[A5 collector](a/paired5-continuation.py)、[A3 closed-loop plan](a/artifacts/closedloop-heldout-caliper-plan.json)。全部正式 root/case 选择不读取新续跑 G。

完整 branch G 是固定世界、固定旧部署策略续策的确定性实现回报，不是 Q*，也不是未指定条件分布下的期望 Q。20 人完整标签是诊断监督，不称 OOD。

## 【已经确认的直接机制】

1. 在线 actor 的旧 epsilon early-return 未追加真实当前帧；修复只增加该帧追加。ReplayBufferMC 的 episode 重建使用逐步真实访问状态的连续存档，包含 epsilon 帧；训练存档可能带下述raw34噪声，不等同于无噪声actor窗口。不能称训练 buffer 也缺 epsilon 帧。
2. 终止动作原评分仍 bootstrap。主修复沿已有 endpoint collision/success_radius 与已知 timeout 判据加 mask，不统一 rpred 与环境 swept reward/到达半径。
3. 原空间编码器只使用 5 个已选人类；合法全 N 替换保持 13 维 token、256 维编码、参数、attention/maxpool 不变，仅把 slice 和两处 reshape 的 5 改为 N。

源码：[spatial/online actor/scoring](a/project/crowd_nav/policy/mamba_rl.py)、[原 token contract](a/project/crowd_nav/contracts.py)、[ReplayBufferMC](a/project/crowd_nav/utils/ppo_buffer.py)。前两项属于基线维护事实，不是新算法贡献。

主公共验收原文：[acceptance](acceptance.json)。8 个冻结 episode 的 original/history-only action/history/window/score 重现 PASS；terminal-fixed 在这些 episode 的 outcome/time/G 不变，但部分最后动作不同。此结论仅覆盖已验收 8 集，不外推为全部根无影响。

### 训练、部署与本轮A5的完整合同边界

以下为A独立读取本地冻结源码后的合同记录，不是新增实验，也未将这些合同统一。范围为train.py的非vectorized Explorer采集分支及本轮固定checkpoint部署；不冒称重现全部历史训练运行。

|合同项|原训练源码路径|冻结部署 / 本轮A5|
|---|---|---|
|模型mode与dropout|train.py:2330每ep调用policy.train()；spatial fusion含p=.1的Dropout，时序dropout读取配置（419两backbone配置为.03，生效方式随实现）|部署与A5四臂均net.eval()、dropout关闭；A5的train标签指续策规则，不指PyTorch train mode|
|探索及动作规则|train.py:2546-2561按配置epsilon_start=.3→epsilon_end=.05调度，phase='train'采集；无仅test/val/eval启用的平滑/filter/risk|部署epsilon0、alpha=.3、clearance=.2/risk=.8；A5 train规则固定epsilon=.05且无上述部署规则，不重演完整epsilon日程|
|在线actor历史|actor从JointState决策；旧epsilon分支跳过在线history追加，公共修复补追加|A5仅old/fixed切换此post-root行为，四臂使用同checkpoint、同起始真状态及隔离探索RNG|
|raw34训练存档噪声|explorer.py:102-128先转float32、补齐/截为34维；env.phase=='train'时逐维加高斯噪声，均值0、标准差.02；451-466在动作已选出后调用并记录s_arr|部署无该Explorer训练存档噪声；A5使用恢复的真实状态及真实窗口，不复现该replay噪声。不能将此源码路径称作直接向actor输入注入噪声|
|轨迹入库及终止|explorer.py:547-555保留RL轨迹；train.py:2570-2609将success/collision/timeout轨迹push_episode，最后一个reward对应dones[-1]=True；离散index缺失等校验仍可能skip|本轮无原训练buffer更新、无完整学习；A5保留early-terminal分母且不替换root，实际64个root均有效|
|Replay窗口与MC标签|ppo_buffer.py:226的push_episode转store_episode；343-397将存档转tok，按所有连续步（包括epsilon）重建窗口，短前缀repeat首帧padding；从真实环境reward逆序累加MC G|不是从旧actor稀疏_history复制训练窗口；“连续真实访问帧”不等于“未加噪声token”。A5的257参数/300steps只读出一个固定root动作标签，不重演原buffer学习|
|权重及目标分布|原训练逐ep采集/优化，回放可来自不同历史权重下的轨迹|A5冻结单一checkpoint、同真state/action预算；未复现过去checkpoint混合或完整训练状态分布|

源码定位：[Explorer转换/存档](a/project/crowd_nav/utils/explorer.py)、[逐集mode/epsilon/轨迹push](a/project/crowd_nav/train.py)、[ReplayBufferMC窗口及MC](a/project/crowd_nav/utils/ppo_buffer.py)、[Dropout及phase条件动作规则](a/project/crowd_nav/policy/mamba_rl.py)。核验源码SHA256：explorer=670def0cd05874492c4d3e3ab96034079c4f09b47662650e98302c5df4c2a3a1，train=b777188aedfcb61375f0d28c9748a7655bef8612ae90c5be24f0e40f789315ca，buffer=4155eba3c199165320f55aeefa1f4e7693a2b04c523ee3ffadb2892ce96b2744。

SUPPORTED仅指上述源码合同差异存在。噪声/dropout各自导致原错排、导航失败或A5终局变化的命题均INCONCLUSIVE：没有对应单因素干预，不能把“存在差异”写成“已证实因果”。本轮没有统一训练/部署，没有证明统一它们会改善导航；A5的有限结果不包含这些未干预因素，也不授权追加实验。

## 【已执行的因果干预】

### A1/A2 原始标签与严格三项分解：COMPLETE

定义 D=alternative-A0，P=Qpred，F=Qpinned，A=Qactual，G=完整旧部署续策回报：

`D(G-P) = D(F-P) + D(A-F) + D(G-A)`

- 第一项：真实人体下一位置/速度、真实即时 reward 与 terminal mask，保持 root-pinned IDs。
- 第二项：保持相同真实物理下一状态及过去前缀，仅切换为真正下一时刻的 TTC 选人/编码。
- 第三项：未来值残差，混合值拟合、历史混叠、目标续策和分布差异，不命名为 KDA 长期记忆失效。

实际 CPU 单次流式复算约 28.35 秒；核对 root 全字段、去重 action index、逐步重算 G、核对原 oracle flags，恒等式最大误差 2.220446049250313e-16。逐动作对含 source gzip 行号、执行 commands、四种数值与三项差：[raw pairs](a/artifacts/a2-pairs.jsonl.gz)、[summary](a/artifacts/a2-summary.json)、[复算入口](a/a2-decompose.py)。这是误差恒等式，不是模块的因果贡献百分比。

追加离线细分已完成，未重采、未调用模型、未覆盖旧A2产物：利用原branch的first_step.value/predicted_value/reward/q_oracle1_unmasked/q_oracle1_terminal_corrected和snapshot.rewards，固定原a2-pairs中original_strict_error分母，分三步重建：①原rpred+真实pinned V；②再换真实即时reward；③再应用实际terminal mask。一次CPU流式处理1,661对/23个model-cell-scope组，20.61秒，PASS。[独立入口](a/a2-factor-split.py)、[协议及源SHA](a/artifacts/a2-factor-split/protocol.json)、[逐对原始组件与行号](a/artifacts/a2-factor-split/pairs.jsonl.gz)、[全部分组与浮点残差](a/artifacts/a2-factor-split/summary.json)

该physical-only是保持pinned IDs、替换真实下一帧位置/速度等字段后的V，不是纯CV位置误差。原gamma*V按torch.float32先乘，再按原reward tensor dtype相加；oracle保持原float32乘积转Python float后加real reward。P全80候选、unmasked oracle和terminal F重建最大残差均为0；若错误改用float64乘法，差可达3.8743019126e-8，额外float32 reward加法舍入可达2.9549378167e-8。零重建残差仅针对存档数值/运算合同，不宣称实数无限精度。

### A3 合法全 N 表示接口：提取及原 V 实际效果 COMPLETE

原 contracts 函数通过结构化 AST 仅扩展 token shape、人类 slice/reshape、TTC 截取 N；所有字段计算复用，不手拼字段。保留机器人 float32 往返、人体精度、TTC 排序、零 padding、social/ttc 字段和 repeat-first 左填充。每个真实 prefix frame 都重现原 token；original windows hash 强制一致，actualTTC windows 也核对原 oracle hash。

N5 gate 包括原/扩展 token、原/扩展 spatial、temporal latent256 和 V bitwise；5 人全 N/原窗口 bitwise。CPU GRU 冒烟通过；6 项工程单测通过，另覆盖 terminal score 数值顺序和正式 heldout caliper plan。跨 CPU/GPU 的旧 V 仅作误差容差检查，未称 bitwise。

主 GPU 全量1,279 roots PASS；本地已读取4个extract-done及protocol：KDA419/443分别336/367根，GRU419/443分别292/284根，合计1,279。四者max_saved_value_error均为0、GPU_archival_bitwise=true；done记录各自99.05/112.13/87.28/86.99秒，主调度含启动开销418秒。正式A4原metrics已独立读取，不再仅引用主转述。

入口：[run-a](a/run-a.py)、[维度扩展](a/all-n-contract.py)、[tests](a/test-a.py)。run-a 的 archival scoring 显式从 cd8b793 绑定，故保留旧 Qpred；不充作修复后的新闭环 baseline。

### A4 等容量读出：COMPLETE，gate FAIL

所有头 linear256->1，257 参数，原 V 初始化，固定相同 candidate G 标签预算，300 full-batch Adam steps，lr=.001。case%5 train0/1/2、dev3、test4，同 case 跨 seed 不泄漏；无 dev 选参或早停。MC 与 pairwise 不增加 rollout 标签。P4 同时提供 MC/pairwise。

预测窗口固定原 rpred；actual 窗口使用真实 reward/terminal，后者是 oracle-only 诊断。原 r_env 混入预测窗口的早期 CPU 读出保留但撤回，不作为正式证据。正式版本：[readout source](a/readout-a.py)、[complete raw metrics](a/artifacts/readout-complete/readout-summary.json)。14 groups/70 records 包括 6 个无法形成 train/dev/test 的小 5 人组 INCOMPLETE，不能称 70 个有效训练臂。

### A5 有限冻结续策采集与读出：COMPLETE；完整学习因果：INCOMPLETE

固定 5Circle cases90000..90015、t=5s(step20)，每模型保持分母16。重放原 prefix，四臂从完全相同真实 state/完整 prefix 出发，共享强制 A0 一步，随后 old/fixed 在线历史 × train/deploy 续策。

train 是冻结 checkpoint 下 epsilon=.05、无部署平滑/filter/risk；deploy epsilon0、alpha=.3、clearance=.2/risk=.8。old/fixed 只切换 post-root epsilon 在线追加，terminal 修复四臂一致。所有臂保持net.eval()，dropout关闭，并逐臂记录所有module的eval状态；不复现原训练actor的train-mode dropout、过去checkpoint混合或replay状态噪声。这是控制其他因素的冻结续策合同，不是完整原training行为再现。私有固定探索 RNG 与环境 RNG 隔离；部署 old/fixed 的 command/reward 全迹必须相同。提前终止保留分母且不替换。N=5 direct first5 与 TTC5 为同集合，未将它推广到20人。

同一真实 next window 的 latent256 对四臂标签分别做匹配 257 参数读出，fit<=300 steps，无 IL/RL。该实验识别冻结生成策略合同的有限干预效应，不是完整学习/重训练的因果效应。[A5 collector](a/paired5-continuation.py)

实际4model×16case×4臂=256续跑完成，无提前结束root；old/fixed deploy的全部command/reward全迹一致，四臂共享root/history/true-next-window/forced command及reward。两口径sweep覆盖全部续跑。本地逐步重算G与summary对齐PASS。[原rows](a/artifacts/paired5-complete/paired5-rows.jsonl.gz)、[协议](a/artifacts/paired5-complete/paired5-protocol.json)、[有限CPU读出](a/artifacts/paired5-complete/paired5-readout.json)、[A3/A5独立复算](a/artifacts/verified-a3-a5.json)

|A5 model|history修复(train)command改变根/16|history修复(train)平均deltaG|train→deploy平均deltaG(old/fixed)|实际终局改变|
|---|---:|---:|---:|---|
|KDA419|5|+.000904374|+.005658997/+.004754623|0/16|
|KDA443|7|0|-.042948388/-.042948388|1/16，success→timeout|
|GRU419|2|0|+.007707981/+.007707981|0/16|
|GRU443|5|-.000444096|+.006526553/+.006970649|0/16|

history修复在四model的train和deploy均未改变终局；deploy效应全为0。history×deployment的G差分交互分别为-.000904374/0/0/+.000444096，终局交互均0。KDA443 case90005(train split)的train→deploy导致success→timeout，deltaG=-.7853516799，old/fixed两种历史都如此；不能归为epsilon漏帧。GRU419每臂均15success+1collision，其余KDA419/GRU443每臂16success，KDA443两train臂16success、两deploy臂15success+1timeout。全部环境success均无actual swept负clearance；共同碰撞并未被合同切换修复。

CPU读出每model每臂257参数/300steps，test仅3个固定case；同case跨seed分组不泄漏。old-train target的test MSE旧V→fit：KDA419 .001512964→.000767147、KDA443 .003122485→.000852007、GRU419 .003485263→.000293277、GRU443 .001513791→.000392706。old-deploy则为.001452320→.000111399、.003507817→.101506703、.004287247→.000121578、.002514526→.000182031；KDA443明显失败必须保留。它们只检验一个固定root动作的标量目标校准，不检验80动作排序，不授权新head闭环，也不推翻A4 gate。可支持“冻结续策合同改变某些目标/一个实际终局”，不足以建立稳定受保护改进或完整学习因果；不继续投资本轮续策-only候选，不把缺证据称技术不可识别。

先前仅部署读出对 GRU419/443 test MSE 为 .003704→.024306 / .024039→.025449；两 seed 不改善。此结果不替代必要的四臂采集。“缺训练续策标签”是先前未采集，不是技术不可识别。

## 【各因素的实际导航效果】

### A2 20Square：G 改善对，不是获救 episode

|Model|失败改善对/独立根|actual 正确对/根|actual 仍反向对/根|成功对照改善对/根|
|---|---:|---:|---:|---:|
|KDA419|192/96|85/58|107/75|118/67|
|KDA443|216/111|60/45|156/87|164/99|
|GRU419|126/70|40/32|86/49|96/57|
|GRU443|127/74|52/38|75/51|126/72|

GRU419 有一个原 Qpred 在1e-6阈值内的近似平局，所以“原严格错误且最终仍错”为85，不是把86全部称原严格错误。各组独立 case 计数见原 summary。

固定原Qpred严格错排分母的20Square细分如下；每格为动作对/独立root数，后两列是相对前一阶段的新增纠正，不是累计正确数：

|Model/原scope|固定错排分母|physical V纠正|再realReward纠正|再terminal纠正|最终pinned F仍错|
|---|---:|---:|---:|---:|---:|
|KDA419/失败|192/96|37/24|0/0|0/0|155/89|
|KDA443/失败|216/111|38/32|0/0|0/0|178/100|
|GRU419/失败|125/70|19/16|0/0|0/0|106/66|
|GRU443/失败|127/74|31/28|2/2|0/0|94/58|
|KDA419/成功保护|118/67|22/19|0/0|0/0|96/57|
|KDA443/成功保护|164/99|26/24|0/0|0/0|138/86|
|GRU419/成功保护|96/57|12/11|0/0|0/0|84/52|
|GRU443/成功保护|126/72|27/27|1/1|0/0|98/58|

以上每组physical V、realReward、terminal三阶段反向纠正均为0对/0根；全部23组也均为0。physical反向为0部分是分母设计使然：初始全为错排，没有原本正确对可被首阶段破坏，不能称物理替换一般不会反向。reward与terminal仍可撤销前阶段纠正，但本固定分母未发生。terminal增量为0对应这些对首步均非terminal，不否认其他状态或后续终止mask修复的工程必要性。GRU419固定严格分母125，不混入前表126个G改善对中的近似平局。各纠正集合root可重叠，不能把31/28与2/2的根数相加；GRU443失败最终F正确33对/29根。

作为另一分母的保留结果，对全部严格非平局 G 对，合并pinned F纠正与actualTTC历史纠正均有反向纠正：

|Model/失败组|合并F纠正 对/根|合并F反纠正 对/根|history 纠正 对/根|history 反纠正 对/根|
|---|---:|---:|---:|---:|
|KDA419|37/24|7/7|62/48|35/28|
|KDA443|38/32|8/8|41/29|32/28|
|GRU419|19/16|7/6|29/25|14/14|
|GRU443|33/29|25/19|35/28|28/25|

这些列的分母是全部严格 G 对，不能与只计 G 改善的第一表或上述固定原错排表直接相减；合并F包含physical V、reward及terminal，不再简称CV误差。本表各列独立 case 数与根数相同，不能把各列根数相加成总分母。逐 flag 计数见原summary；新增离线细分独立保留，不改旧数字。

具体原始反例为KDA seed443、20Square、case90021、root239、step25/t6.25s的A0=69与预设A2=24（同root另有A1=9和A3=64，不能混对）：D(P)=-.03443247，D(F)=-.03451210，D(A)=-.03966385，D(G)=+.63607687。三项分别 -.00007963、-.00515175、+.67574073。A0 command=(.413570863,-.428355106)，A2=(.145692460,.918360566)；预测clearance .734443/.422896均过.2，r均0，raw与filtered评分相同.677033544/.642601073。原真实续跑A0 timeout50.25s/rootG=-.088990890，A2 success20.25s/rootG=.547085984；即没有filter否决，真实一步后继及真实TTC也没有修正这一动作对的排序。证据为a2-pairs的root239/alternative24、旧branches-kda-s443-t24-cell5.jsonl.gz第18行及roots.json快照。它是旧冻结续策实现回报反例，不证明残差属于某一个机制或条件期望Q错误。

### A4 预测窗口 heldout 排名与成功保护

|Model|P1|P2 MC|P3 pairwise|P4 MC|P4 pairwise|
|---|---:|---:|---:|---:|---:|
|KDA419|.541880|.521368|.542735|.456410|.485470|
|KDA443|.471384|.525786|.486792|.519182|.486792|
|GRU419|.469524|.472857|.499048|.491905|.476190|
|GRU443|.525926|.483333|.543519|.534259|.529630|

P3 相对 P1：KDA419/443 为 +.000855/+.015409，GRU419/443 为 +.029524/+.017593；均未达两 seed 至少+.05。KDA P3 成功对照 regret .041742→.126422 / .043319→.171606；GRU亦上升。MC、全 N 读出均未通过联合门槛。动作集内 regret 局部下降不能推翻 gate。

### 同世界全 N 闭环：COMPLETE，保护门FAIL

正式授权仅 TTC5-V/allN-V，原 head，不训练。每模型 first10 heldout failure，在 heldout 原 success-root pool 按 time2s/dgoal1/nsafe16/crowd2m2 caliper 最近匹配，无放回；不足列 unmatched，剩余最多10成功保护列 independent_not_matched。匹配 KDA419/443=6/8，GRU419/443=8/6；其余失败保持原分母。共80根×2臂，另8次 A0完整重复门。

每模型首 failure/success 的 baseline 全续跑重复 hash 必须一致；根 current/history/windows 与快照 exact，原 V bitwise，terminal raw mask 差异独立列。新 baseline failure→success 时，不用旧 failure scope 冒称 rescue，救援分母按实际 repaired TTC5-V outcome 定义。

A3/A5在队列启动前补入只读几何安全日志，不改决策/arm/seed/root，不增加模型调用：每步同步计算min_t ||d0+t*(实际执行robot command-human velocity)||-两者radius，t∈[0,.25]。env口径使用step前当前人速，actual口径使用真实human位移/.25；A5包括强制root一步。每步保存两口径clearance、整续跑保存各自min及负clearance时间步数，physical_collision_count为actual口径的时间步计数，不是不同人或独立碰撞事件数。summary分别报告环境碰撞outcome和两口径几何碰撞，特别列出环境success但actual几何重叠的情况；旧无sweep文件标INCOMPLETE，不拿endpoint补值。同步几何与两入口实现一致、零相对速度及输入/RNG不变已通过CPU最小测试，正式A3/A5全量原rows的clearance最小值/计数已复核PASS。

正式A3执行前另补首rootstep只读root_decision日志：original_parent及used_input各自保存80个values/raw_scores/scores、windows_sha256/current_input_sha256与shape，TTC5 baseline同字段；只读取已有forward结果，不额外调用模型或改变选择。补后closedloop SHA256=a0db37797b06d704087f45cbc719eb35307624e9eeb42500ba5a39b912abfad3，paired5 SHA256=5c6941071c5bf244f9257efce0edb191640e057ad8b57ab6dce4d0e3e1a997a8。5项CPU工程测试exit0，其中日志测试确认读取完整且不修改原张量。collector至此完全冻结；旧smoke未含该日志，不作为该日志项的验收证据。

递归输入替换与单次 all N 选动作后恢复 parent 是两个不同 estimands；旧单次入口/六臂计划保留但当前未执行。最新授权是递归两臂；六臂和新 head 闭环均已撤销，代码硬门禁止。[closed-loop source](a/closedloop-a.py)、[正式 plan](a/artifacts/closedloop-heldout-caliper-plan.json)

实际80根160条科学续跑+8次A0重复门完成，主执行539.96秒。A逐条复算G、完整steps hash、新baseline、两口径sweep计数、root两套80评分及跨臂parent一致性PASS；8个A0完整重复hash均相同，80个root_decision检查均PASS。实际newbaseline各model均10失败+10成功，故本次分母与旧scope碰巧一致，但实现仍按实际终局定义。[原rows](a/artifacts/closedloop-complete/closedloop-rows.jsonl.gz)、[协议及实际日志](a/artifacts/closedloop-complete/closedloop-protocol.json)、[独立复算及2000次case bootstrap](a/artifacts/verified-a3-a5.json)

|Model|失败获救/10|成功破坏/10|新增/移除环境碰撞(20根)|新增/移除actual几何碰撞|20根平均deltaG|共同成功平均delta时间(s)|步骤加权latency TTC/allN(ms)|
|---|---:|---:|---:|---:|---:|---:|---:|
|KDA419|6|0|1/1|1/1|+.212100|-4.725|16.77/41.32|
|KDA443|6|1|0/1|0/1|+.250608|-4.472|16.40/41.04|
|GRU419|7|2|2/2|2/2|+.208108|-3.719|14.42/36.42|
|GRU443|8|1|1/3|1/2|+.320824|-3.028|14.15/36.89|

KDA419新增碰撞发生于旧失败组，移除另一碰撞不能抵销预设逐根新增安全损害；KDA443成功破坏为timeout，GRU两seed成功破坏均为collision。环境success均没有两口径负swept clearance。GRU443移除3个环境碰撞但仅移除2个actual几何碰撞，明确保留环境/物理口径差异，不强行统一。latency含原TTC诊断计算+全N计算以及root日志/重复评分开销，不冒认优化后独立allN部署成本。case bootstrap仅描述固定样本不确定性、不新增显著性门；逐model/原失败/成功保护/匹配status分层原数值均保存在独立复算JSON。

日志边界补核：closedloop rows本身只记录root后的160条科学续跑，8次重复验收另存门记录。主Agent用四份旧baseline的实际相邻帧和同一真实command，离线逐步重算80个共同prefix的env/actual扫掠，并与原始存档逐位数值一致（最大误差0）；80个prefix均无actual负净距。接上现有续跑后，所有环境success的完整episode也无actual几何重叠。此补核不新增rollout或修改raw，不把仅root后min冒称原本已存全程min。[全程拼接安全核验](a/artifacts/verified-prefix-safety.json)、[只读入口](check-a-prefix-safety.py)。初次把168个执行次数误当160个科学gzip行数的断言失败及修正保留于execution-corrections.json，未改变实验分母。

不能把该递归策略G差归因于单个root动作。A独立核验两例：KDA419 root108/case90249在root step20两臂同为index79，command逐位相同，首步reward/sweep/物理state hash也相同；首次动作分歧在step21（TTC5 index63、allN index10），最终timeout→collision，deltaG=-.4688348201。KDA443 root414/case90354在step20两臂同为index49且command/首步物理state hash相同；step21才分歧（index49→23），最终success→timeout，deltaG=-.4516336225。因此这些结果是root之后持续输入/续策规则替换的效果，不是root选择了不同动作的单步branch效应。两例root评分hash虽不同，不能据此把后续终局损害归给已相同的首动作。

SUPPORTED：本固定样本、冻结权重、递归input替换产生实际救援和损害；REFUTED：当前无条件全N替换满足成功/安全保护。研究决策KILL当前全N-only受保护部署候选。下面的数量控制排除“仅重复同5人即可复现效果”，但不把净成功改善称为自然分布信息截断的主因，也不分离所有内容与统计交互。

### A3 token-count负对照：COMPLETE

独立count-matched-a.py固定同80roots，不改原collector或allN输出。每帧将parent out.windows的5human rows各repeat4次，3robot/goal/motion行逐位不动，实际used窗口为[80,24,23,13]；来自parent同一真实TTC历史/当前锚定候选，不按候选重选人，不引入额外15人信息。它是统计负对照，不是20个独立行人world。结构化AST只替换current/full_windows两个构造，N5原窗口/值/score gate保留；upstream SHA强制匹配a0db3779，wrapper SHA为d83501c224b785ecd35d6014c5d314ed2aecbbde73dedf4e0bdecef4fee5e4bd。[wrapper](a/count-matched-a.py)、[3项CPU门](a/test-count-matched-a.py)

主已执行smoke及正式80根×2臂+8次A0重复门，formal PASS80。负对照的TTC5 baseline与已有formalTTC完整steps/hash/outcome/time/G exact，根used windowhash/currenthash及80values/raw/final均由真正重复输入产生。科学比较使用canonical count-matched-rows，不把wrapper内部遗留allN-V标签当真实全N。仅allN优于dup且原保护通过才能推进，是实验前已有合同；本次第二项已FAIL，不重选根、不新增科学门槛，不自动复活候选。[原count rows](a/artifacts/count-matched-complete/count-matched-rows.jsonl.gz)、[执行summary及源码SHA](a/artifacts/count-matched-complete/count-matched-summary.json)、[主CPU逐根核验](a/artifacts/verified-count-control.json)

A从canonical原rows独立复算PASS80：[独立完整核验](a/artifacts/verified-count-independent.json)。count原rows SHA256=2da67c7dda09b9c03985b1b4d4059c58090244b5c58253a35feb5f38ff8fa6d6，原formal SHA256=27390c1fa8d6e955e7b2e414326407d65a22054299b7caf8a36554d1b52b50b2，与主核验源hash相同。校验同80冻结根/plan SHA、原baseline全日志重现、三臂根80commands/rpred/clearance/terminal及original_parent不变、真实used窗口shape和80values/raw/scores长度、折扣G及双口径sweep最小值/计数。

TTC5对dup的实际command序列精确相同80/80、reward序列80/80、physical_state_sha256序列80/80，三项联合一致80/80；每model分别20/20，每失败/成功保护组分别10/10，最大abs(deltaG)=0。完整step日志仅0/80相同，real_history与score hash本应不同；没有拿整体日志hash不同冒称物理轨迹不同，也没有仅凭G相等断言轨迹相同。上述统计独立于同一TTC baseline在两次执行间要求的全日志exact复现。

dup四model每个失败/成功保护组均10根，G改善/下降(1e-6)均0，rescue/damage及新增碰撞均0。allN相对dup的失败组描述性case-bootstrap如下（2000次、固定seed20261011；非新增gate、非自然成功率估计）：

|Model|dup rescue/damage(20根)|allN rescue/damage(20根)|失败10根allN−dup平均deltaG|描述性95% case区间|
|---|---:|---:|---:|---|
|KDA419|0/0|6/0|+.354571|[+.085359,+.612048]|
|KDA443|0/0|6/1|+.451290|[+.205330,+.693521]|
|GRU419|0/0|7/2|+.563266|[+.317783,+.793007]|
|GRU443|0/0|8/1|+.681419|[+.442242,+.894989]|

REFUTED的具体竞争解释是“在同冻结根、原encoder/head及原安全规则下，将同5人重复至20足以复现allN闭环效果”。支持内容敏感的输入作用，不证明所有收益均来自某15人的独立信息、不排除内容×attention统计交互，也不构成新导航算法。原allN仍新增4个环境碰撞并破坏4个成功保护根，保护失败不能被dup无变化或均值/区间抵销；最终KILL保持不变。

## 【被排除的竞争解释】

- “全部排序损失只是一步 CV/pinned 物理误差”：REFUTED，真实下一 TTC 历史仍有大量反向对。
- “root 一步 terminal mask 漏项单独造成这些原始改善对的排序错误”：REFUTED，这些改善对首步均非 terminal，root 的 bootstrap mask 不变。此检验不排除修复改变后续动作/新 G；必须保留工程修复，也不能把旧 G 直接冒作修复后的结果。
- “这些错误是 KDA 独有”：REFUTED，已有GRU同类实际反例；另一个命题“KDA是否额外严重”仍INCONCLUSIVE，本轮不能据此替换骨干。
- “同容量 pairwise/MC 重拟合已足够可靠”：本轮固定门槛不通过；KILL 当前有限候选，不外推为所有更大模型理论无效。
- “训练 buffer 缺 epsilon 帧”：不符合当前源码；旧 actor 缺帧与 buffer 连续真实窗口必须分开。
- “全N收益只是把同5人重复成20的token-count效应”：在同80固定根控制中REFUTED；不外推为所有池化/分布统计解释均已排除。
- “运行成功、bitwise 重现、信息可读出即独立贡献”：均不成立；这些只是工程/诊断事实。

主filter-probe按原 root rawtop 被 mask 的事实选样，每模型 first2 failure+first2 control，原 rawtop 预设 alternative，不看 future G；仅强制一次后恢复 parent，共16根32续跑。A已独立阅读源码并逐条复算COMPLETE：校验源roots SHA、原rawtop预选及clearance<.2、两臂root前所有steps相同、root80评分与命令相同、强制动作index正确、全程/root折扣G、时间及actual swept min；与主summary所有逐根deltaG/rescue/damage/newcollision一致。[执行源码](filter-probe.py)、[原plan](filter-probe/plan.json)、[原records](filter-probe/records.jsonl.gz)、[A独立核验](a/artifacts/verified-filter-probe.json)

|Model|独立根|G改善|新baseline救援|成功破坏|新增环境碰撞|新增actual几何碰撞|
|---|---:|---:|---:|---:|---:|---:|
|KDA419|4|2|1|0|0|0|
|KDA443|4|2|1|1|0|0|
|GRU419|4|1|0|0|0|0|
|GRU443|4|1|0|1|1|1|

KDA root42/case90070与root284/case90090的deltaG分别+.4492784578/+.6933188518并timeout→success；强制command的预测clearance分别.181613/.196577，均违反原.2 margin。root336/case90201 deltaG=-.5829919201且success→timeout；GRU443 root716/case90067 deltaG=-.3895380348且success→timeout。GRU443 root708/case90050 timeout→collision，deltaG=-.5046467873，actual swept min由+.2015622944变为-.0130738227，物理安全破坏不是只看环境标签。以上rescue/damage均以实际修复后A0续跑终局为准。

科学裁决SUPPORTED：选定根存在被过滤但实现回报更高的动作；REFUTED：此无条件绕过在预定保护集无成功/安全破坏。研究决策维持KILL bypass：原预注册成功/碰撞保护门FAIL，2例救援不等于filter错误，因为强制被mask command违反原.2 margin约束；不能用局部G收益绕过保护门，也不由16个定向选样根估计自然损失比例。A不重复执行或改正式filter。

## 【无法单独归因的交互作用】

隐藏意图实验由 main 执行，A 已阅读原始代码、48条 records 并逐步复算 root G：8根每根6条世界/动作记录，根合同 hash 每根唯一，5根(8,229,499,508,682)出现 realized A1-A0 G 符号翻转。root8 原+.54254297、+90°-.06452832、-90°-.02165372。[main protocol/source](hidden-intent.py)、[raw records](hidden-intent/records.jsonl.gz)、[summary](hidden-intent/summary.json)

干预只转 omitted humans 的隐藏 goals，当前及过去所有合法可见全 N state、TTC tokens、窗口、80 commands/r/V/filter 不变；未来由真实 ORCA演化。因此它是人工未来意图敏感性反例，不是可观测15人截断的效应。5/8不是自然失败根因比例，不是自然 static-goal 分布采样，不支持“全 N 可解除该混叠”，也不是条件期望 Q 的估计。

主 step0 observation-alias 已执行，A 独立只读核验 COMPLETE：16根、64条完整续跑，8根发生 alternative-A0 realized G 符号翻转，反转根为0、2、237、222、226、504、685、691。每根四条记录完整，首动作与冻结候选命令相同，逐步奖励折扣和与原记录 G 的绝对差小于1e-12，时间/terminal索引一致；summary反转与逐根数值全部相符。[原执行源码](observation-alias.py)、[root-only plan/资格记录](observation-alias/plan.json)、[原 records](observation-alias/records.jsonl.gz)、[A独立复算](a/artifacts/verified-alias-and-readout.json)、[CPU核验入口](a/summarize-a.py)

|人工world twins|失败采样根反转/根数|成功保护采样根反转/根数|
|---|---:|---:|
|KDA419|0/2|2/2|
|KDA443|1/2|2/2|
|GRU419|1/2|0/2|
|GRU443|2/2|0/2|

T24 是 step0 合法真实首帧 padding，未拼接过去帧；机器人及选中5人的完整 state/goals不变，omitted15重置远lane。16根记录的资格检查均通过 windows/value/reward/clearance/raw scores/final scores/current token/selected IDs/80commands exact，且history为空；续跑源码在每臂首步再次断言window hash、IDs、scores、commands一致。A只复核这些执行记录，未自行GPU重执行。

终局改变的主要展示case为KDA419 root0/case90000，A0=index24、alternative=index19。两动作均只在首步强制，之后恢复相同的修复后TTC5 parent；原世界alternative更快成功，far twin同一alternative超时，G差分别+.1844411534和-.7791062408，不是仅千分之一幅度的翻转：

|root0世界|预设首动作|实际终局|全程时间(s)|G|
|---|---|---|---:|---:|
|原世界|A0=24|success|18.25|.4308727300|
|原世界|alternative=19|success|12.25|.6153138834|
|far omitted|A0=24|success|8.75|.7105532273|
|far omitted|alternative=19|timeout|50.25|-.0685530135|

此case同时给出结论边界的直接例子：A0在两世界都成功，故偏好反转并不代表没有共同成功动作。另一实际终局变化case为KDA443 root237/case90019：原世界A0 collision(G=-.5236701682)、alternative timeout(G=-.1769897061)，far世界二者均success(G=.5639051905/.5111844917)，偏好差+.3466804621→-.0527206987。以上case仅作原始结果展示，不替换16根全集或反转判据；包括GRU443 root691仅-.0009475的反向差在内，所有弱反转和无反转均保留。

科学裁决 SUPPORTED 的精确命题仅为：在这些人工合法reset世界及固定parent续策下，同一TTC5/root立即reward/filter输入对应的两个预设首动作，长期实现回报偏好可以反转。未穷举80动作的实际后果，不证明两个世界最优动作集合不相交，不能推出任意5人策略不可同时成功，也不把已测试两动作的较优者称为全动作最优。

它与 hidden-intent 实验不同：本实验全N可见输入不同，但far构造同时改变omitted人的可见初始位置及隐藏goals，不是“只改可见位置”的单因素归因。8/16不是自然失败比例，不能用原/far成功率差宣称截断导航损失，也不证明全N策略会利用差异或改善导航；仍以同世界冻结全N两臂实际结果判部署代价。失败/成功scope仅是原世界采样标签，不是far世界或修复后baseline结果。

future 残差、有限历史、在线采样策略/标签分布与值读出相互耦合。本轮只允许上述有限干预，不据解释合理性分配因果份额，不启动二轮训练以“补齐”机制。

## 【最主要的可修复机制】

当前可确定的维护项是在线 epsilon 真实历史追加与 terminal bootstrap mask，已由主独立修复和验收；两者并未建立独立导航方法。A3 的最小 N 兼容改动是表示能力开关，不自动是创新。A4 候选已 KILL。

不能从A2 residual或人工双世界宣称“最主要自然根因”。same-world全N确有救援、dup不能复现，但保护FAIL；paired5修复在线漏帧未改变终局，完整续策合同只在一个固定case改变终局。A2追加细分区分了物理V与reward，terminal在指定错排分母未贡献纠正。这些均不建立新算法，不用“有价值观察/奠基”替代方法级进展。

## 【下一步是否具备算法研发依据】

当前没有建立独立算法贡献，METHOD_SPEC_FREEZE=no；不启动完整 IL/RL 或第二轮受控训练。主已授权的两臂输入和 A5有限续策只用于完成第一轮归因，不复活失败的线性MC/pairwise候选。

本轮动作已结束：A2追加分解和count控制不授权新拟合/续跑；A3/A5原始复算包含实际分母、双口径碰撞/超时、共同成功效率、latency和case bootstrap，只汇总既有CPU线性读出。[只读汇总入口](a/summarize-a.py) 的5项CPU工程单测通过，独立count wrapper的3项CPU门通过，不算导航证据。研究主决策KILL，封存这些有限候选；没有独立方法差异与受保护导航收益，METHOD_SPEC_FREEZE=no。

### 最终验收表

执行验收PASS只表示对应数据/合同检查完成，不等于科学命题成立或候选GO。最终主决策仅KILL。

|验收项|实际范围与结果|原始证据 / 最终处置|
|---|---|---|
|A2严格三项分解|COMPLETE；恒等式最大误差2.22e-16，分母及独立root计数已列|[原pairs/summary](a/artifacts/a2-summary.json)；不是因果占比|
|A2 physical V / realReward / terminal细分|PASS 1,661对、23组；P/unmasked/F重建残差0，float32舍入差异单列；原产物未覆盖|[fine split及逐对索引](a/artifacts/a2-factor-split/summary.json)；COMPLETE，仅离线诊断|
|A3合法全N提取及N5合同|PASS 1,279roots/4model，原窗口/旧V重现及N5数值门保留|[extract示例manifest](a/artifacts/a-extract-kda-s419-t24/extract-done.json)；工程正确性非方法贡献|
|A3原head递归闭环及安全|PASS采集/复算80roots、160rows+8 A0重复；救援27/40、成功破坏4/40，新增环境碰撞4|[A3/A5逐根复核](a/artifacts/verified-a3-a5.json)；保护FAIL，KILL|
|count正式TTC baseline对原formal|PASS 80/80根完整steps/hash/outcome/time/G exact，冻结plan及源SHA一致|[count执行summary](a/artifacts/count-matched-complete/count-matched-summary.json)；无重选样|
|count TTC baseline对dup实际序列|PASS command 80/80、reward 80/80、physical_state_sha256 80/80精确相同，联合80/80；每model 20/20、每scope 10/10；最大abs(deltaG)=0|[独立count复核](a/artifacts/verified-count-independent.json)；完整诊断日志0/80相同属预期，不混称轨迹不同|
|count与allN竞争解释|COMPLETE；dup rescue/damage均0，allN−dup均值/描述性case区间已列|单纯重复数量不足以复现allN；不抵销allN原保护FAIL，不复活候选|
|A4等容量读出|COMPLETE；正式两seed联合gate FAIL；6个小5人split不足组保留INCOMPLETE|[原metrics](a/artifacts/readout-complete/readout-summary.json)；KILL当前线性MC/pairwise，不运行新head闭环|
|A5冻结四臂及有限CPU读出|PASS 4model×16root×4arm=256续跑；无early-terminal root；history修复未改变终局，合同切换1固定case终局变化|[原rows](a/artifacts/paired5-complete/paired5-rows.jsonl.gz)；不是完整学习因果，停止续策-only候选|
|完整A5学习因果 / 整个A根因归因|INCOMPLETE；未授权匹配IL/RL，完成范围仅既有有限干预、冻结续策采集与读出|不将有限采集COMPLETE冒作整个A5或根因归因完成；不追加实验|
|人工alias/hidden-intent与filter probe|COMPLETE；alias16roots/64ep、hidden-intent8roots/48ep、filter16roots/32ep，限制与强case已列|人工比例非自然归因；只比较预设动作对；bypass保护FAIL、KILL|
|训练/部署完整合同记录|PASS源码只读核验；raw34噪声、train mode、连续replay及terminal/timeout push均记录|未统一合同，噪声/dropout causality未检验；不追加实验|
|本轮执行边界|原collectors冻结，未修改原工程或b/engine；收尾未追加GPU实验，零完整IL/RL及第二轮训练|AUTHORIZED_EXECUTION_COMPLETE；完整学习因果/根因归因INCOMPLETE，没有方法级进展，停止|

可复制交付入口：

/home/abc/workspace/ab-causal-isolation-20261011/a/run-a.py

/home/abc/workspace/ab-causal-isolation-20261011/a/paired5-continuation.py

/home/abc/workspace/ab-causal-isolation-20261011/a/closedloop-a.py

/home/abc/workspace/ab-causal-isolation-20261011/a/artifacts/closedloop-heldout-caliper-plan.json

本报告状态：AUTHORIZED_EXECUTION_COMPLETE，完整A5学习因果及整个A根因归因INCOMPLETE（未授权匹配IL/RL）。已完成仅指本轮获授权的A2离线细分、A3有限干预、A4读出、A5有限冻结续策采集与读出、人工alias/filter probe及dup控制，不指整个A5学习任务或完整根因归因。未执行的完整IL/RL、新读出head闭环和第二轮训练保持未授权/不执行；A4六个小5人分组因split不足保留INCOMPLETE而不冒称有效训练臂。没有方法级进展，主决策KILL，A本轮停止。
