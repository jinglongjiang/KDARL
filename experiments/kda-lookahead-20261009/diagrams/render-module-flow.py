"""Render the inspected KDA-VL module flow without importing the project."""
from pathlib import Path
import io
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
WIDTH = 2400
PAD = 72
COMMON = '''
graph [rankdir=LR, bgcolor="transparent", pad=0.12, nodesep=0.28,
       ranksep=0.38, splines=polyline, fontname="Noto Sans CJK SC"];
node [shape=box, style=filled, fillcolor="#F2F5F7", color="#A4B0B8",
      penwidth=1.2, margin="0.16,0.12", fontname="Noto Sans CJK SC", fontsize=19];
edge [color="#687780", arrowsize=0.7, fontname="Noto Sans CJK SC", fontsize=15];
'''

PANELS = [
    ('01  训练链路：IL 预训练 + 在线 Monte Carlo 价值回归',
     '两阶段优化同一个 Vθ；IL 模仿回报，不是动作分类；在线更新不是 PPO / TD bootstrap。',
     '''
digraph training {
COMMON
teacher [label="ORCA 示范\n5 人 circle", fillcolor="#E3F2E8", color="#6A9B7B"];
il_data [label="示范状态 + reward\n构建 24 帧 token 窗口"];
il_target [label="完整轨迹 MC target\nGₜ = rₜ + 0.99 Gₜ₊₁"];
il_loss [label="IL 价值预训练\nMSE(Vθ(Hₜ), Gₜ)", fillcolor="#E3F2E8", color="#6A9B7B"];
init [label="IL checkpoint\n初始化在线策略"];
teacher -> il_data -> il_target -> il_loss -> init;
rollout [label="原策略 + ε-greedy\n5 人 circle 闭环采样", fillcolor="#E4EEF8", color="#7093B4"];
trajectory [label="真实 episode\n状态 / 动作 / reward"];
buffer [label="ReplayBufferMC\n历史窗口 + MC return"];
rl_loss [label="在线 MC 回归\nMSE(Vθ(Hₜ), Gₜ)", fillcolor="#E4EEF8", color="#7093B4"];
final [label="RL checkpoint\n冻结后评估 5 / 10 / 20 人"];
rollout -> trajectory -> buffer -> rl_loss -> final;
init -> rollout [constraint=false, style=dashed, label="warm start"];
rl_loss -> rollout [constraint=false, style=dashed, color="#397493", label="更新同一个 Vθ"];
{rank=same; teacher; rollout;}
{rank=same; il_data; trajectory;}
{rank=same; il_target; buffer;}
{rank=same; il_loss; rl_loss;}
{rank=same; init; final;}
}
'''),
    ('02  模型内部：scene-level KDA temporal value backbone',
     'Mamba / GRU 对照只替换时序骨干；空间编码与 scalar Value Head 结构相同，不代表训练后权重相同。',
     '''
digraph model {
COMMON
input [label="历史 token 窗口\n[B, 24, 8, 13]"];
robot [label="机器人 token：第 0 行\n13 → 64\nLinear / ReLU / LayerNorm"];
relation [label="行人 token：第 3–7 行\n13 维 + 8 维关系特征\n五人内部按距离重排"];
human [label="行人特征 MLP\n21 → 64\nLinear / ReLU / LayerNorm"];
attention [label="行人 self-attention\n4 heads\n五人 max pooling → 64"];
fusion [label="拼接机器人 + 人群\n64 + 64 = 128\nDropout / fusion MLP\n→ [B, 24, 256]", fillcolor="#E7F2F0", color="#5D958D"];
kda [label="4 × KDA block\nFLA KimiDeltaAttention\n4 heads × 64\nx ← x + LayerNorm(KDA(x))", fillcolor="#CFE9E5", color="#347C72", penwidth=2];
last [label="取最后时刻特征\nz[:, -1, :]\n[B, 256]"];
value [label="Scalar Value Head\nLinear(256, 1)\nVθ(H) ∈ R", fillcolor="#CFE9E5", color="#347C72", penwidth=2];
input -> robot -> fusion;
input -> relation -> human -> attention -> fusion;
fusion -> kda -> last -> value;
}
'''),
    ('03  在线决策：80 候选的一步 value lookahead → 安全处理 → 执行',
     '当前诊断基线为外部脚本 TTC5；仓库默认为 first5。H1/H2/H3 是冻结权重诊断，不属于训练架构。',
     '''
digraph decision {
COMMON
state [label="真实当前状态 sₜ\n机器人 + 全部 N 个行人", fillcolor="#E4EEF8", color="#7093B4"];
select [label="当前状态锚定选 5 人\n本轮 TTC5 / 仓库 first5\n全部候选共享相同索引"];
history [label="当前 token + 真实历史\n只读前缀 / 左侧重复填充"];
grid [label="原生 80 grid 动作\n5 速度 × 16 方向\nH2：末尾另加真停止"];
smooth [label="候选执行 command\nα·上一动作 + (1−α)·grid\nH1：α = 0.3 / 0"];
successor [label="单步后继：Δt = 0.25 s\n机器人：按候选 command 推进\n全部行人：CV 推进一次\nH3：同状态真实后继替换"];
windows [label="每个候选的假想窗口\n[80, 24, 8, 13]\n仅最后一帧因候选而异"];
network [label="同一个价值网络 Vθ\n调用上方第 02 部分\n输出 80 个 scalar value", fillcolor="#CFE9E5", color="#347C72"];
raw [label="原始候选评分\nr(sₜ,aᵢ) + 0.99 Vθ(Hⁱₜ₊₁)"];
safety [label="安全 mask + 风险罚项\nclearance ≥ 0.2 m\n风险系数 0.8\n保留 index 0 的 −0.001", fillcolor="#F9E9E8", color="#BE8480"];
execute [label="argmax → 选定 command\n直接执行同一 command\n追加真实当前 token", fillcolor="#E4EEF8", color="#7093B4"];
geometry [label="全部 N 人的后继几何\n计算 reward / clearance\n不限于价值网络的 5 人", fillcolor="#F9E9E8", color="#BE8480"];
state -> select -> history -> windows;
state -> grid -> smooth -> successor -> windows;
successor -> geometry;
select -> windows [style=dashed, label="同一选择集合", constraint=false];
windows -> network -> raw -> safety -> execute;
geometry -> raw [label="reward"];
geometry -> safety [label="clearance"];
execute -> state [constraint=false, style=dashed, color="#397493", label="环境反馈下一帧"];
}
'''),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    namespace = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', namespace)
    panels = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, (title, subtitle, dot) in enumerate(PANELS):
            dot = dot.replace('COMMON', COMMON)
            source = Path(tmp) / ('panel%d.dot' % index)
            source.write_text(dot, encoding='utf-8')
            svg = subprocess.run(['dot', '-Tsvg', str(source)], check=True, capture_output=True).stdout
            png = subprocess.run(['dot', '-Gdpi=160', '-Tpng', str(source)], check=True, capture_output=True).stdout
            root = ET.fromstring(svg)
            box = [float(x) for x in root.attrib['viewBox'].split()]
            panels.append((title, subtitle, root, box, Image.open(io.BytesIO(png)).convert('RGBA')))
    scale = (WIDTH - 2 * PAD) / max(p[3][2] for p in panels)
    heights = [round(p[3][3] * scale) for p in panels]
    total = 160 + sum(h + 112 for h in heights) + 180
    canvas = Image.new('RGB', (WIDTH, total), 'white')
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(BOLD, 39)
    section_font = ImageFont.truetype(BOLD, 29)
    body_font = ImageFont.truetype(FONT, 23)
    root = ET.Element('{%s}svg' % namespace, width=str(WIDTH), height=str(total),
                      viewBox='0 0 %d %d' % (WIDTH, total))
    ET.SubElement(root, '{%s}rect' % namespace, width=str(WIDTH), height=str(total), fill='white')

    def text(x, y, label, size, bold=False):
        node = ET.SubElement(root, '{%s}text' % namespace, x=str(x), y=str(y + size),
            fill='#20343E', attrib={'font-family': 'Noto Sans CJK SC', 'font-size': str(size),
                                   'font-weight': '700' if bold else '400'})
        node.text = label
        draw.text((x, y), label, fill='#20343E', font=title_font if size == 39 else section_font if bold else body_font)

    text(PAD, 28, 'KDA-VL 项目模块流向图', 39, True)
    text(PAD, 87, '当前源码事实版  |  训练：5 人 circle  |  T = 24，D = 256  |  不代表第一篇投稿版本', 23)
    y = 155
    for index, ((title, subtitle, panel, box, bitmap), height) in enumerate(zip(panels, heights)):
        text(PAD, y, title, 29, True)
        text(PAD, y + 45, subtitle, 23)
        top = y + 83
        width = round(box[2] * scale)
        x = (WIDTH - width) // 2
        resized = bitmap.resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, top), resized)
        group = ET.SubElement(root, '{%s}g' % namespace,
            transform='translate(%s,%s) scale(%s)' % (x, top, scale))
        for child in panel:
            if child.tag.endswith('g'):
                for element in child.iter():
                    if 'id' in element.attrib:
                        element.attrib['id'] = 'panel%d-' % index + element.attrib['id']
                group.append(child)
        y = top + height + 29
        if index < len(panels) - 1:
            draw.line((PAD, y - 12, WIDTH - PAD, y - 12), fill='#D8E0E4', width=2)
            ET.SubElement(root, '{%s}line' % namespace, x1=str(PAD), x2=str(WIDTH - PAD),
                y1=str(y - 12), y2=str(y - 12), stroke='#D8E0E4', attrib={'stroke-width': '2'})
    text(PAD, y + 10, '非主链：q_head / future_pred_head 虽有定义，但当前 SARL scalar-value 训练与选动作不走这些头。', 23)
    text(PAD, y + 48, '真实 history 只写入已发生的当前帧；候选后继不写回。评分与执行使用同一 command。', 23)
    text(PAD, y + 86, 'KDA 每个窗口独立计算，无跨窗口 cache；它输出 scalar value，不直接生成动作。', 23)
    text(PAD, y + 124, '空间编码不读取独立目标/运动 token 行；目标信息仍包含在机器人 token 中。', 23)
    svg_path = OUT / 'kda-vl-module-flow.svg'
    png_path = OUT / 'kda-vl-module-flow.png'
    ET.ElementTree(root).write(svg_path, encoding='utf-8', xml_declaration=True)
    canvas.save(png_path)
    assert svg_path.stat().st_size > 0 and png_path.stat().st_size > 0
    print(json.dumps({'status': 'rendered', 'width': WIDTH, 'height': total,
        'files': {str(p): p.stat().st_size for p in (png_path, svg_path)}}))


if __name__ == '__main__':
    main()
