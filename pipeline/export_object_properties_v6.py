"""AI-readable candidates with per-object attributes and visible observation history."""
import argparse,json
from pathlib import Path
from ocr_routing import read_jsonl
from object_evidence_v6 import summarize_record

LABELS={'window_kind':'前景窗口身份','checkerboard':'密集棋盘格','pieces':'散落碎片','hand':'手部形状',
        'blue_circle':'蓝色圆形效果','godot_logo':'Godot启动页标志','solid_triangles':'实心深色三角形','first_checked':'菜单第一项勾选'}
VALUES={'present':'可见','absent':'未见','unknown':'无法确定','settings':'项目设置窗口','running_game':'游戏运行窗口','editor_only':'编辑器本体'}
SCOPES={'foreground_window':'前景窗口','demonstration_content':'演示内容','menu_first_item':'菜单第一项'}


def run(root,out):
    tasks={t['task_id']:t for t in read_jsonl(root/'sources/tasks.public.jsonl')}
    records=read_jsonl(out/'records.jsonl')
    lines=['# Godot 局部视觉属性候选\n',
           '视频：BV1cyHR67EbJ。领域：游戏开发 / Godot / 视觉证据。状态：待源复核，禁止自动当作已核准知识。\n',
           '只覆盖现有实验的24个采样位置。对象范围由任务预先给定，不代表自动发现对象或完成跨帧追踪。时间为源视频 PTS。\n',
           '原始观察来自原生单图；时序单帧是对照；回看只追加观察。多次一致不自动确认，存在分歧也不改写原始记录。可见/未见仅针对当前画面，不推断点击、过程结果或因果。\n']
    for record in records:
        initial=record['initial'];task=tasks[initial['task_id']];summary=summarize_record(record)
        temporal=next(x['observation'] for x in record['additional'] if x['purpose']=='encoding_control')
        rereads=[x['observation'] for x in record['additional'] if x['purpose']=='targeted_reread']
        lines += [f"## {task['case_id']} / 位置{task['position']} / {task['source']['time']:g}秒 / {SCOPES[task['object_scope']]}\n",
                  '| 属性 | 原始单图观察 | 时序输入对照 | 补读观察 | 记录状态 |', '|---|---|---|---|---|']
        for key,value in initial['values'].items():
            status=summary['properties'][key]['status']
            status_text='存在分歧，待复核' if status=='unreviewed_conflict' else '新增候选，初值不变' if status=='unreviewed_additional_candidate' else '无法确定' if status=='unknown' else '候选，未复核'
            additions='；'.join(VALUES[r['values'][key]] for r in rereads) or '未补读'
            lines.append(f"| {LABELS[key]} | {VALUES[value]} | {VALUES[temporal['values'][key]]} | {additions} | {status_text} |")
        lines += [f"\n[原始画面](<{task['source']['crop']}>) · [原始单图回执](<{initial['receipt']}>) · [时序回执](<{temporal['receipt']}>)\n"]
        for r in rereads:lines.append(f"[补读回执](<{r['receipt']}>) · [实际补读图](<{r['view']['image_path']}>)\n")
    directory=out/'knowledge/Godot';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'BV1cyHR67EbJ.局部视觉属性候选.md';path.write_text('\n'.join(lines)+'\n')
    (out/'knowledge/README.md').write_text('# v6 候选知识\n\n[Godot 局部视觉属性候选](Godot/BV1cyHR67EbJ.局部视觉属性候选.md)\n\n所有条目保持待复核状态，不等于全片知识整理完成。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.out.resolve())
