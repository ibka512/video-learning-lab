"""Readable candidate records, deliberately without semantic promotion."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ocr_routing import read_jsonl
from visual_ledger_v5 import effective_state

TITLES = {'W':'前景窗口切换','B':'运行区域棋盘格','F':'背景碎裂与手部内容','T':'启动与三角形效果','M':'嵌入选项勾选','C':'背景碎裂与蓝色圆形'}


def run(root,out):
    specs={s['case_id']:s for s in read_jsonl(root/'sources/tasks.public.jsonl')}
    cases={s['case_id']:s for s in read_jsonl(root/'sources/cases.public.jsonl')}
    results=read_jsonl(out/'results.jsonl')
    directory=out/'knowledge/Godot/视觉状态候选'; directory.mkdir(parents=True,exist_ok=True)
    index=['# Godot 视觉状态候选\n','这些文件是开发实验候选，不是已经核准的教程知识。画面选择仍由 Codex 完成；没有全片操作或因果验证。\n',
           '验收结论：D 回看策略发生退化，不作为默认输出。本页先展示 C 的单图候选，所有流程仍需源复核，不能直接用于自动知识入库。\n']
    for cid,title in TITLES.items():
        spec=specs[cid]; lines=[f'# {title}\n','状态：待复核。主表为 C 单图候选，未由 Codex 改成参考答案。D 回看未通过默认启用验收，只保留作诊断对照。\n',
            '时间取自原视频 PTS。位置是本组呈现顺序；状态变化只定位到相邻样本之间，不能据此编造点击时刻。\n']
        for arm in ['C','D','B']:
            result=next(r for r in results if r['case_id']==cid and r['arm']==arm and r['mode']=='visual_only')
            lines += [f'## {arm} 流程\n',result['description']+'\n',
                      '| 位置 | 原片时间 | 可输出状态 | 模型原始状态 | 来源 |',
                      '|---|---|---|---|---|']
            for row in result['frames']:
                source=row['source']; state=effective_state(row); raw=row['observation']['state']
                lines.append(f"| {row['position']} | {source['time']:g} 秒 | {spec['labels'][state]} | {spec['labels'][raw]} | [原画面](<{source['crop']}>) · [原始回执](<{row['receipt']}>) |")
            lines += ['\n程序推导的旧题答案：`'+json.dumps(result['answers'],ensure_ascii=False)+'`\n',
                      '这些答案只适用于本轮公开题目，程序适配规则不代表模型具备通用过程推理能力。\n']
            for row in result['frames']:
                if row.get('review'):
                    lines.append(f"- 位置 {row['position']} 回看：{row['review']['status']}；{row['review']['stop_reason']}；补读 {len(row['review']['attempts'])} 次。")
        lines += ['\n## 讲解来源（未升级成执行事实）\n']
        lines += [f"- {s['ref']}：{s['text']}" for s in cases[cid]['transcript']]
        lines += ['\n## 证据预览\n']
        for i,src in enumerate(cases[cid]['frames']):
            lines += [f"位置 {i}，原片 {src['time']:g} 秒。\n",f"![源画面 {i}](<{src['crop']}>)\n"]
        name=f'{cid}-{title}.md'; (directory/name).write_text('\n'.join(lines)+'\n')
        index.append(f'- [{title}](Godot/视觉状态候选/{name})')
    (out/'knowledge/README.md').write_text('\n'.join(index)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.root.resolve(),a.out.resolve())
