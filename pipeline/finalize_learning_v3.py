"""Explicit, digest-bound source corrections; model output remains immutable."""
import argparse,json
from pathlib import Path
from fact_ledger import hash_object
from ocr_routing import dump,read_jsonl,write_jsonl
from run_learning_v3 import load_validated,LABELS,load_legacy
from learning_schema import ROLES,build_steps,audit_procedure


def finalize(root):
    m,ps,cs,raw=load_validated(root)
    reviews=read_jsonl(root/'sources/source-reviews.jsonl')
    by={f['fact_id']:f for f in raw}
    rs={r['fact_id']:r for r in reviews}
    if len(rs)!=len(reviews) or set(rs)!=set(by):raise ValueError('review every source unit exactly once')
    approved=[]
    for fid,f in by.items():
        r=rs[fid]
        if r['fact_sha256']!=hash_object(f):raise ValueError('stale source review')
        if r['status']=='approve':approved.append(f)
        elif r['status']!='reject':raise ValueError('unknown review status')
    cp=root/'sources/source-corrections.jsonl'
    corrections=read_jsonl(cp) if cp.exists() else []
    seen=set()
    for c in corrections:
        fid=c['fact_id']
        if fid not in by or fid in seen or rs[fid]['status']!='reject':raise ValueError('correction needs rejected original fact')
        seen.add(fid);f=by[fid]
        if c['fact_sha256']!=hash_object(f) or c['status']!='approve':raise ValueError('stale/unapproved correction')
        if c['corrected_role'] not in ROLES or c['corrected_role']==f['role']:raise ValueError('invalid role correction')
        core={**f,'role':c['corrected_role'],'origin':'Codex_source_review_role_correction',
              'original_fact_id':fid,'original_fact_sha256':hash_object(f),'correction_reason':c['reason']}
        core.pop('fact_id')
        approved.append({**core,'fact_id':'R'+hash_object(core)[:18]})
    order={f['unit_id']:i for i,f in enumerate(raw)}
    approved.sort(key=lambda f:order[f['unit_id']])
    reviewgraphs=read_jsonl(root/'sources/procedure-corrections.jsonl') if (root/'sources/procedure-corrections.jsonl').exists() else []
    graphs=[];cardby={c['packet_id']:c for c in cs};packetby={p['packet_id']:p for p in ps};graphseen=set()
    for g in reviewgraphs:
        pid=g['packet_id'];c=cardby[pid];p=packetby[pid]
        if pid in graphseen:raise ValueError('duplicate graph review')
        graphseen.add(pid)
        if g['original_card_sha256']!=hash_object(c) or g['status']!='approve':raise ValueError('stale/unapproved graph correction')
        rows=[{'unit_id':f['unit_id'],'role':f['role']} for f in approved if f['packet_id']==pid]
        if {r['unit_id'] for r in rows}!={u['unit_id'] for u in p['units']}:raise ValueError('procedure requires full source-unit accounting')
        procedure=build_steps(g['procedure'],p['units'],rows)
        alarms=audit_procedure(p['units'],rows,procedure)
        if alarms:raise ValueError('reviewed procedure still has alarms: '+str(alarms))
        graphs.append({'packet_id':pid,'procedure':procedure,'origin':'Codex_source_review_graph_correction',
                       'original_card_sha256':hash_object(c),'reason':g['reason'],
                       'after_semantics':'explicit source-reviewed prerequisite; conditional actions only execute when their condition holds',
                       'observed_execution':False})
    write_jsonl(root/'sources/units.reviewed.jsonl',approved)
    write_jsonl(root/'sources/procedures.reviewed.jsonl',graphs)
    gp={g['packet_id']:g for g in graphs}
    target=root/'knowledge'/m['domain']/(m['source_identity']['bvid']+'.reviewed.md')
    target.parent.mkdir(parents=True,exist_ok=True)
    lines=['---','schema_version: 3','complete_video_knowledge: false',
           'review_basis: Codex_source_review_not_independent_human','execution_verified: false','---','',
           '# '+m['title'],'',f"来源：[原视频]({m['source_url']})。",'',
           '本稿保留作者原话及其来源，按知识角色和操作流程组织。模型原始输出、源复核修正分别留档。',
           '短句时间沿用所属语音段，未进行短句级重新对齐。',
           '转写依据：'+m['transcript_review_basis'],'']
    for p in ps:
        fs=[f for f in approved if f['packet_id']==p['packet_id']]
        if not any(f['role'] not in ['overview','housekeeping'] for f in fs):continue
        es={u['unit_id']:u for u in p['units']}
        def q(r):
            u=es[r]
            return f"{u['quote']}〔{r}；{u['start']:.2f}–{u['end']:.2f}s〕"
        lines+=['## '+p['subject'],'']
        if p['packet_id'] in gp:
            lines+=['### 操作流程（作者讲解）','','条件不满足时跳过对应分支；作者描述的结果不代表本轮实际运行验证。','']
            for i,s in enumerate(gp[p['packet_id']]['procedure']['steps']):
                lines += [f"{i+1}. "+' '.join(q(r) for r in s['action_refs'])]
                for k,label in [('context_refs','动作上下文'),('condition_refs','执行条件'),('result_refs','作者预期结果')]:
                    if s[k]:lines+=['    '+label+'：'+' '.join(q(r) for r in s[k])]
                lines+=['    前置步骤：'+(', '.join(str(n+1) for n in s['after']) or '无')]
            lines.append('')
        for role in ROLES:
            if role in ['action','overview','housekeeping','unresolved']:continue
            fields=[f for f in fs if f['role']==role]
            if fields:
                lines+=['### '+LABELS[role],'']+['- '+q(f['unit_id']) for f in fields]+['']
        if any(f['role']=='action' for f in fs) and p['packet_id'] not in gp:
            lines+=['### 独立动作片段（未组成完整流程）','']+['- '+q(f['unit_id']) for f in fs if f['role']=='action']+['']
        relevant={f['segment_ref'] for f in fs if f['role'] not in ['overview','housekeeping']}
        lines+=['### 原文上下文（按讲解顺序）','']
        for e in p['speech']:
            if e['ref'] in relevant:lines+=[f"- {e['text']}〔{e['ref']}；{e['start']:.2f}–{e['end']:.2f}s〕"]
        lines.append('')
    legacy=load_legacy(m)
    if legacy:
        lines+=['## 既有画面设置与状态记录','','沿用上一版源复核结果，未计为本轮模型新抽取。默认值与单帧状态分别记录。','']
        for f in legacy:lines+=[f"- `{f['subject']}` · `{f['kind']}`：{f['value']}"]
        lines.append('')
    lines+=['## 待处理范围','','本轮依据已有校正语音与既有画面事实，未新增逐帧动作理解，也未在目标软件中复现操作。',
            '视频未讲清的实现细节需要补充材料；本稿不根据概述推导代码、参数或具体操作。',
            '数量建议保留为作者建议；比喻不等同组件之间的技术差异。']
    target.write_text('\n'.join(lines)+'\n')
    summary={'raw_units':len(raw),'raw_approved':sum(r['status']=='approve' for r in reviews),
             'raw_rejected':sum(r['status']=='reject' for r in reviews),'explicit_role_corrections':len(corrections),
             'reviewed_units':len(approved),'reviewed_procedures':len(graphs),
             'formal_procedure_steps':sum(len(g['procedure']['steps']) for g in graphs),
             'model_automatically_produced_reviewed_graph':False if graphs else None,
             'markdown':str(target),'complete_video_knowledge':False}
    dump(root/'sources/finalized-summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    finalize(p.parse_args().root.resolve())
