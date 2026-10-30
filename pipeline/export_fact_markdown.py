"""Render supported quotations; explicit source reviews are separate from model output."""
from pathlib import Path
import argparse,json,os,shutil
from fact_ledger import hash_object,evaluate_response
from ocr_routing import read_jsonl,dump,digest,write_jsonl
LABELS={'setting_key':'设置键','help_default':'帮助默认值（不是当前状态）','help_statement':'界面帮助正文','author_statement':'作者讲解原文','observed_state':'已复核的单帧状态'}

def reviewed_facts(facts,reviews):
    approved=[];pending=[]
    for f in facts:
        d=reviews.get(f['fact_id'])
        if d and d['fact_sha256']!=hash_object(f):raise ValueError('review belongs to different fact')
        if d and d['status']=='reject':pending.append({**f,'export_block_reason':d['reason']});continue
        if f['review_status']=='source_supported_quote' or (d and d['status']=='approve'):
            approved.append({**f,'export_status':'source_reviewed_quote','export_review_basis':d['reason'] if d else 'existing source-reviewed quotation; not independent human review'})
        else:pending.append({**f,'export_block_reason':'needs explicit source review'})
    return approved,pending

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();r=a.root.resolve();m=json.loads((r/'sources/input-manifest.json').read_text());facts=read_jsonl(r/'sources/facts.completed.jsonl');reviewpath=r/'sources/fact-source-reviews.jsonl';reviews={d['fact_id']:d for d in read_jsonl(reviewpath)} if reviewpath.exists() else {}
    for f in facts:
        packetpath=r/'packets'/(f['packet_id']+'.json');assert digest(packetpath)==m['packet_files'][f['packet_id']];packet=json.loads(packetpath.read_text());v,invalid,_=evaluate_response({'facts':[{k:f[k] for k in ('kind','value','citations')}],'overflow':False},packet)
        if invalid or not v or v[0]['fact_id']!=f['fact_id']:raise ValueError('invalid fact export')
        for e in packet['evidence']:
            if e.get('source_image_sha256'):assert digest(Path(e['source_image']))==e['source_image_sha256']
    approved,pending=reviewed_facts(facts,reviews);write_jsonl(r/'sources/facts.exported.jsonl',approved);write_jsonl(r/'sources/facts.pending.jsonl',pending)
    domain=Path(m['domain'])
    if domain.is_absolute() or '..' in domain.parts:raise ValueError('invalid domain path')
    target=r/'knowledge'/domain/(m['source_identity']['bvid']+'.md');target.parent.mkdir(parents=True,exist_ok=True)
    lines=['---','video_id: '+m['source_identity']['bvid'],'domain: '+m['domain'],'status: source_grounded_partial_notes','complete_video_knowledge: false','---','', '# '+m['title'],'',f"来源：[{m['source_identity']['bvid']}]({m['source_url']})。本稿采用可追溯的事实字段与原文引文，不补写未出现的操作或代码。",'','当前状态只属于列出的实际帧；默认值来自界面帮助。讲解保留为原文附录，不等于已经提炼出的知识点。','','## 设置、帮助和画面状态','']
    def render(f):
        loc=f['source_locations'];ts=', '.join(f"{l['start']:.2f}–{l['end']:.2f}s" for l in loc);out=[f"### {LABELS[f['kind']]} · {ts}",'']
        if f.get('subject'):out += ['对象：`'+f['subject']+'`。','']
        out+=['> '+f['value'].replace('\n','\n> '),'',f"事实编号：`{f['fact_id']}`。来源方式：`{f['origin']}`。"]
        for l in loc:
            if l.get('source_image'):
                copy=r/'evidence'/Path(l['source_image']).name;copy.parent.mkdir(exist_ok=True)
                if copy.exists():assert digest(copy)==l['source_image_sha256']
                else:shutil.copyfile(l['source_image'],copy)
                rel=os.path.relpath(copy,target.parent);out.append(f"源帧：[{l['frame_id']}](<{rel}>)；位置：`{l['bbox']}`。")
        out += ['证据：'+', '.join('`'+x['ref']+'`' for x in f['source_spans'])+'。',''];return out
    for f in sorted((f for f in approved if f['kind']!='author_statement'),key=lambda f:(min(x['start'] for x in f['source_locations']),f['subject'] or '',f['kind'])):lines+=render(f)
    lines+=['## 作者讲解原文附录','','以下保留已有校正转写的时间和措辞，不据此证明操作已经执行。','']
    for f in sorted((f for f in approved if f['kind']=='author_statement'),key=lambda f:f['source_locations'][0]['start']):lines+=render(f)
    lines+=['## 尚未确认的材料','',f'{len(pending)} 条记录未进入上面的已复核内容；详情见结果包的 sources/facts.pending.jsonl。','', '本稿不代表全片知识已完整提取，也没有独立人工验证或新视频泛化成绩。']
    target.write_text('\n'.join(lines)+'\n');dump(r/'sources/export-summary.json',{'path':str(target),'supported_fact_records':sum(f['kind']!='author_statement' for f in approved),'transcript_quotations':sum(f['kind']=='author_statement' for f in approved),'pending':len(pending),'full_video_knowledge_complete':False,'export_code_sha256':digest(Path(__file__))});print(target)
if __name__=='__main__':main()
