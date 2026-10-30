"""Keep supplemental text inference and complete-sentence results separate."""
import argparse,json
from pathlib import Path
from fact_ledger import hash_object
from ocr_routing import dump,digest,read_jsonl,write_jsonl
from run_semantic_groups_v4 import sentence_groups


def evaluate(root):
    gold=json.loads((root/'references/visual-gold.json').read_text())
    public=read_jsonl(root/'sources/cases.public.jsonl')
    control=root/'transcript-control';protocol=json.loads((control/'protocol.json').read_text())
    if protocol['input_sha256']!=digest(root/'sources/cases.public.jsonl'):raise ValueError('changed text input')
    rows=[];seconds=0
    for c in public:
        record=json.loads((control/(c['case_id']+'.json')).read_text());seconds+=record['wall_seconds']
        assert record['protocol_sha256']==hash_object(protocol)
        assert record['input']=={'questions':c['questions'],'transcript':c['transcript']}
        target=next(g for g in gold['cases'] if g['case_id']==c['case_id'])
        for q in target['questions']:
            a=record['output']['answers'][q['question_id']] if record['output'] else None
            rows.append({'case_id':c['case_id'],'question_id':q['question_id'],'answer':a,
                'content_match':bool(a and a['value']==q['answer']),
                'unproven_cancellation_claim':bool(c['case_id']=='M' and q['question_id']=='change' and a and a['value']=='cancel_observed'),
                'basis_inconsistent':bool(a and a['value']=='unknown' and a['basis']!='unknown')})
    summary={'questions':len(rows),'unknown':sum(r['answer'] is not None and r['answer']['value']=='unknown' for r in rows),
        'content_match':sum(r['content_match'] for r in rows),'unproven_cancellation_claims':sum(r['unproven_cancellation_claim'] for r in rows),
        'basis_inconsistent':sum(r['basis_inconsistent'] for r in rows),'seconds':seconds,
        'scope':'post-hoc diagnostic, permits transcript inference; cannot establish visual grounding'}
    write_jsonl(control/'scores.jsonl',rows);dump(control/'evaluation.json',summary)
    packs=read_jsonl(root/'sources/semantic.public.jsonl');ts=json.loads((root/'references/semantic-targets.json').read_text())
    groupdir=root/'sentence-groups';plan=json.loads((groupdir/'protocol.json').read_text())
    assert plan['input_sha256']==digest(root/'sources/semantic.public.jsonl')
    fs=[];elapsed=0;groupcount=0
    for p in packs:
        rec=json.loads((groupdir/(p['packet_id']+'.json')).read_text());elapsed+=rec['wall_seconds']
        assert rec['protocol_sha256']==hash_object(plan)
        units=sentence_groups(p);assert rec['source_group_mapping']==units
        assert ''.join(u['text'] for u in units)==''.join(s['text'] for s in p['speech'])
        groupcount+=len(units);by={u['ref']:u for u in units}
        if not rec['output']:continue
        for kind,refs in rec['output']['fields'].items():
            if not refs:continue
            segs={s for r in refs for s in by[r]['segment_refs']}
            us=[u for u in p['units'] if u['segment_ref'] in segs]
            fs.append({'subject':p['subject'],'kind':kind,'refs':refs,'quote':'\n'.join(u['text'] for u in units if u['ref'] in refs),
                'unit_ids':[u['unit_id'] for u in us],'source_start':min(u['start'] for u in us),'source_end':max(u['end'] for u in us),
                'status':'candidate_not_source_approved'})
    policy={'term_identity':{'definition','usage','behavior','limitation','comparison','analogy'},'definition':{'definition','behavior'},'limitation':{'limitation'},'comparison':{'comparison'},'usage':{'usage'}}
    norm=lambda s:''.join(s.split()).casefold();scores=[]
    for t in ts['targets']:
        a,b=t['window'];term=t['id'].split('-')[0].casefold()
        related=[f for f in fs if term in f['subject'].casefold() and f['source_start']<b and f['source_end']>a and f['kind'] in policy[t['kind']]]
        value=norm(' '.join(f['subject']+' '+f['quote'] for f in related))
        missing=[s for s in t['all_text'] if norm(s) not in value]
        scores.append({'target':t['id'],'typed_wording_retained':not missing,'missing':missing})
    sem={'source_groups':groupcount,'candidate_fields':len(fs),'target_count':len(scores),
        'typed_wording_retained':sum(s['typed_wording_retained'] for s in scores),'seconds':elapsed,
        'scope':'post-hoc source sentence-end grouping, same topic material and field task; groups may contain multiple knowledge types',
        'independent_human_accuracy':False}
    write_jsonl(groupdir/'candidates.jsonl',fs);write_jsonl(groupdir/'scores.jsonl',scores);dump(groupdir/'evaluation.json',sem)
    print(json.dumps({'transcript_control':summary,'sentence_groups':sem},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);evaluate(p.parse_args().root.resolve())
