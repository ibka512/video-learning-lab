"""Offline evaluator. This is the only v5 pipeline entry that reads frozen references."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from ocr_routing import digest, dump, read_jsonl, write_jsonl
from visual_ledger_v5 import effective_state, derive_answers, render_description
from evaluate_video_understanding_v4 import expected_support


def run(root, out, old):
    freeze = json.loads((root/'references/freeze.json').read_text())
    for name,sha in freeze.items():
        if digest(root/name) != sha:
            raise ValueError('frozen file changed: '+name)
    frame_gold = {c['case_id']:[f['state'] for f in c['frames']] for c in json.loads((root/'references/frame-states.json').read_text())['cases']}
    questions = {c['case_id']:c['questions'] for c in json.loads((root/'references/v4-questions.json').read_text())['cases']}
    cases = {c['case_id']:c for c in read_jsonl(root/'sources/cases.public.jsonl')}
    specs = {c['case_id']:c for c in read_jsonl(root/'sources/tasks.public.jsonl')}
    results = read_jsonl(out/'results.jsonl')
    states = []; question_scores = []
    for result in results:
        cid = result['case_id']; reverse = result['mode']=='reversed'
        expected_sources = cases[cid]['frames'][::-1] if reverse else cases[cid]['frames']
        ref = frame_gold[cid][::-1] if reverse else frame_gold[cid]
        if result['answers'] != derive_answers(result['frames'],specs[cid]):
            raise ValueError('answers not derived from ledger')
        if result['description'] != render_description(result['frames'],specs[cid]):
            raise ValueError('description not derived from ledger')
        for row in result['frames']:
            p = row['position']; src = expected_sources[p]
            if row['source'] != src:
                raise ValueError('source binding mismatch')
            state = effective_state(row)
            receipt = json.loads(Path(row['receipt']).read_text())
            # Inspect the original text independently of structural acceptance. A
            # rejected object reference must not erase an otherwise readable state.
            try:
                raw_data = json.loads(receipt['raw_output'])
                raw = raw_data[f"position_{row['view']['position']}"]['state']
                if raw not in specs[cid]['states']:
                    raw = 'unknown'
            except (KeyError,TypeError,ValueError):
                raw = 'unknown'
            states.append({'arm':result['arm'],'mode':result['mode'],'case_id':cid,'position':p,'reference':ref[p],
                           'raw_state':raw,'state':state,'raw_correct':raw==ref[p], 'correct':state==ref[p],
                           'wrong':state not in ['unknown',ref[p]],'unknown':state=='unknown',
                           'record_valid':receipt['output'] is not None,
                           'issues':row['issues'],'receipt':row['receipt']})
        for q in questions[cid]:
            a = result['answers'][q['question_id']]
            ref_answer = q['reversed_answer'] if reverse else q['answer']
            necessary = expected_support(q,reverse)
            ps = set(a['positions'])
            question_scores.append({'arm':result['arm'],'mode':result['mode'],'case_id':cid,'question_id':q['question_id'],
                'value':a['value'],'reference':ref_answer,'positions':a['positions'],'correct':a['value']==ref_answer,
                'correct_with_required_positions':a['value']==ref_answer and necessary<=ps,
                'extra_positions':sorted(ps-necessary),'missing_required_positions':sorted(necessary-ps),
                'unknown':a['value']=='unknown'})
    groups = {}
    for arm,mode in sorted({(r['arm'],r['mode']) for r in results}):
        fs = [s for s in states if (s['arm'],s['mode'])==(arm,mode)]
        qs = [q for q in question_scores if (q['arm'],q['mode'])==(arm,mode)]
        groups[arm+'-'+mode] = {'frames':len(fs),'raw_correct':sum(f['raw_correct'] for f in fs),
            'correct':sum(f['correct'] for f in fs),'wrong':sum(f['wrong'] for f in fs),'unknown':sum(f['unknown'] for f in fs),
            'frames_with_internal_conflicts':sum(bool(f['issues']) for f in fs),
            'questions':len(qs),'question_correct':sum(q['correct'] for q in qs),
            'question_grounded':sum(q['correct_with_required_positions'] for q in qs),
            'question_unknown':sum(q['unknown'] for q in qs),
            'extra_positions_on_correct_questions':sum(len(q['extra_positions']) for q in qs if q['correct'])}
    paired = []
    for cid,qs in questions.items():
        for q in qs:
            if q['answer']==q['reversed_answer']:
                continue
            rows = [x for x in question_scores if x['arm']=='B' and x['case_id']==cid and x['question_id']==q['question_id']
                    and x['mode'] in ['aligned','reversed']]
            paired.append({'case_id':cid,'question_id':q['question_id'],
                           'both_correct':len(rows)==2 and all(x['correct'] for x in rows),
                           'both_grounded':len(rows)==2 and all(x['correct_with_required_positions'] for x in rows)})
    queue = read_jsonl(out/'review.queue.jsonl')
    index = {(s['arm'],s['case_id'],s['position']):s for s in states if s['mode']=='visual_only'}
    triggers = {(e['case_id'],e['position']) for e in queue}
    cs = [s for s in states if s['arm']=='C']
    review_stats = {'triggered_frames':len(queue),'C_wrong_frames':sum(s['wrong'] for s in cs),
        'C_wrong_frames_triggered':sum(s['wrong'] and (s['case_id'],s['position']) in triggers for s in cs),
        'C_unknown_frames':sum(s['unknown'] for s in cs),
        'C_unknown_frames_triggered':sum(s['unknown'] and (s['case_id'],s['position']) in triggers for s in cs),
        'reasons':dict(Counter(reason for e in queue for reason in e['reasons'])),
        'resolved':sum(e['result']['review']['status']=='corroborated_candidate' for e in queue),
        'unresolved':sum(e['result']['review']['status']=='unresolved' for e in queue),
        'wrong_corroborated':sum(e['result']['review']['status']=='corroborated_candidate' and index[('D',e['case_id'],e['position'])]['wrong'] for e in queue),
        'wrong_to_correct':sum(s['wrong'] and index[('D',s['case_id'],s['position'])]['correct'] for s in cs),
        'wrong_to_unknown':sum(s['wrong'] and index[('D',s['case_id'],s['position'])]['unknown'] for s in cs),
        'correct_to_unknown':sum(s['correct'] and index[('D',s['case_id'],s['position'])]['unknown'] for s in cs),
        'correct_to_wrong':sum(s['correct'] and index[('D',s['case_id'],s['position'])]['wrong'] for s in cs)}
    base_scores = json.loads((old/'run-r1/visual.evaluation.json').read_text())
    receipts = [json.loads(p.read_text()) for p in (out/'runs').glob('*.json')]
    if (out/'base-reuse.json').exists():
        receipts += [json.loads(Path(p).read_text()) for p in json.loads((out/'base-reuse.json').read_text())['receipts']]
    b_visual_cost = sum(r['wall_seconds'] for r in receipts if Path(r['receipt']).stem.startswith('B-') and Path(r['receipt']).stem.endswith('visual_only'))
    metrics = json.loads((out/'metrics.json').read_text())
    costs = {'B_visual_only_calls':6,'B_visual_only_seconds':b_visual_cost,
             'C_calls':metrics['by_arm']['C']['calls'],'C_seconds':metrics['by_arm']['C']['seconds'],
             'D_incremental_calls':metrics['by_arm']['D']['calls'],'D_incremental_seconds':metrics['by_arm']['D']['seconds'],
             'D_required_pipeline_calls':6+metrics['by_arm']['C']['calls']+metrics['by_arm']['D']['calls'],
             'D_required_pipeline_seconds':b_visual_cost+metrics['by_arm']['C']['seconds']+metrics['by_arm']['D']['seconds'],
             'note':'D requires B visual_only for conflict detection. Preparation/generation times exclude model load, source verification and Codex review.'}
    summary = {'groups':groups,'B_joint_pipeline_reversal':{'targets':len(paired),'both_correct':sum(r['both_correct'] for r in paired),
        'both_grounded':sum(r['both_grounded'] for r in paired)},'review':review_stats,'costs':costs,
        'A_v4_evaluation_unchanged':base_scores,
        'limits':['C/D reordering is program sorting, not a model temporal score.',
                  'Reference is Codex-reviewed development data, not independent human gold.',
                  'Template consistency is not perception accuracy; cues and boxes require separate source review.',
                  'Primary-state scoring does not measure recall of all visible objects.']}
    write_jsonl(out/'frame.scores.jsonl',states); write_jsonl(out/'question.scores.jsonl',question_scores)
    write_jsonl(out/'reversal.scores.jsonl',paired); dump(out/'evaluation.json',summary)
    print(json.dumps({'groups':groups,'review':review_stats,'reversal':summary['B_joint_pipeline_reversal'],'costs':costs},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args(); run(a.root.resolve(),a.out.resolve(),a.baseline.resolve())
