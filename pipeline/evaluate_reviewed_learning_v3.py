"""Score source-reviewed artifacts separately from immutable model candidates."""
import argparse,json,re
from pathlib import Path
from ocr_routing import read_jsonl,dump
from finalize_learning_v3 import finalize
from evaluate_learning_v3 import evaluate


def score(root,reference,targets=None):
    finalize(root)  # Recheck source, original fact and graph digests before scoring.
    raw=evaluate(root,reference,targets)
    gold=json.loads(reference.read_text());bygold={u['unit_id']:u for u in gold['units']}
    fs=read_jsonl(root/'sources/units.reviewed.jsonl')
    errors=[{'unit_id':f['unit_id'],'actual':f['role'],'allowed':bygold[f['unit_id']]['allowed_roles']} for f in fs if f['role'] not in bygold[f['unit_id']]['allowed_roles']]
    result={'benchmark':'source_reviewed_development_regression_not_model_accuracy',
            'source_units':len(bygold),'reviewed_units':len(fs),'role_matches':len(fs)-len(errors),'role_errors':errors,
            'explicit_Codex_role_corrections':sum(f['origin']=='Codex_source_review_role_correction' for f in fs),
            'raw_role_matches_for_comparison':raw['raw_role_matches'],
            'independent_human_accuracy_measured':False,'execution_verified':False}
    for f in fs:
        g=bygold[f['unit_id']]
        assert all(f[k]==g[k] for k in ['quote','segment_ref','char_start','char_end','start','end'])
    if 'required_action_refs' in gold:
        graphs=read_jsonl(root/'sources/procedures.reviewed.jsonl')
        steps=graphs[0]['procedure']['steps'] if graphs else []
        def find(uid):return next((i for i,s in enumerate(steps) if uid in s['action_refs']),None)
        def ancestors(i):
            result=set();stack=steps[i]['after'].copy()
            while stack:
                j=stack.pop()
                if j not in result:result.add(j);stack+=steps[j]['after']
            return result
        actions={r for s in steps for r in s['action_refs']}
        missing=set(gold['required_action_refs'])-actions;extras=actions-set(gold['required_action_refs'])
        required=gold['mandatory_action_refs']
        order=not missing and all(find(a)<=find(b) for a,b in zip(required,required[1:]))
        dependency=order and all(find(a)==find(b) or find(a) in ancestors(find(b)) for a,b in zip(required,required[1:]))
        branch=gold['branch']
        conditions=all(find(r) is not None and branch['condition'] in steps[find(r)]['condition_refs'] for r in branch['actions'])
        context=all(find(r) is not None and ctx in steps[find(r)]['action_refs']+steps[find(r)]['context_refs'] for r,ctx in gold['shared_context'].items())
        linked={r for s in steps for r in s['result_refs']}
        result['procedure']={'source_reviewed_steps':len(steps),'required_action_units':len(gold['required_action_refs']),
            'missing_action_units':sorted(missing),'unsupported_action_units':sorted(extras),
            'mandatory_order_retained':order,'mandatory_dependency_paths_retained':dependency,
            'branch_condition_on_both_actions':conditions,'shared_verb_context_retained':context,
            'required_results_linked':set(gold['results'])<=linked,
            'reference_acceptance':not missing and not extras and order and dependency and conditions and context and set(gold['results'])<=linked,
            'graph_origin':'Codex_source_review_graph_correction'}
    if targets:
        ts=json.loads(targets.read_text())['targets'];types={'definition':{'identity','definition','behavior','comparison'},'limitation':{'limitation'},
            'comparison':{'comparison','definition','behavior','limitation'},'usage':{'usage'},'term_identity':{'identity'}}
        compact=lambda x:re.sub(r'\s+','',x).casefold()
        rows=[]
        for t in ts:
            a,b=t['window'];subject=t['id'].split('-')[0].casefold()
            related=[f for f in fs if subject in f['subject'].casefold() and f['start']<b and f['end']>a and f['role'] in types[t['kind']]]
            text=compact(''.join(f['quote'] for f in related))
            rows.append({'id':t['id'],'retained':all(compact(s) in text for s in t['all_text'])})
        result['targets']=rows;result['typed_wording_retained']=sum(r['retained'] for r in rows);result['target_count']=len(rows)
        result['type_policy_note']=raw['target_policy']
    dump(root/'sources/evaluation.reviewed.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--frontend-targets',type=Path);a=p.parse_args();score(a.root,a.reference,a.frontend_targets)
