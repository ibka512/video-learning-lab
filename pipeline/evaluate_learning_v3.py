"""Evaluation-only references, exact source integrity, role and dependency scoring."""
import argparse,json,re
from pathlib import Path
from fact_ledger import hash_object
from ocr_routing import digest,dump,read_jsonl
from run_learning_v3 import load_validated


def evaluate(root,reference_path,targets_path=None):
    m,ps,cs,fs=load_validated(root)
    ref=json.loads(reference_path.read_text())
    freeze=json.loads((reference_path.parent/'freeze-manifest.json').read_text())
    for name,sha in freeze.items():
        if digest(reference_path.parent/name)!=sha:raise ValueError('reference freeze changed')
    refs={u['unit_id']:u for u in ref['units']}
    facts={f['unit_id']:f for f in fs}
    if set(facts)-set(refs):raise ValueError('unknown evaluation source')
    errors=[]
    for uid,f in facts.items():
        gold=refs[uid]
        for k in ['quote','char_start','char_end','segment_ref','start','end']:
            assert f[k]==gold[k],(uid,k)
        if f['role'] not in gold['allowed_roles']:
            errors.append({'unit_id':uid,'quote':f['quote'],'actual':f['role'],'allowed':gold['allowed_roles']})
    missing=set(refs)-set(facts)
    exported=read_jsonl(root/'sources/units.exported.jsonl')
    reviewpath=root/'sources/source-reviews.jsonl'
    reviews={r['fact_id']:r for r in read_jsonl(reviewpath)} if reviewpath.exists() else {}
    for f in exported:
        assert reviews[f['fact_id']]['status']=='approve'
        assert reviews[f['fact_id']]['fact_sha256']==hash_object(f)
    modelchecks=read_jsonl(root/'sources/model-audits.jsonl')
    wrong={e['unit_id'] for e in errors}
    result={'reference_sha256':digest(reference_path),'reference_basis':ref['review_basis'],
            'benchmark':'development_regression_not_new_holdout','source_units':len(refs),'classified_units':len(fs),
            'missing_units':sorted(missing),'raw_role_matches':len(fs)-len(errors),'raw_role_errors':errors,
            'reviewed_units':len(exported),'rejected_or_pending_units':len(fs)-len(exported),
            'auxiliary_supported_but_reference_role_wrong':[r['unit_id'] for r in modelchecks if r['verdict']=='supported' and r['unit_id'] in wrong],
            'source_spans_exact':True,'clauses_independently_time_aligned':False,
            'no_speech_promoted_to_observed_execution':all(f['observed_execution'] is False for f in fs)}
    if 'required_action_refs' in ref:
        assert len(cs)==1
        c=cs[0];steps=c['procedure']['steps']
        def locate(uid):return next((i for i,s in enumerate(steps) if uid in s['action_refs']),None)
        def ancestors(i):
            found=set();stack=list(steps[i]['after'])
            while stack:
                j=stack.pop()
                if j in found:continue
                found.add(j);stack.extend(steps[j]['after'])
            return found
        actions={r for s in steps for r in s['action_refs']}
        missingactions=set(ref['required_action_refs'])-actions
        must=ref['mandatory_action_refs']
        orderok=not any(locate(r) is None for r in must) and all(locate(a)<=locate(b) for a,b in zip(must,must[1:]))
        dependencyok=bool(orderok and all(locate(a)==locate(b) or locate(a) in ancestors(locate(b)) for a,b in zip(must,must[1:])))
        branch=ref['branch'];condition=branch['condition']
        branchok=all(locate(r) is not None and condition in steps[locate(r)]['condition_refs'] for r in branch['actions'])
        ctxok=all(locate(r) is not None and ctx in steps[locate(r)]['context_refs']+steps[locate(r)]['action_refs']
                  for r,ctx in ref['shared_context'].items())
        falsebranch=[r for s in steps for r in s['action_refs'] if condition in s['condition_refs'] and r not in branch['actions']]
        matchedresults={r for s in steps for r in s['result_refs']}
        rolesresult={r for r,f in facts.items() if f['role']=='result'}
        legacy=read_jsonl(root/'sources/legacy-reviewed-fields.jsonl')
        before=[f for f in read_jsonl(Path(m['legacy_root'])/'sources/facts.exported.jsonl') if f['kind']!='author_statement']
        assert legacy==before
        procedureok=not missingactions and orderok and dependencyok and branchok and ctxok and not falsebranch and not c['alarms'] and not c['link_error']
        result['procedure']={'steps':len(steps),'required_action_units':len(ref['required_action_refs']),
             'represented_action_units':len(actions),'missing_action_units':sorted(missingactions),
             'extra_action_units':sorted(actions-set(ref['required_action_refs'])),
             'mandatory_action_order_retained':orderok,'mandatory_dependency_paths_retained':dependencyok,
             'both_branch_actions_keep_condition':branchok,'shared_verb_context_retained':ctxok,
             'unconditional_actions_wrongly_made_conditional':falsebranch,
             'expected_results_classified':sorted(set(ref['results'])&rolesresult),
             'expected_results_linked_to_steps':sorted(set(ref['results'])&matchedresults),
             'procedural_reference_acceptance':procedureok,'alarms':c['alarms'],
             'execution_success_verified':False}
        result['legacy_exact_match']=True;result['legacy_fields_preserved']=len(legacy)
    if targets_path:
        targets=json.loads(targets_path.read_text())
        kindmap={'definition':{'identity','definition','behavior','comparison'},
                 'limitation':{'limitation'},'comparison':{'comparison','definition','behavior','limitation'},
                 'usage':{'usage'},'term_identity':{'identity'}}
        compact=lambda x:re.sub(r'\s+','',x).casefold()
        scores=[]
        for t in targets['targets']:
            subject=t['id'].split('-')[0].casefold();a,b=t['window']
            def check(items):
                related=[f for f in items if subject in f['subject'].casefold() and f['start']<b and f['end']>a and f['role'] in kindmap[t['kind']]]
                text=compact(' '.join(f['quote'] for f in related))
                return all(compact(s) in text for s in t['all_text'])
            scores.append({'id':t['id'],'raw_typed_wording':check(fs),'reviewed_typed_wording':check(exported)})
        result['targets']=scores;result['target_count']=len(scores)
        result['raw_typed_wording_retained']=sum(s['raw_typed_wording'] for s in scores)
        result['reviewed_typed_wording_retained']=sum(s['reviewed_typed_wording'] for s in scores)
        result['target_policy']='v3 atomic roles; interaction/definition roles jointly express meaning; comparison qualifier may live in adjacent role. Not identical type policy to v2.'
    dump(root/'sources/evaluation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--frontend-targets',type=Path)
    a=p.parse_args();evaluate(a.root,a.reference,a.frontend_targets)
