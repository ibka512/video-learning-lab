"""Read-only source/ablation audit; references never used by inference runner."""
import argparse,json
from collections import Counter
from pathlib import Path
from fact_ledger import hash_object
from ocr_routing import digest,dump,read_jsonl,write_jsonl
from run_video_understanding_v4 import MODES,semantic_units,KINDS


def expected_support(question,reversed_order):
    qid=question['question_id']
    if qid=='first':return {0}
    if qid=='last':return {3}
    positions=set(question['support_positions'])
    return {3-p for p in positions} if reversed_order else positions


def score_visual(cases,gold,records):
    scores=[];details=[]
    for c in cases:
        target=next(t for t in gold['cases'] if t['case_id']==c['case_id'])
        for mode in MODES:
            rec=records[(c['case_id'],mode)];output=rec.get('output')
            for q in target['questions']:
                expected=q['reversed_answer'] if mode=='reversed' else q['answer']
                actual=output['answers'][q['question_id']] if output else None
                supports=expected_support(q,mode=='reversed')
                correct=bool(actual and actual['value']==expected)
                cited=bool(correct and mode!='text_only' and supports<=set(actual['positions']))
                scores.append({'case_id':c['case_id'],'mode':mode,'question_id':q['question_id'],
                     'answer':actual,'expected_visible_answer':expected,'content_correct':correct,
                     'required_positions':sorted(supports),'correct_with_required_positions':cited,
                     'appropriate_text_abstention':bool(mode=='text_only' and actual and actual['value']=='unknown'),
                     'unknown':bool(actual and actual['value']=='unknown')})
            details.append({'case_id':c['case_id'],'mode':mode,'output':output})
    summary={}
    for mode in MODES:
        rows=[s for s in scores if s['mode']==mode]
        summary[mode]={'questions':len(rows),'content_correct':sum(s['content_correct'] for s in rows),
             'correct_with_required_positions':sum(s['correct_with_required_positions'] for s in rows),
             'unknown':sum(s['unknown'] for s in rows),
             'appropriate_text_abstentions':sum(s['appropriate_text_abstention'] for s in rows)}
    pairs=[]
    for t in gold['cases']:
        for q in t['questions']:
            a=next(s for s in scores if (s['case_id'],s['mode'],s['question_id'])==(t['case_id'],'aligned',q['question_id']))
            b=next(s for s in scores if (s['case_id'],s['mode'],s['question_id'])==(t['case_id'],'reversed',q['question_id']))
            if q['answer']!=q['reversed_answer']:
                pairs.append({'case_id':t['case_id'],'question_id':q['question_id'],
                       'both_directions_correct':a['content_correct'] and b['content_correct'],
                       'both_directions_correct_with_positions':a['correct_with_required_positions'] and b['correct_with_required_positions'],
                       'answer_changed':bool(a['answer'] and b['answer'] and a['answer']['value']!=b['answer']['value'])})
    return {'modes':summary,'reversal_sensitive_targets':len(pairs),
            'both_directions_correct':sum(p['both_directions_correct'] for p in pairs),
            'both_directions_correct_with_positions':sum(p['both_directions_correct_with_positions'] for p in pairs),
            'answers_changed':sum(p['answer_changed'] for p in pairs),'reversal_pairs':pairs,
            'measurement_limit':'position checks show cited evidence coverage, not independent causal or free-description correctness'},scores,details


def evaluate(root,out):
    freeze=json.loads((root/'references/freeze.json').read_text())
    for name,h in freeze.items():
        if digest(root/'references'/name)!=h:raise ValueError('reference changed: '+name)
    identity=json.loads((out/'run-identity.json').read_text())
    for name,h in identity['sources'].items():
        if digest(Path(name))!=h:raise ValueError('public source changed')
    origin=json.loads((root/'sources/source-identity.json').read_text())
    if digest(Path(origin['video']))!=origin['source_file_sha256']:raise ValueError('source video changed')
    cases=read_jsonl(root/'sources/cases.public.jsonl')
    for c in cases:
        for row in c['frames']:
            for name in ['source_image','crop']:
                if digest(Path(row[name]))!=row[name+'_sha256']:raise ValueError('source image changed')
            from PIL import Image
            cropped=Image.open(row['source_image']).convert('RGB').crop(row['bbox_top_left'])
            actual=Image.open(row['crop']).convert('RGB')
            if cropped.size!=actual.size or cropped.tobytes()!=actual.tobytes():raise ValueError('crop not exact source region')
    records={}
    allrecords=[]
    for path in (out/'runs').glob('*.json'):
        r=json.loads(path.read_text());allrecords.append(r)
        # Inference requests have neither reference roles nor expected answers.
        if any(s in json.dumps(r['request']) for s in ['reversed_answer','support_positions','allowed_roles','all_text']):raise ValueError('reference label leaked')
        if r['model_identity_sha256']!=hash_object(identity):raise ValueError('identity mismatch')
    packing=[]
    for c in cases:
        cid=c['case_id']
        for mode in MODES:records[(cid,mode)]=json.loads((out/'runs'/(cid+'-'+mode+'.json')).read_text())
        plain=records[(cid,'visual_only')]['visual_input'];aligned=records[(cid,'aligned')]['visual_input'];rev=records[(cid,'reversed')]['visual_input']
        assert records[(cid,'text_only')]['visual_input'] is None
        assert plain['video_tensor_sha256']==aligned['video_tensor_sha256']
        assert [m['crop_sha256'] for m in rev['mapping']]==list(reversed([m['crop_sha256'] for m in aligned['mapping']]))
        assert all(r['source_timestamps_visible_to_model'] is False for r in [plain,aligned,rev])
        packing.append({'case_id':cid,'visual_only_aligned_pixels_identical':True,'reversed_same_source_crops':True,
                        'media_tokens':aligned['media_tokens'],'resized_size':[aligned['packing']['resized_frame_width'],aligned['packing']['resized_frame_height']]})
    vis,scores,details=score_visual(cases,json.loads((root/'references/visual-gold.json').read_text()),records)
    write_jsonl(out/'visual.scores.jsonl',scores);dump(out/'visual.evaluation.json',vis)
    dump(out/'input-audit.json',{'source_and_crops_verified':True,'references_unchanged':True,'request_count':len(allrecords),
          'reference_labels_absent_from_requests':True,'packing':packing,'reference_basis':'Codex reviewed source crops, not independent human gold'})
    packets=read_jsonl(root/'sources/semantic.public.jsonl');targets=json.loads((root/'references/semantic-targets.json').read_text())
    reference=json.loads((root/'references/semantic-gold.json').read_text());goldunits=reference['units']
    candidates=[];sem_summary={};sem_scores=[]
    typepolicy={'term_identity':set(KINDS),'definition':{'definition','behavior'},'limitation':{'limitation'},'comparison':{'comparison'},'usage':{'usage'}}
    for mode in ['clauses','sentences']:
        fs=[]
        for p in packets:
            record=json.loads((out/'runs'/(p['packet_id']+'-'+mode+'.json')).read_text())
            units={u['ref']:u for u in semantic_units(p,mode)}
            if not record['output']:continue
            for kind,refs in record['output']['fields'].items():
                if not refs:continue
                selected=[u for u in semantic_units(p,mode) if u['ref'] in refs]
                involved=[u for u in p['units'] if (u['unit_id'] in refs if mode=='clauses' else u['segment_ref'] in refs)]
                # This is diagnostic clause-role consistency, not semantic field precision.
                allowed={'definition':{'identity','definition','behavior'},'usage':{'usage','condition'},'behavior':{'behavior','definition'},
                         'limitation':{'limitation','condition'},'comparison':{'comparison'},'analogy':{'analogy'}}[kind]
                check=[next(g for g in goldunits if g['unit_id']==u['unit_id']) for u in involved]
                consistent=[g for g in check if allowed & set(g['allowed_roles'])]
                core={'mode':mode,'subject':p['subject'],'kind':kind,'refs':refs,
                      'quote':'\n'.join(u['text'] for u in selected),'unit_ids':[u['unit_id'] for u in involved],
                      'source_start':min(u['start'] for u in involved),'source_end':max(u['end'] for u in involved),
                      'diagnostic_compatible_units':len(consistent),'diagnostic_source_units':len(check),
                      'packet_id':p['packet_id'],'origin':'model_source_selection_program_verbatim_copy','status':'candidate_not_source_approved'}
                fs.append({**core,'candidate_id':'C'+hash_object(core)[:18]})
        candidates+=fs
        for t in targets['targets']:
            term=t['id'].split('-')[0].casefold();a,b=t['window']
            related=[f for f in fs if term in f['subject'].casefold() and f['source_start']<b and f['source_end']>a and f['kind'] in typepolicy[t['kind']]]
            norm=lambda s:''.join(s.split()).casefold()
            value=norm(' '.join(f['subject']+' '+f['quote'] for f in related))
            missing=[s for s in t['all_text'] if norm(s) not in value]
            sem_scores.append({'mode':mode,'target':t['id'],'typed_wording_retained':not missing,'missing':missing})
        total=sum(f['diagnostic_source_units'] for f in fs);compatible=sum(f['diagnostic_compatible_units'] for f in fs)
        sem_summary[mode]={'candidate_fields':len(fs),'typed_wording_retained':sum(s['typed_wording_retained'] for s in sem_scores if s['mode']==mode),
             'target_count':len(targets['targets']),'diagnostic_compatible_unit_occurrences':compatible,'diagnostic_unit_occurrences':total,
             'all_selected_units_role_compatible_fields':sum(f['diagnostic_compatible_units']==f['diagnostic_source_units'] for f in fs),
             'source_units_selected':len({u for f in fs for u in f['unit_ids']})}
    write_jsonl(out/'semantic.candidates.jsonl',candidates);write_jsonl(out/'semantic.scores.jsonl',sem_scores)
    dump(out/'semantic.evaluation.json',{'modes':sem_summary,'same_prompt_and_field_schema':True,
         'only_source_unit_granularity_differs':True,'gold_unit_compatibility_is_not_semantic_accuracy':True,
         'target_policy':'Same 18 target wording/time/object/type rules for both representations; no Codex corrections in scores'})
    print(json.dumps({'visual':vis['modes'],'reversal_pair_correct':vis['both_directions_correct'],
                      'semantic':sem_summary,'source_audit':'passed'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();evaluate(a.root.resolve(),a.out.resolve())
