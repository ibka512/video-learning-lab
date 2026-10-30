"""Offline scoring for frozen independent attributes, not a general-video accuracy claim."""
import argparse,json
from pathlib import Path
from ocr_routing import digest,dump,read_jsonl,write_jsonl
from object_evidence_v6 import summarize_record


def run(root,out):
    for name,sha in json.loads((root/'references/freeze.json').read_text()).items():
        if digest(root/name)!=sha:raise ValueError('frozen source changed: '+name)
    targets={r['task_id']:r['expected'] for r in json.loads((root/'references/properties.json').read_text())['targets']}
    tasks={r['task_id']:r for r in read_jsonl(root/'sources/tasks.public.jsonl')}
    records=read_jsonl(out/'records.jsonl');base=read_jsonl(out/'base.records.jsonl');pairs=read_jsonl(out/'pairs.jsonl')
    scores=[];metrics={};review_scores=[]
    for pair in pairs:
        tid=pair['task_id'];task=tasks[tid]
        for arm in ['native','temporal']:
            rec=json.loads(Path(pair[arm]).read_text());data=rec['output'] or {key:'unknown' for key in task['properties']}
            for key,value in data.items():
                expected=targets[tid][key]
                scores.append({'task_id':tid,'case_id':task['case_id'],'position':task['position'],'arm':arm,
                    'property':key,'value':value,'expected':expected,'correct':value==expected,
                    'unknown':value=='unknown','wrong':value not in ['unknown',expected],
                    'receipt':rec['receipt']})
    for arm in ['native','temporal']:
        rows=[s for s in scores if s['arm']==arm];windows=[s for s in rows if s['property']=='window_kind']
        content=[s for s in rows if s['property']!='window_kind']
        positives=[s for s in content if s['expected']=='present'];negatives=[s for s in content if s['expected']=='absent']
        tp=sum(s['value']=='present' for s in positives);fp=sum(s['value']=='present' for s in negatives)
        metrics[arm]={'targets':len(rows),'correct':sum(s['correct'] for s in rows),'wrong':sum(s['wrong'] for s in rows),
            'unknown':sum(s['unknown'] for s in rows),'window_correct':sum(s['correct'] for s in windows),'window_total':len(windows),
            'content_correct':sum(s['correct'] for s in content),'content_total':len(content),
            'positive_total':len(positives),'true_positive':tp,'false_positive':fp,
            'false_negative':sum(s['value']=='absent' for s in positives),
            'positive_unknown':sum(s['unknown'] for s in positives),
            'true_negative':sum(s['value']=='absent' for s in negatives),'negative_unknown':sum(s['unknown'] for s in negatives),
            'positive_recall':tp/len(positives),'positive_precision':tp/(tp+fp) if tp+fp else None,
            'all_properties_correct_frames':sum(all(s['correct'] for s in rows if (s['case_id'],s['position'])==(c,p))
                   for c,p in {(s['case_id'],s['position']) for s in rows})}
        coexist=[tid for tid,gold in targets.items() if len(gold)==2 and set(gold.values())=={'present'}]
        metrics[arm]['coexistence_frames']=len(coexist)
        metrics[arm]['both_coexisting_properties_correct']=sum(all(s['correct'] for s in rows if s['task_id']==tid) for tid in coexist)
    for original,record in zip(base,records):
        if record['initial']!=original['initial']:raise ValueError('initial observation overwritten')
        tid=record['initial']['task_id']
        for extra in record['additional']:
            if extra['purpose']!='targeted_reread':continue
            for key,value in extra['observation']['values'].items():
                initial=record['initial']['values'][key];expected=targets[tid][key]
                review_scores.append({'task_id':tid,'property':key,'initial':initial,'reread':value,'expected':expected,
                                      'reread_correct':value==expected,'would_replace_correct_with_wrong':initial==expected and value not in ['unknown',expected],
                                      'new_correct_candidate':initial!=expected and value==expected})
    summaries=[summarize_record(r) for r in records]
    statuses=[s['status'] for row in summaries for s in row['properties'].values()]
    queue=read_jsonl(out/'reread.queue.jsonl')
    review={'initial_records_preserved':len(records),'initial_fields_preserved':sum(len(r['initial']['values']) for r in records),
            'triggered_tasks':len(queue),'completed_tasks':sum(r['status']=='appended_without_replacement' for r in queue),
            'deferred_tasks':sum(r['status']=='deferred_budget' for r in queue),
            'reread_property_count':len(review_scores),'reread_correct':sum(r['reread_correct'] for r in review_scores),
            'correct_initial_fields_that_reread_would_corrupt':sum(r['would_replace_correct_with_wrong'] for r in review_scores),
            'new_correct_candidates':sum(r['new_correct_candidate'] for r in review_scores),
            'conflict_fields':statuses.count('unreviewed_conflict'),'automatic_promotions':0}
    by_key={(s['arm'],s['task_id'],s['property']):s for s in scores}
    comparison={'native_only_correct':0,'temporal_only_correct':0,'both_correct':0,'neither_correct':0,'answer_disagreements':0}
    for tid,gold in targets.items():
        for key in gold:
            n=by_key[('native',tid,key)];t=by_key[('temporal',tid,key)]
            bucket='both_correct' if n['correct'] and t['correct'] else 'native_only_correct' if n['correct'] else 'temporal_only_correct' if t['correct'] else 'neither_correct'
            comparison[bucket]+=1;comparison['answer_disagreements']+=n['value']!=t['value']
    result={'arms':metrics,'paired_comparison':comparison,'append_only_review':review,
            'input_pairs':len(pairs),'pairs_with_different_pixel_tensors':sum(p['different_tensor'] for p in pairs),
            'scope':'40 frozen attribute targets across 24 selected positions, existing Godot development clip; not independent validation',
            'semantic_approval':False}
    dump(out/'evaluation.json',result);write_jsonl(out/'property.scores.jsonl',scores);write_jsonl(out/'reread.scores.jsonl',review_scores)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.out.resolve())
