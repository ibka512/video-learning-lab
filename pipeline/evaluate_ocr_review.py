"""Evaluate frozen targets only AFTER automatic routing. Never supplies routing input."""
import argparse,importlib.util,json,re
from pathlib import Path
from ocr_routing import read_jsonl,write_jsonl,dump,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);a=p.parse_args();r=a.root;b=a.benchmark
 spec=importlib.util.spec_from_file_location('frozen_score',b/'工具/score.py');sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
 gold=json.loads((b/'sources/benchmark.json').read_text());plan=json.loads((r/'sources/route-plan.json').read_text());refined=read_jsonl(r/'sources/ocr.refined.jsonl');results=[]
 critical={'setting_key','setting_key_transition','help_header','menu','resource','resource_transition','window_title'}
 for u in gold['targets']:
  texts=[{'text':x['text'],'bbox':x['bbox_top_left']} for x in refined if x['frame_id']==u['frame_id']]
  selected,variants=sc.candidates(texts,u['bbox_source'],[0,0]);wanted=sc.norm(u['expected']);best=min(variants,key=lambda s:sc.distance(wanted,sc.norm(s)),default='');exact=any(sc.norm(s)==wanted for s in variants)
  pattern=re.escape(wanted)
  if u['kind'] in ['setting_key','setting_key_transition','node_name','resource','resource_transition']:pattern=r'(?<![A-Za-z0-9_/])'+pattern+r'(?![A-Za-z0-9_/])'
  box=u['bbox_source'];cy=(box[1]+box[3])/2
  # Loose hand-drawn target boxes contain whitespace; this is a candidate overlap,
  # not proof that the whole text was inside the crop. Exact recovery measures that.
  routed=any(o['frame_id']==u['frame_id'] and o['bbox'][1]<=cy<=o['bbox'][3] and min(box[2],o['bbox'][2])>max(box[0],o['bbox'][0]) for j in plan['jobs'] for o in j['occurrences'])
  results.append({**u,'critical_category':u['kind'] in critical,'region_candidate_overlap':routed,'exact':exact,'contains_complete_target_secondary':any(re.search(pattern,sc.norm(v)) is not None for v in variants),'no_text_in_target_region':not selected,'best_spatial_candidate':best})
 def stat(rows):return {'targets':len(rows),'region_candidate_overlap':sum(x['region_candidate_overlap'] for x in rows),'exact':sum(x['exact'] for x in rows),'contains_complete_target_secondary':sum(x['contains_complete_target_secondary'] for x in rows),'no_text_in_target_region':sum(x['no_text_in_target_region'] for x in rows)}
 previous=read_jsonl(b/'sources/scores.jsonl');comparisons={}
 for condition in [('apple-vision','full'),('paddle-v6-medium','crop'),('paddle-v6-medium','full')]:
  rows=[x for x in previous if (x['engine'],x['mode'])==condition];q=[x for x in rows if x['kind'] in critical];comparisons['/'.join(condition)]={'all_exact':sum(x['exact'] for x in rows),'critical_exact':sum(x['exact'] for x in q),'critical_total':len(q)}
 summary={'benchmark_sha256':digest(b/'sources/benchmark.json'),'route_plan_sha256':digest(r/'sources/route-plan.json'),'scope':'same 69 pre-existing targets in 24 correlated frames of ONE video; NFC+whitespace normalization only; no new independent benchmark; v1 used generic rules; v2 rules revised after v1 failures, so these are regression results, not an independent holdout','all_targets':stat(results),'critical_targets':stat([x for x in results if x['critical_category']]),'categories':{k:stat([x for x in results if x['kind']==k]) for k in sorted({x['kind'] for x in results})},'prior_conditions':comparisons,'not_routed_critical':[x for x in results if x['critical_category'] and not x['region_candidate_overlap']],'critical_not_exact':[x for x in results if x['critical_category'] and not x['exact']]}
 write_jsonl(r/'sources/automatic-region-scores.jsonl',results);dump(r/'sources/automatic-region-evaluation.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
