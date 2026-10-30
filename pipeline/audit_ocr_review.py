"""Validate persisted provenance and actual knowledge inputs; no OCR truth claims."""
from pathlib import Path
import argparse,json,hashlib
from ocr_routing import digest,read_jsonl,dump
from refinement_context import for_interval

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--baseline',required=True,type=Path);a=p.parse_args();r=a.root
 plan=json.loads((r/'sources/route-plan.json').read_text());refined=read_jsonl(r/'sources/ocr.refined.jsonl');jobs={j['job_id']:j for j in plan['jobs']}
 assert digest(a.baseline/'sources/ocr.jsonl')==plan['source_ocr_sha256'],'baseline changed'
 assert digest(Path(__file__).with_name('ocr_routing.py'))==plan['router_sha256'],'stale routing plan'
 receipts={};images={}
 def check_image(path,expected):
  if path not in images:images[path]=digest(r/path)
  assert images[path]==expected,path
 for j in jobs.values():
  check_image(j['image'],j['image_sha256']);rp=r/'receipts'/(j['job_id']+'.json');receipt=json.loads(rp.read_text());assert receipt['status']=='success';assert receipt['identity']['image_sha256']==j['image_sha256'];receipts[j['job_id']]=(receipt,digest(rp))
  for o in j['occurrences']:check_image(o['source_image'],o['source_image_sha256'])
 seen=set()
 for row in refined:
  assert row['ocr_id'] not in seen;seen.add(row['ocr_id']);j=jobs[row['job_id']];receipt,rh=receipts[row['job_id']];assert row['receipt_sha256']==rh
  assert row['review_status']=='unverified_ocr_evidence' and row['time_scope']=='this_exact_frame_only'
  assert row['source_identity']==plan['source_identity']
  x,y,w,h=row['bbox_bottom_left'];assert min(x,y,w,h)>=-1e-5 and x+w<=1.00001 and y+h<=1.00001
  occurrences=[o for o in j['occurrences'] if o['frame_id']==row['frame_id'] and o['time']==row['time']];assert len(occurrences)==1
  o=occurrences[0];assert row['source_image']==o['source_image'];assert row['source_image_sha256']==o['source_image_sha256']
  matching=[t for t in receipt['texts'] if t['text']==row['text'] and [t['bbox_crop'][0]+o['bbox'][0],t['bbox_crop'][1]+o['bbox'][1],t['bbox_crop'][2]+o['bbox'][0],t['bbox_crop'][3]+o['bbox'][1]]==row['bbox_top_left']];assert matching
 packets=list((r/'knowledge-inputs').glob('*-page-*.json'));intervals=set();pages=set();unique_injected=set();model_packets=[]
 for path in packets:
  q=json.loads(path.read_text());c=q['context'];s=c['source_interval'];intervals.add(s['segment_id']);pages.add((s['segment_id'],c['page_index']));expected=for_interval(refined,s['start'],s['end']);rs=c['refined_ocr_rows'];assert len(rs)==len(expected)
  for i,row in enumerate(expected):
   assert q['canonical_ref_aliases'][f'R{i}']==row['ocr_id'];assert rs[i][1]==row['paddle_text'] and rs[i][4:6]==[row['frame_id'],row['time']];unique_injected.add(row['ocr_id'])
 source_intervals=read_jsonl(a.baseline/'sources/source-intervals.jsonl');assert intervals=={s['segment_id'] for s in source_intervals}
 # Every new line inside the defined source timeline must reach at least one packet.
 expected_injected={x['ocr_id'] for x in refined if any(s['start']<=x['time']<s['end'] for s in source_intervals)};assert unique_injected==expected_injected
 for path in (r/'knowledge-generation').glob('*-page-*.json'):
  q=json.loads(path.read_text());ctx=q['context'];assert q['identity']['context_sha256']==hashlib.sha256(json.dumps(ctx,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  assert q['identity']['code_sha256']==digest(Path(__file__).with_name('extract_knowledge.py'))
  s=ctx['source_interval'];inp=json.loads((r/'knowledge-inputs'/path.name).read_text());assert inp['context']==ctx
  permitted={s['segment_id']}|{n['segment_id'] for n in ctx['neighbor_speech']}|{e['event_id'] for e in ctx['verified_source_events']}|set(q['canonical_ref_aliases'])
  claims=q['output']['claims'] if q['output'] else [];assert all(set(c['evidence_refs'])<=permitted for c in claims)
  model_packets.append({'path':str(path.relative_to(r)),'claims':len(claims),'new_evidence_rows_in_context':len(ctx['refined_ocr_rows']),'claims_citing_new_evidence':sum(any(v.startswith('R') for v in c['evidence_refs']) for c in claims),'semantic_correctness_not_asserted':True})
 conflicts=read_jsonl(r/'sources/ocr.conflicts.jsonl');assert all(q['ocr_id'] in seen and q['status']=='pending_source_review' for q in conflicts)
 dump(r/'sources/integration-audit.json',{'passed':True,'checked_region_receipts':len(receipts),'checked_images':len(images),'checked_refined_lines':len(refined),'checked_conflict_rows':len(conflicts),'source_ocr_unchanged':True,'prepared_intervals':len(intervals),'prepared_packets':len(packets),'unique_refined_lines_in_packets':len(unique_injected),'model_packets':model_packets,'no_semantic_correctness_or_all_video_recall_claim':True})
 print('Provenance and input audit passed',len(receipts),'regions',len(refined),'lines',len(packets),'packets')
if __name__=='__main__':main()
