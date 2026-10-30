"""Attach fallible regional OCR to exact source frames and preserve engine disagreements."""
from pathlib import Path
import argparse,json,re,hashlib,unicodedata
from ocr_routing import *
def norm(t):return re.sub(r'\s+','',unicodedata.normalize('NFC',t))
def overlap_line(a,b):
 iy=max(0,min(a[3],b[3])-max(a[1],b[1]));ix=max(0,min(a[2],b[2])-max(a[0],b[0]));return iy/max(1,min(a[3]-a[1],b[3]-b[1]))>.35 and ix/max(1,min(a[2]-a[0],b[2]-b[0]))>.2

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);a=p.parse_args();r=a.root.resolve();plan=json.loads((r/'sources/route-plan.json').read_text());base=read_jsonl(a.baseline/'sources/ocr.jsonl');raw={x['frame_id']:from_native(x) for x in base};sizes={x['frame_id']:(x['width'],x['height']) for x in base}
 for f in json.loads((a.benchmark/'sources/benchmark.json').read_text())['frames']:
  raw[f['frame_id']]=json.loads((a.benchmark/'runs/apple-vision'/f'{f["frame_id"]}-full.json').read_text())['texts']
 records=[];queue=[];failed=[]
 for job in plan['jobs']:
  rp=r/'receipts'/(job['job_id']+'.json')
  if not rp.exists() or json.loads(rp.read_text())['status']!='success':failed.append(job['job_id']);continue
  receipt=json.loads(rp.read_text());assert receipt['identity']['image_sha256']==job['image_sha256']
  for occurrence in job['occurrences']:
   bx,by,_,_=occurrence['bbox'];width,height=sizes[occurrence['frame_id']]
   for n,t in enumerate(receipt['texts']):
    x0,y0,x1,y1=t['bbox_crop'];box=[x0+bx,y0+by,x1+bx,y1+by];matched=[x for x in raw[occurrence['frame_id']] if overlap_line(box,x['bbox'])]
    combined=''.join(x['text'] for x in sorted(matched,key=lambda x:x['bbox'][0]));agree=norm(t['text']) in [norm(x['text']) for x in matched]+([norm(combined)] if combined else [])
    relation='exact_text_agreement' if agree else 'different_readings' if matched else 'new_detection'
    ref='PR'+hashlib.sha256(json.dumps([plan['source_identity'],job['job_id'],occurrence['frame_id'],n,t['text']],sort_keys=True).encode()).hexdigest()[:18]
    row={'ocr_id':ref,'source_identity':plan['source_identity'],'engine':'PP-OCRv6_medium','frame_id':occurrence['frame_id'],'time':occurrence['time'],'text':t['text'],'confidence':t['confidence'],'bbox_top_left':box,'bbox_bottom_left':[box[0]/width,1-box[3]/height,(box[2]-box[0])/width,(box[3]-box[1])/height],'source_image':occurrence['source_image'],'source_image_sha256':occurrence['source_image_sha256'],'crop_image':job['image'],'job_id':job['job_id'],'receipt_sha256':digest(rp),'apple_readings':matched,'relation':relation,'trigger_reasons':occurrence['reasons'],'review_status':'unverified_ocr_evidence','basis':'onscreen_text_not_current_control_state','time_scope':'this_exact_frame_only'}
    records.append(row)
    if relation!='exact_text_agreement' and reasons(t['text'],t['confidence']):queue.append({'ocr_id':ref,'frame_id':row['frame_id'],'time':row['time'],'relation':relation,'paddle_text':t['text'],'apple_texts':[x['text'] for x in matched],'status':'pending_source_review'})
 write_jsonl(r/'sources/ocr.refined.jsonl',records);write_jsonl(r/'sources/ocr.conflicts.jsonl',queue)
 dump(r/'sources/refinement-summary.json',{'lines':len(records),'relations':{k:sum(x['relation']==k for x in records) for k in ['exact_text_agreement','different_readings','new_detection']},'queued_critical_lines':len(queue),'failed_regions':failed,'no_texts_auto_marked_correct':True,'ocr_frames':len({x['frame_id'] for x in records}),'provenance':'only representative actual frames; exact-pixel cache reused with perframe mapping'})
 if failed:raise SystemExit('missing region evidence')
 print('refined_lines',len(records),'critical_conflicts',len(queue))
if __name__=='__main__':main()
