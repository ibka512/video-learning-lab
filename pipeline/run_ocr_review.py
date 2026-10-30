"""Paddle worker; content-identified local cache, checkpointed one region at a time."""
from pathlib import Path
import argparse,os,json,time,hashlib,resource
from ocr_routing import digest,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args();r=a.root.resolve();w=a.runtime.resolve()
 for k,v in {'PADDLE_PDX_CACHE_HOME':w/'cache/paddlex','HF_HOME':w/'cache/huggingface','MODELSCOPE_CACHE':w/'cache/modelscope','XDG_CACHE_HOME':w/'cache','TMPDIR':w/'tmp','PADDLE_HOME':w/'cache/paddle'}.items():os.environ[k]=str(v)
 os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True';os.environ['HF_HUB_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='4'
 plan=json.loads((r/'sources/route-plan.json').read_text());out=r/'receipts';out.mkdir(exist_ok=True);start=time.monotonic()
 modelroot=w/'cache/paddlex/official_models';files=sorted(p for p in modelroot.rglob('*') if p.is_file() and p.suffix in ['.pdiparams','.json','.yml']);models={str(p.relative_to(modelroot)):digest(p) for p in files}
 config=dict(text_detection_model_name='PP-OCRv6_medium_det',text_recognition_model_name='PP-OCRv6_medium_rec',text_detection_model_dir=str(modelroot/'PP-OCRv6_medium_det'),text_recognition_model_dir=str(modelroot/'PP-OCRv6_medium_rec'),use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',enable_mkldnn=False,cpu_threads=4,text_det_limit_type='max',text_det_limit_side_len=1920,text_rec_score_thresh=0.0)
 engine_id=hashlib.sha256(json.dumps({'config':config,'models':models,'code':digest(Path(__file__))},sort_keys=True).encode()).hexdigest()
 ocr=None;fresh=0;cached=0;errors=[];infer_seconds=0.;load_seconds=0.
 for n,job in enumerate(plan['jobs']):
  image=r/job['image'];assert digest(image)==job['image_sha256'];path=out/(job['job_id']+'.json');identity={'engine_id':engine_id,'image_sha256':job['image_sha256'],'pixel_sha256':job['pixel_sha256']}
  if path.exists():
   old=json.loads(path.read_text())
   if old['identity']!=identity:raise ValueError('cache identity differs '+job['job_id'])
   if old['status']=='success':cached+=1;continue
  if ocr is None:
   load_start=time.monotonic();from paddleocr import PaddleOCR
   ocr=PaddleOCR(**config);load_seconds=time.monotonic()-load_start;print(json.dumps({'event':'loaded','seconds':load_seconds}),flush=True)
  tick=time.monotonic()
  try:
   raw=list(ocr.predict(str(image)))[0].json;raw=raw.get('res',raw)
   texts=[{'text':t,'confidence':float(c),'bbox_crop':[min(z[0] for z in poly),min(z[1] for z in poly),max(z[0] for z in poly),max(z[1] for z in poly)]} for t,c,poly in zip(raw['rec_texts'],raw['rec_scores'],raw['rec_polys'])]
   record={'job_id':job['job_id'],'identity':identity,'status':'success','texts':texts,'raw':raw,'seconds':time.monotonic()-tick,'review_status':'unverified_ocr_evidence'}
  except Exception as e:
   record={'job_id':job['job_id'],'identity':identity,'status':'failed','error':type(e).__name__+': '+str(e),'seconds':time.monotonic()-tick};errors.append(job['job_id'])
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=lambda v:v.tolist())+'\n');tmp.replace(path);fresh+=1;infer_seconds+=record['seconds']
  print(json.dumps({'job':n+1,'total':len(plan['jobs']),'status':record['status'],'seconds':record['seconds']}),flush=True)
 metrics={'jobs':len(plan['jobs']),'new_requests':fresh,'cache_hits':cached,'failed':errors,'all_planned_regions_processed':not errors,'inference_seconds_this_run':infer_seconds,'import_load_seconds':load_seconds,'wall_seconds_this_run':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'engine_id':engine_id,'config':config,'model_file_hashes':models,'source_frames_are_not_all_Paddle_inputs':True}
 name='metrics.json' if fresh else 'resume-validation.json';dump(r/'sources'/name,metrics)
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
