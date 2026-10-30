from pathlib import Path
import argparse,json,hashlib,time
import av
from PIL import Image
from ocr_routing import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();start=time.monotonic()
 (a.out/'evidence').mkdir(parents=True,exist_ok=True)
 raw=read_jsonl(a.baseline/'sources/ocr.jsonl');manifest=json.loads((a.baseline/'sources/manifest.json').read_text());assert digest(a.video)==manifest['source_video_sha256']
 anchors=select_anchors(raw);b=json.loads((a.benchmark/'sources/benchmark.json').read_text());selected={r['index']:{'native':r,'kinds':['timeline_anchor']} for r in anchors}
 # Additional regression frames get automatic regions from prior Apple FULL outputs, not gold/crop coordinates.
 for f in b['frames']:
  i=f['index'];selected.setdefault(i,{'native':raw[i],'kinds':[]})['kinds'].append('regression_frame')
  selected[i]['benchmark_frame']=f;selected[i]['apple_record']=json.loads((a.benchmark/'runs/apple-vision'/f'{f["frame_id"]}-full.json').read_text())
 jobs={};records=[]
 for i,frame in enumerate(av.open(str(a.video)).decode(video=0)):
  if i not in selected:continue
  entry=selected[i];row=entry['native'];assert abs(float(frame.pts*frame.time_base)-row['part_time'])<1/16000
  im=frame.to_image().convert('RGB');texts=entry['apple_record']['texts'] if 'apple_record' in entry else from_native(row)
  source=a.out/'evidence'/f'{row["frame_id"]}.png';im.save(source)
  rs=[]
  for j,region in enumerate(propose(texts,im.width,im.height)):
   crop=im.crop(region['bbox']);pixel_id=hashlib.sha256(str(crop.size).encode()+crop.tobytes()).hexdigest();job_id='R'+pixel_id[:20];cp=a.out/'evidence'/f'{job_id}.png'
   if not cp.exists():crop.save(cp)
   ref={'frame_id':row['frame_id'],'frame_index':i,'time':row['part_time'],'bbox':region['bbox'],'reasons':region['reasons'],'apple_trigger_texts':[texts[k] for k in region['trigger_indices']],'source_image':str(source.relative_to(a.out)),'source_image_sha256':digest(source),'selection_kinds':entry['kinds']}
   job=jobs.setdefault(job_id,{'job_id':job_id,'image':str(cp.relative_to(a.out)),'image_sha256':digest(cp),'pixel_sha256':pixel_id,'size':crop.size,'occurrences':[]});job['occurrences'].append(ref);rs.append(job_id)
  records.append({'frame_id':row['frame_id'],'time':row['part_time'],'selection_kinds':entry['kinds'],'regions':rs,'apple_text_count':len(texts)})
 plan={'version':VERSION,'source_identity':{k:manifest[k] for k in ['bvid','cid','source_video_sha256']},'source_ocr_sha256':digest(a.baseline/'sources/ocr.jsonl'),'router_sha256':digest(Path(__file__).with_name('ocr_routing.py')),'source_frames_scanned':len(raw),'timeline_anchors':len(anchors),'selected_frames':len(selected),'regression_frames':len(b['frames']),'jobs':list(jobs.values()),'selected':records,'selection_policy':'one supplementary representative persecond selected by critical-text count/confidence plus24 regression frames; no claim all sourceframes receive Paddle','propagation_policy':'only exact crop RGB+dimensions may share OCR; every actual frame/time recorded separately; never extend to interval','gold_used_for_routing':False,'prepare_seconds':time.monotonic()-start}
 dump(a.out/'sources/route-plan.json',plan);print(json.dumps({k:v for k,v in plan.items() if k not in ['jobs','selected']},ensure_ascii=False));print('unique_jobs',len(jobs),'region_occurrences',sum(len(j['occurrences']) for j in jobs.values()))
if __name__=='__main__':main()
