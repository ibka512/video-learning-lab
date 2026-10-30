import sys,json
from pathlib import Path
sys.path.insert(0,'pipeline')
import av
from PIL import Image,ImageDraw
from ocr_routing import digest,dump,read_jsonl,write_jsonl
root=Path('outputs/声画理解对照-v4-2026-10-29')
video=Path('work/pilot-20261026/media/video.m4s')
manifest=json.loads(Path('outputs/第二轮视频验证-2026-10-27/sources/manifest.json').read_text())
assert digest(video)==manifest['source_video_sha256']
specs=[('W',[21,22,23,24],[400,130,1805,905]),('B',[28,29,40,41],[400,130,1805,905]),
       ('F',[52,52.5,53,54],[400,130,1805,905]),('T',[57,58,59,60],[400,130,1805,905]),
       ('M',[30,34,38,40],[1490,150,1805,380]),('C',[4.5,5,6,7],[400,130,1805,905])]
wanted={round(t*30) for _,times,_ in specs for t in times};frames={};assets=[]
with av.open(str(video)) as c:
 for i,f in enumerate(c.decode(video=0)):
  if i in wanted:
   im=f.to_image();frames[i]=im
   path=root/'evidence'/f'F{i:07}.png';path.parent.mkdir(exist_ok=True);im.save(path)
   assets.append({'frame_id':f'F{i:07}','frame_index':i,'pts':f.pts,'time_base':str(f.time_base),'time':float(f.pts*f.time_base),
                  'source_image':str(path.resolve()),'source_image_sha256':digest(path)})
rows=[]
for cid,times,box in specs:
 ims=[];inputs=[]
 for n,t in enumerate(times):
  i=round(t*30);source=next(a for a in assets if a['frame_index']==i)
  crop=frames[i].crop(box);path=root/'evidence'/f'{cid}-{n}.png';crop.save(path)
  inputs.append({**source,'bbox_top_left':box,'crop':str(path.resolve()),'crop_sha256':digest(path)})
  thumb=crop.copy();thumb.thumbnail((600,340))
  tile=Image.new('RGB',(610,370),'white');tile.paste(thumb,(0,25));ImageDraw.Draw(tile).text((5,5),f'{cid} position {n} source {i} {t}s',fill='black');ims.append(tile)
 sheet=Image.new('RGB',(1220,740),'white')
 for n,im in enumerate(ims):sheet.paste(im,((n%2)*610,(n//2)*370))
 sheet.save(root/'evidence'/f'{cid}-review.png')
 rows.append({'case_id':cid,'frames':inputs,'source_interval':[min(times),max(times)]})
write_jsonl(root/'sources/frames.jsonl',assets)
write_jsonl(root/'sources/cases.assets.jsonl',rows)
speech=read_jsonl(Path('outputs/步骤完整性-v3-2026-10-29/r4/BV1cyHR67EbJ/sources/speech.jsonl'))
write_jsonl(root/'sources/speech.jsonl',speech)
dump(root/'sources/source-identity.json',{'source_identity':{k:manifest[k] for k in ['bvid','cid','source_video_sha256']},
 'video':str(video.resolve()),'source_file_sha256':digest(video),'selection':'Codex selected development-regression events, not automatic event discovery',
 'full_frame_semantic_coverage_claimed':False,'speech_sha256':digest(root/'sources/speech.jsonl')})
