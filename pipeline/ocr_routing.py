"""Deterministic critical-text routing; all results remain fallible source evidence."""
import re,math,hashlib,json
VERSION='critical-ocr-v2'
def reasons(text,confidence):
 r=[]
 if re.search(r'[A-Za-z][A-Za-z0-9_]*(?:[/\\][A-Za-z0-9_.-]+){2,}',text):r.append('configuration_or_path')
 if re.search(r'[A-Za-z0-9_-]+\.(?:gd|tscn|py|js|ts|json|yaml|toml|ini|png|jpg|svg|ttf|mp4|wav)\b',text,re.I):r.append('resource_or_filename')
 if re.search(r'(?:\b(?:bool|true|false|null)\b|[=:：]\s*[-+]?\d|\d+(?:\.\d+)?\s*(?:%|px|ms|fps|GB|MB|秒|毫秒))',text,re.I):r.append('typed_value_or_numeric_parameter')
 if re.search(r'注意|如果|默认|性能|生效|无效|设置|选项|参数',text) and len(text)>8:r.append('help_or_setting_statement')
 if re.search(r'\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b',text):r.append('compound_identifier')
 if re.search(r'DEBUG|ERROR|Exception',text):r.append('runtime_title_or_error')
 if re.search(r'运行|启用|禁用|取消|允许|保持|适应|输出',text) and 3<=len(text)<=26:r.append('control_label')
 if r and confidence<.6:r.append('low_confidence_critical_text')
 return r

def union(a,b):return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
def near(a,b):
 return min(a[2],b[2])-max(a[0],b[0])>=-16 and min(a[3],b[3])-max(a[1],b[1])>=-18

def propose(texts,width,height):
 seeds=[]
 for i,t in enumerate(texts):
  why=reasons(t['text'],t['confidence'])
  if why:
   a=t['bbox'];seeds.append({'bbox':[max(0,math.floor(a[0])-20),max(0,math.floor(a[1])-12),min(width,math.ceil(a[2])+20),min(height,math.ceil(a[3])+12)],'trigger_indices':[i],'reasons':why})
 # Include complete adjacent OCR rows once, based on the original seed only.
 # Avoid cutting a neighboring line in half; do not flood-fill the whole screen.
 for seed in seeds:
  seed_box=seed['bbox'][:]
  for t in texts:
   box=t['bbox'];ix=min(seed_box[2],box[2])-max(seed_box[0],box[0]);gap=max(box[1]-seed_box[3],seed_box[1]-box[3],0)
   if ix>0 and gap<=8:
    seed['bbox']=union(seed['bbox'],[max(0,math.floor(box[0])-20),max(0,math.floor(box[1])-12),min(width,math.ceil(box[2])+20),min(height,math.ceil(box[3])+12)])
 # Merge nearby text into readable panels; never cap or silently discard candidates.
 changed=True
 while changed:
  changed=False
  for i in range(len(seeds)):
   for j in range(i+1,len(seeds)):
    if near(seeds[i]['bbox'],seeds[j]['bbox']):
     a,b=seeds[i],seeds.pop(j);a['bbox']=union(a['bbox'],b['bbox']);a['trigger_indices']+=b['trigger_indices'];a['reasons']=sorted(set(a['reasons']+b['reasons']));changed=True;break
   if changed:break
 return sorted(seeds,key=lambda x:(x['bbox'][1],x['bbox'][0]))
def from_native(row):
 w,h=row['width'],row['height'];texts=[]
 for t in row['texts']:
  x,y,bw,bh=t['bbox_bottom_left'];texts.append({'text':t['text'],'confidence':t['confidence'],'bbox':[x*w,(1-y-bh)*h,(x+bw)*w,(1-y)*h]})
 return texts

def select_anchors(rows):
 buckets={}
 for row in rows:
  ts=from_native(row);critical=[t for t in ts if reasons(t['text'],t['confidence'])]
  # Count and confidence use source OCR only. Temporal sampling is supplemental OCR, not full-frame VLM replacement.
  score=(len(critical),sum(t['confidence'] for t in critical),-row['index'])
  second=int(row['part_time'])
  if second not in buckets or score>buckets[second][0]:buckets[second]=(score,row)
 return [buckets[k][1] for k in sorted(buckets)]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read_jsonl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write_jsonl(p,rs):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs))
