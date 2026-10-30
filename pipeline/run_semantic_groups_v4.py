"""Supplemental complete-sentence groups; no gold labels enter inference."""
import argparse,json,os,re,time
from pathlib import Path
from fact_ledger import hash_object
from ocr_routing import dump,digest,read_jsonl
from run_video_understanding_v4 import SEMANTIC_PROMPT,KINDS,validate_semantic


def sentence_groups(packet):
    groups=[];pending=[]
    for s in packet['speech']:
        pending.append(s)
        if re.search(r'[。！？!?]$',s['text'].strip()):
            groups.append(pending);pending=[]
    if pending:groups.append(pending)
    return [{'ref':packet['packet_id']+f'G{i:02}','text':''.join(s['text'] for s in rows),
             'segment_refs':[s['ref'] for s in rows]} for i,rows in enumerate(groups)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=root/'sentence-groups';out.mkdir(exist_ok=True)
    source=root/'sources/semantic.public.jsonl'
    protocol={'scope':'supplemental full punctuation sentence groups; fixed source-heading topics; not automatic semantic task discovery',
          'reason':'ASR segments often end in commas; supplement original segment-unit comparison with groups joined through sentence end',
          'input_sha256':digest(source),'runner_sha256':digest(Path(__file__)),
          'model_revision':json.loads((a.model/'pilot-download-receipt.json').read_text())['revision'],
          'instruction':SEMANTIC_PROMPT,'temperature':0,'max_tokens':1200,'gold_labels_not_loaded':True}
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('supplement identity changed')
    dump(path,protocol)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HOME']=str(a.model.resolve().parent.parent/'cache/huggingface')
    from mlx_vlm import load,generate
    from mlx_vlm.structured import build_json_schema_logits_processor
    import mlx.core as mx
    model,processor=load(str(a.model.resolve()),trust_remote_code=False)
    for packet in read_jsonl(source):
        units=sentence_groups(packet);refs=[u['ref'] for u in units];dest=out/(packet['packet_id']+'.json')
        if dest.exists():continue
        payload={'subject':packet['subject'],'evidence':[{'ref':u['ref'],'text':u['text']} for u in units]}
        schema={'type':'object','properties':{'fields':{'type':'object','properties':{
                k:{'type':'array','items':{'type':'string','enum':refs},'maxItems':len(refs)} for k in KINDS},
                'required':KINDS,'additionalProperties':False}},'required':['fields'],'additionalProperties':False}
        prompt=processor.apply_chat_template([{'role':'user','content':SEMANTIC_PROMPT+'\n'+json.dumps(payload,ensure_ascii=False)}],tokenize=False,add_generation_prompt=True)
        tick=time.monotonic();res=generate(model,processor,prompt=prompt,temperature=0,max_tokens=1200,verbose=False,
             logits_processors=[build_json_schema_logits_processor(processor.tokenizer,schema)])
        try:
            if res.finish_reason=='length':raise ValueError('truncated')
            output=validate_semantic(json.loads(res.text),units);error=None
        except (ValueError,TypeError,KeyError) as exc:output=None;error=str(exc)
        dump(dest,{'protocol_sha256':hash_object(protocol),'input':payload,'source_group_mapping':units,
                    'raw_output':res.text,'output':output,'finish_reason':res.finish_reason,'error':error,'wall_seconds':time.monotonic()-tick})
        print(packet['packet_id'],res.finish_reason,flush=True);mx.clear_cache()
if __name__=='__main__':main()
