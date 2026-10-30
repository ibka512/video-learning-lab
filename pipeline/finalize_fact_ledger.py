"""Deterministic recovery for literal source fields only; no semantic invention."""
from pathlib import Path
import argparse,json
from fact_ledger import HEADER,evaluate_response,hash_object
from ocr_routing import dump,digest,read_jsonl,write_jsonl

def literal_candidates(packet):
    candidates=[]
    for e in packet['evidence']:
        def add(kind,value):candidates.append({'kind':kind,'value':value,'citations':[{'ref':e['ref'],'quote':e['text']}]})
        if e['modality']=='ocr_help' and e.get('role')=='header':
            m=HEADER.search(e['text'].strip())
            if m and m.group(1)==packet['subject']:
                add('setting_key',m.group(1));add('help_default',m.group(3).strip())
        elif e['modality']=='ocr_help' and e.get('role')=='body':add('help_statement',e['text'])
        elif e['modality']=='speech':add('author_statement',e['text'])
        elif e['modality']=='reviewed_state':add('observed_state',e['observed_value'])
    return candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();manifest=json.loads((r/'sources/input-manifest.json').read_text());model_facts=read_jsonl(r/'sources/facts.jsonl');all_facts=[];log=[]
    # Completion deliberately refuses a partial model run masquerading as whole-video work.
    metrics=json.loads((r/'sources/extraction-metrics.json').read_text())
    if not metrics['all_manifest_packets_processed']:raise ValueError('finish all requested source packets first')
    for pid,sha in manifest['packet_files'].items():
        path=r/'packets'/(pid+'.json');assert digest(path)==sha;packet=json.loads(path.read_text());seen=set()
        for f in model_facts:
            if f['packet_id']!=pid:continue
            original={k:f[k] for k in ('kind','value','citations')};valid,invalid,_=evaluate_response({'facts':[original],'overflow':False},packet)
            if invalid:raise ValueError('stored model fact no longer validates')
            f={**valid[0],'origin':'model_extraction'};seen.add((f['kind'],f['value']));all_facts.append(f)
        added=[];blocked=[]
        for c in literal_candidates(packet):
            if (c['kind'],c['value']) in seen:continue
            valid,invalid,_=evaluate_response({'facts':[c],'overflow':False},packet)
            if valid:
                f={**valid[0],'origin':'deterministic_literal_recovery','recovery_reason':'bounded model attempts did not return this source-derived field'};all_facts.append(f);seen.add((f['kind'],f['value']));added.append(f['fact_id'])
            else:blocked+=invalid
        missing=sorted(set(packet['required_kinds'])-{k for k,_ in seen});log.append({'packet_id':pid,'literal_recovered_fact_ids':added,'blocked':blocked,'missing_kinds':missing})
    write_jsonl(r/'sources/facts.completed.jsonl',all_facts);dump(r/'sources/literal-recovery.json',{'facts':len(all_facts),'literal_recovered':sum(f['origin']=='deterministic_literal_recovery' for f in all_facts),'packets':log,'source_truth_not_inferred':True,'code_sha256':digest(Path(__file__))})
if __name__=='__main__':main()
