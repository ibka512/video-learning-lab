"""Fixed regression of former failures and expected fields; does not score general recall."""
from pathlib import Path
import argparse,json
from fact_ledger import validate_fact,hash_object
from ocr_routing import read_jsonl,dump,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--previous',required=True,type=Path);a=p.parse_args();r=a.root.resolve();old=a.previous.resolve();manifest=json.loads((r/'sources/input-manifest.json').read_text());raw=read_jsonl(r/'sources/facts.jsonl');final=read_jsonl(r/'sources/facts.completed.jsonl');exported=read_jsonl(r/'sources/facts.exported.jsonl');old_record=json.loads((old/'knowledge-generation/SC0004-page-000.json').read_text());ctx=old_record['context'];aliases=old_record['canonical_ref_aliases']
    old_evidence=[]
    for ref,text,*_ in ctx['ocr_rows']:old_evidence.append({'ref':aliases[ref],'modality':'ocr','text':text})
    for e in ctx['verified_source_events']:old_evidence.append({'ref':e['event_id'],'modality':'visual_summary','text':e['source_review_summary']})
    for e in [ctx['source_interval']]+ctx['neighbor_speech']:old_evidence.append({'ref':e['segment_id'],'modality':'speech','text':e['text']})
    index={e['ref']:e for e in old_evidence};regressions=[]
    for c in read_jsonl(old/'sources/knowledge.candidates.jsonl'):
        fact={'kind':'author_statement','value':c['text'],'citations':[{'ref':ref,'quote':index[ref]['text']} for ref in c['evidence_refs'] if ref in index][:6]}
        packet={'allowed_kinds':['author_statement'],'subject':None,'evidence':old_evidence}
        errors,_=validate_fact(fact,packet);regressions.append({'candidate_id':c['candidate_id'],'original_fact_sha256':hash_object(c),'rejected':bool(errors),'errors':errors,'mapping':'old author_statement -> full quote references; no synthetic replacement of original claim'})
    expected=[]
    for pid,sha in manifest['packet_files'].items():
        packet=json.loads((r/'packets'/(pid+'.json')).read_text())
        expected += [(pid,k) for k in packet['required_kinds']]
    def coverage(fs):
        pairs={(f['packet_id'],f['kind']) for f in fs};return {'required_source_fields':len(expected),'present':sum(x in pairs for x in expected),'missing':[x for x in expected if x not in pairs]}
    # Independent exact-location audit, using saved packet spans rather than trusting fact IDs.
    for f in final:
        packet=json.loads((r/'packets'/(f['packet_id']+'.json')).read_text());evidence={e['ref']:e for e in packet['evidence']}
        for span in f['source_spans']:
            e=evidence[span['ref']];assert e['text'][span['char_start']:span['char_end']]==span['quote']
        for loc in f['source_locations']:
            e=evidence[loc['ref']]
            for k in loc:assert loc[k]==e[k],(f['fact_id'],k)
    # Publication is reviewed output, not just model schema success.
    assert all(f['review_status']=='source_supported_quote' or f.get('export_review_basis') for f in exported)
    dump(r/'sources/regression-evaluation.json',{'previous_candidates':regressions,'model_only_field_coverage':coverage(raw),'with_literal_recovery_field_coverage':coverage(final),'exported_field_coverage':coverage(exported),'exact_spans_and_locations_passed':True,'source_unit_coverage_not_full_video_knowledge_recall':True,'old_candidates_and_model_input_sha256':digest(old/'knowledge-generation/SC0004-page-000.json'),'new_independent_video_tested':False,'code_sha256':digest(Path(__file__))})
    print('previous rejected',sum(x['rejected'] for x in regressions),'/',len(regressions),'fields',coverage(final)['present'],'/',len(expected))
if __name__=='__main__':main()
