"""Small source units with exact offsets; full OCR remains a linked, hashed attachment."""
from pathlib import Path
import argparse,json,re
from collections import defaultdict
from ocr_routing import read_jsonl,dump,digest,write_jsonl
from fact_ledger import HEADER,hash_object

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--ocr',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--domain',required=True);p.add_argument('--state-reviews',type=Path);a=p.parse_args();out=a.out.resolve();base=a.baseline.resolve();ocr=a.ocr.resolve()
    manifest=json.loads((base/'sources/manifest.json').read_text());identity={k:manifest[k] for k in ['bvid','cid','source_video_sha256']};rows=read_jsonl(ocr/'sources/ocr.refined.jsonl');frames=defaultdict(list);packets=[];all_blocks=[];selected_refs=set()
    for row in rows:
        if row['source_identity']!=identity:raise ValueError('OCR belongs to a different video')
        frames[row['frame_id']].append(row)
    def location(row):return {'start':row['time'],'end':row['time'],'frame_id':row['frame_id'],'bbox':row['bbox_top_left'],'source_image':str(ocr/row['source_image']),'source_image_sha256':row['source_image_sha256'],'review_status':'unverified_ocr'}
    for fid,rs in frames.items():
        for header in rs:
            m=HEADER.search(header['text'].strip())
            if not m:continue
            x,y,x1,y1=header['bbox_top_left'];body=[];last=y1
            for r in sorted(rs,key=lambda r:(r['bbox_top_left'][1],r['bbox_top_left'][0])):
                b=r['bbox_top_left']
                if b[1]<y1 or abs(b[0]-x)>35 or b[1]-y1>180:continue
                if b[1]-last>28:break
                body.append(r);last=b[3]
            if not body:continue
            subject=m.group(1);body_text='\n'.join(r['text'] for r in body);block={'subject':subject,'header':header,'body':body,'text':body_text,'valid_default':m.group(2)!='bool' or m.group(3).strip() in ['true','false']}
            all_blocks.append(block)
    # Preserve every block in an index; choose one actual frame per setting for this pilot.
    # Longest body, then confidence, then earlier time. No benchmark answer used.
    grouped=defaultdict(list)
    for block in all_blocks:grouped[block['subject']].append(block)
    for subject,blocks in sorted(grouped.items()):
        block=max(blocks,key=lambda b:(b['valid_default'],len(b['text']),b['header']['confidence'],-b['header']['time']))
        h=block['header'];body=block['body'];selected_refs.update([h['ocr_id']]+[r['ocr_id'] for r in body]);ref='B'+hash_object([r['ocr_id'] for r in body])[:18]
        packet={'packet_id':'HELP-'+hash_object(subject)[:10],'source_identity':identity,'subject':subject,'allowed_kinds':['setting_key','help_default','help_statement'],'required_kinds':['setting_key','help_default','help_statement'],
                'evidence':[{'ref':h['ocr_id'],'modality':'ocr_help','role':'header','subject':subject,'text':h['text'],**location(h)},
                            {'ref':ref,'modality':'ocr_help','role':'body','subject':subject,'text':block['text'],'member_refs':[r['ocr_id'] for r in body],**location(body[0]),'bbox':[min(r['bbox_top_left'][0] for r in body),min(r['bbox_top_left'][1] for r in body),max(r['bbox_top_left'][2] for r in body),max(r['bbox_top_left'][3] for r in body)]}],
                'binding_basis':'same_frame_header_and_adjacent_lines_not_independent_semantic_review'}
        packets.append(packet)
    speech=read_jsonl(base/'sources/transcript.corrected.segments.jsonl')
    for s in speech:
        packets.append({'packet_id':s['segment_id'],'source_identity':identity,'subject':None,'allowed_kinds':['author_statement'],'required_kinds':['author_statement'],
                        'evidence':[{'ref':s['segment_id'],'modality':'speech','text':s['text'],'start':s['start'],'end':s['end'],'review_status':'source_reviewed','review_basis':'existing_source_corrected_transcript_not_independent_listening'}]})
    if a.state_reviews:
        for s in read_jsonl(a.state_reviews):
            assert s['source_identity']==identity and digest(Path(s['source_image']))==s['source_image_sha256']
            if s.get('review_artifact'):assert digest(Path(s['review_artifact']))==s['review_artifact_sha256']
            packets.append({'packet_id':s['ref'],'source_identity':identity,'subject':s['subject'],'allowed_kinds':['observed_state'],'required_kinds':['observed_state'],'evidence':[s]})
    existing=out/'sources/input-manifest.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if set(old['packet_files'])!={p['packet_id'] for p in packets}:raise ValueError('changed packet scope: use a new output directory')
        for packet in packets:
            if json.loads((out/'packets'/(packet['packet_id']+'.json')).read_text())!=packet:raise ValueError('changed evidence: use a new output directory')
    for p in packets:dump(out/'packets'/(p['packet_id']+'.json'),p)
    write_jsonl(out/'sources/help-block-index.jsonl',[{'subject':b['subject'],'frame_id':b['header']['frame_id'],'header_ref':b['header']['ocr_id'],'body_refs':[r['ocr_id'] for r in b['body']],'text':b['text'],'valid_default':b['valid_default']} for b in all_blocks])
    dump(out/'sources/input-manifest.json',{'source_identity':identity,'title':manifest['title'],'source_url':manifest['source_url'],'domain':a.domain,'packets':len(packets),'help_packets':len(grouped),'speech_packets':len(speech),'ocr_rows':len(rows),'selected_ocr_refs':sorted(selected_refs),'all_ocr_attachment':str(ocr/'sources/ocr.refined.jsonl'),'all_ocr_sha256':digest(ocr/'sources/ocr.refined.jsonl'),'baseline':str(base),'baseline_manifest_sha256':digest(base/'sources/manifest.json'),'source_scope':'all timed speech segments, one representative help panel per detected setting, explicit reviewed states; NOT whole-video knowledge recall','excluded_ocr_is_preserved_not_claimed_processed':True,'packet_files':{p['packet_id']:digest(out/'packets'/(p['packet_id']+'.json')) for p in packets},'prepare_code_sha256':digest(Path(__file__))})
    print('packets',len(packets),'help subjects',len(grouped),'help blocks',len(all_blocks))
if __name__=='__main__':main()
