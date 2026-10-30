import copy,unittest
from fact_ledger import validate_fact,evaluate_response

class FactLedgerTests(unittest.TestCase):
    def setUp(self):
        self.key='editor/window/transparent'
        self.header={'ref':'H','modality':'ocr_help','role':'header','subject':self.key,'text':f'设置 {self.key}: bool = false','review_status':'unverified_ocr','start':18.,'end':18.,'frame_id':'F540'}
        self.body={'ref':'B','modality':'ocr_help','role':'body','subject':self.key,'text':'这会降低性能，因此除非需要，请保留\n为false。','review_status':'unverified_ocr','start':18.,'end':18.,'frame_id':'F540'}
        self.p={'packet_id':'P','source_identity':{'bvid':'TEST'},'subject':self.key,'allowed_kinds':['setting_key','help_default','help_statement'],'required_kinds':['setting_key','help_default','help_statement'],'evidence':[self.header,self.body]}
    def fact(self,kind,value,ref='H',quote=None):return {'kind':kind,'value':value,'citations':[{'ref':ref,'quote':quote if quote is not None else self.header['text']}]}
    def test_default_is_bound_to_subject(self):
        d=self.fact('help_default','false');self.assertEqual(validate_fact(d,self.p)[0],[])
        self.p['subject']='editor/window/opacity';self.assertIn('default_not_bound_to_exact_header',validate_fact(d,self.p)[0])
    def test_no_false_typo_repair(self):
        self.header['text']=self.header['text'].replace('false','fal.se');self.assertTrue(validate_fact(self.fact('help_default','false'),self.p)[0])
    def test_exact_path_case(self):self.assertTrue(validate_fact(self.fact('setting_key','Editor/window/transparent'),self.p)[0])
    def test_complete_help_no_missing_false(self):
        d=self.fact('help_statement','这会降低性能，因此除非需要，请保留','B','这会降低性能，因此除非需要，请保留');self.assertIn('help_body_must_be_complete_verbatim',validate_fact(d,self.p)[0])
    def test_ocr_cannot_prove_state(self):
        self.p['allowed_kinds'].append('observed_state');self.assertIn('state_requires_object_specific_reviewed_frame',validate_fact(self.fact('observed_state','false'),self.p)[0])
    def test_ocr_cannot_be_author(self):
        self.p['allowed_kinds'].append('author_statement');self.assertIn('author_statement_requires_speech',validate_fact(self.fact('author_statement',self.header['text']),self.p)[0])
    def test_valid_state_needs_specific_review(self):
        e={'ref':'V','modality':'reviewed_state','subject':self.key,'observed_value':'true','text':'此帧对应复选框已勾选。','source_image_sha256':'abc','frame_id':'F540','start':18.,'end':18.,'review_status':'source_reviewed'}
        p={**self.p,'allowed_kinds':['observed_state'],'required_kinds':['observed_state'],'evidence':[e]};d=self.fact('observed_state','true','V',e['text']);self.assertEqual(validate_fact(d,p)[0],[])
        p['subject']='other/window/path';self.assertTrue(validate_fact(d,p)[0])
    def test_missing_quote_or_id(self):
        self.assertTrue(validate_fact(self.fact('help_default','false',quote='编造证据'),self.p)[0]);self.assertTrue(validate_fact(self.fact('help_default','false','NOT-EXIST'),self.p)[0])
    def test_repeated_quote_requires_disambiguation(self):
        self.header['text']='false false';self.assertTrue(validate_fact(self.fact('help_default','false',quote='false'),self.p)[0])
    def test_empty_output_is_not_complete(self):
        valid,invalid,coverage=evaluate_response({'facts':[],'overflow':False},self.p);self.assertEqual(len(coverage['missing_kinds']),3)
    def test_duplicates_flagged(self):
        d=self.fact('help_default','false');v,i,c=evaluate_response({'facts':[d,d],'overflow':False},self.p);self.assertEqual(len(v),1);self.assertIn('duplicate_fact',i[0]['errors'])
    def test_rule_pass_does_not_approve_ocr(self):
        v,_,_=evaluate_response({'facts':[self.fact('help_default','false')],'overflow':False},self.p);self.assertEqual(v[0]['review_status'],'pending_source_review');span=v[0]['source_spans'][0];self.assertEqual(self.header['text'][span['char_start']:span['char_end']],span['quote'])
    def test_plain_speech_is_preserved(self):
        e={'ref':'S','modality':'speech','text':'重新启动后，再调整大小。','start':2.,'end':4.,'review_status':'source_reviewed'};p={**self.p,'subject':None,'allowed_kinds':['author_statement'],'required_kinds':['author_statement'],'evidence':[e]};v,i,_=evaluate_response({'facts':[self.fact('author_statement',e['text'],'S',e['text'])],'overflow':False},p);self.assertFalse(i);self.assertEqual(v[0]['review_status'],'source_supported_quote')
    def test_response_schema_failure_recorded(self):self.assertTrue(evaluate_response(None,self.p)[1])

class RecoveryAndExportTests(unittest.TestCase):
    def test_literal_recovery_keeps_ocr_typo_blocked(self):
        from finalize_fact_ledger import literal_candidates
        p={'packet_id':'P','source_identity':{},'subject':'tool/window/size','allowed_kinds':['help_default','setting_key'],'required_kinds':['help_default','setting_key'],'evidence':[{'ref':'H','text':'设置 tool/window/size: bool = fal.se','modality':'ocr_help','role':'header','subject':'tool/window/size','review_status':'unverified_ocr'}]}
        candidates=literal_candidates(p);self.assertEqual(next(c['value'] for c in candidates if c['kind']=='help_default'),'fal.se');valid,invalid,_=evaluate_response({'facts':candidates,'overflow':False},p);self.assertTrue(invalid);self.assertNotIn('help_default',[f['kind'] for f in valid])
    def test_export_requires_explicit_review_for_ocr(self):
        from export_fact_markdown import reviewed_facts
        f={'fact_id':'K','review_status':'pending_source_review'};a,p=reviewed_facts([f],{});self.assertFalse(a);self.assertEqual(len(p),1)
    def test_stale_review_cannot_approve_changed_fact(self):
        from export_fact_markdown import reviewed_facts
        with self.assertRaises(ValueError):reviewed_facts([{'fact_id':'K','review_status':'pending_source_review'}],{'K':{'fact_sha256':'old','status':'approve'}})
    def test_rejection_overrides_existing_source_status(self):
        from export_fact_markdown import reviewed_facts
        from fact_ledger import hash_object
        f={'fact_id':'K','review_status':'source_supported_quote'};a,p=reviewed_facts([f],{'K':{'fact_sha256':hash_object(f),'status':'reject','reason':'bad'}});self.assertFalse(a);self.assertEqual(p[0]['export_block_reason'],'bad')



class PacketProvenanceTests(unittest.TestCase):
    def fixtures(self,root):
        import json
        from pathlib import Path
        base=root/'base';ocr=root/'ocr';out=root/'out'
        (base/'sources').mkdir(parents=True);(ocr/'sources').mkdir(parents=True)
        identity={'bvid':'V1','cid':1,'source_video_sha256':'sha-video'}
        (base/'sources/manifest.json').write_text(json.dumps({**identity,'title':'test','source_url':'https://example.invalid'}))
        (base/'sources/transcript.corrected.segments.jsonl').write_text(json.dumps({'segment_id':'S1','text':'先打开菜单。','start':0,'end':1})+'\n')
        (ocr/'sources/ocr.refined.jsonl').write_text('')
        return base,ocr,out,identity
    def run_prepare(self,base,ocr,out):
        import subprocess,sys
        from pathlib import Path
        return subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_fact_packets.py')),'--baseline',str(base),'--ocr',str(ocr),'--out',str(out),'--domain','test'],capture_output=True,text=True)
    def test_changed_evidence_does_not_overwrite_cached_packet(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            base,ocr,out,_=self.fixtures(Path(d));self.assertEqual(self.run_prepare(base,ocr,out).returncode,0)
            packet=out/'packets/S1.json';before=packet.read_bytes()
            (base/'sources/transcript.corrected.segments.jsonl').write_text(json.dumps({'segment_id':'S1','text':'完全不同的指令。','start':0,'end':1})+'\n')
            self.assertNotEqual(self.run_prepare(base,ocr,out).returncode,0);self.assertEqual(packet.read_bytes(),before)
    def test_other_video_ocr_rejected(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            base,ocr,out,identity=self.fixtures(Path(d))
            (ocr/'sources/ocr.refined.jsonl').write_text(json.dumps({'source_identity':{**identity,'bvid':'V2'},'frame_id':'F1'})+'\n')
            result=self.run_prepare(base,ocr,out);self.assertNotEqual(result.returncode,0);self.assertIn('different video',result.stderr)

if __name__=='__main__':unittest.main()
