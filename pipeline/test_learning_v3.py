import copy,unittest,json,tempfile
from pathlib import Path
from fact_ledger import hash_object
from ocr_routing import dump,write_jsonl
from run_learning_v3 import export
from learning_schema import split_units,classify,build_steps,audit_procedure,materialize_units,validate_checks
from run_learning_v3 import packets

class ProcedureContracts(unittest.TestCase):
    def setUp(self):
        speech=[{'ref':'A','text':'先开启设置：','start':1.,'end':2.},
                {'ref':'B','text':'界面—透明；','start':2.,'end':3.},
                {'ref':'C','text':'如果仍有边框，','start':3.,'end':4.},
                {'ref':'D','text':'那么关闭嵌入，','start':4.,'end':5.},
                {'ref':'E','text':'再重新启动场景即可。','start':5.,'end':6.},
                {'ref':'F','text':'窗口成为透明的了。','start':6.,'end':7.}]
        self.u=split_units(speech)
        self.a=[{'unit_id':u['unit_id'],'role':r} for u,r in zip(self.u,['action','action','condition','action','action','result'])]
        self.d={'steps':[
            dict(action_refs=['A.0'],context_refs=[],condition_refs=[],result_refs=[],after=[]),
            dict(action_refs=['B.0'],context_refs=['A.0'],condition_refs=[],result_refs=[],after=[0]),
            dict(action_refs=['D.0'],context_refs=[],condition_refs=['C.0'],result_refs=[],after=[1]),
            dict(action_refs=['E.0'],context_refs=[],condition_refs=['C.0'],result_refs=['F.0'],after=[2])]}
    def test_complete_branch_passes(self):
        self.assertEqual(build_steps(self.d,self.u,self.a),self.d)
        self.assertEqual(audit_procedure(self.u,self.a,self.d),[])
    def test_old_truncated_procedure_reports_exact_continuation(self):
        self.d['steps'].pop()
        alarms=audit_procedure(self.u,self.a,self.d)
        self.assertIn({'unit_id':'E.0','issue':'unrepresented_action'},alarms)
        with self.assertRaisesRegex(ValueError,'unrepresented'):build_steps(self.d,self.u,self.a)
    def test_omission_hiding_in_overview_is_flagged(self):
        self.a[4]['role']='overview';self.d['steps'].pop()
        self.assertIn({'unit_id':'E.0','issue':'possible_missing_continuation_action'},audit_procedure(self.u,self.a,self.d))
    def test_branch_condition_must_carry_to_restart(self):
        self.d['steps'][3]['condition_refs']=[]
        self.assertTrue(any(a['issue']=='branch_condition_not_carried' for a in audit_procedure(self.u,self.a,self.d)))
    def test_reversed_steps_rejected(self):
        self.d['steps'][2:]=reversed(self.d['steps'][2:])
        with self.assertRaises(ValueError):build_steps(self.d,self.u,self.a)
    def test_unknown_sources_rejected(self):
        self.d['steps'][1]['context_refs']=['unknown']
        with self.assertRaisesRegex(ValueError,'unknown'):build_steps(self.d,self.u,self.a)
    def test_actions_cannot_be_fabricated_from_result(self):
        self.d['steps'][3]['action_refs'].append('F.0')
        with self.assertRaisesRegex(ValueError,'action-classified'):build_steps(self.d,self.u,self.a)
    def test_cyclic_future_dependency_rejected(self):
        self.d['steps'][0]['after']=[2]
        with self.assertRaisesRegex(ValueError,'dependency'):build_steps(self.d,self.u,self.a)
    def test_duplicate_action_rejected(self):
        self.d['steps'][2]['action_refs']=['B.0']
        with self.assertRaisesRegex(ValueError,'twice'):build_steps(self.d,self.u,self.a)
    def test_missing_unit_or_wrong_order_rejected(self):
        for rows in [self.a[:-1],list(reversed(self.a))]:
            with self.assertRaisesRegex(ValueError,'every unit'):classify({'assignments':rows},self.u)
    def test_no_generated_summary_or_role(self):
        rows=copy.deepcopy(self.a);rows[0]['summary']='invented'
        with self.assertRaises(ValueError):classify({'assignments':rows},self.u)
    def test_exact_spans_preserve_quantities_negations_and_parent_timing(self):
        es=[{'ref':'X','text':'如果超过五张，不要自动播放。','start':1.,'end':9.}]
        us=split_units(es)
        self.assertEqual(''.join(u['quote'] for u in us),es[0]['text'])
        self.assertTrue(all(u['end']==9. for u in us))
        for u in us:self.assertEqual(es[0]['text'][u['char_start']:u['char_end']],u['quote'])
    def test_wide_sentence_not_duplicated_per_role(self):
        p={'units':self.u,'source_identity':{'bvid':'fixture'},'subject':'toy','packet_id':'P'}
        fs=materialize_units(p,self.a)
        self.assertEqual(len(fs),len(self.u))
        self.assertTrue(all(not f['observed_execution'] for f in fs))
    def test_auxiliary_check_cannot_skip_any_unit(self):
        with self.assertRaises(ValueError):validate_checks({'checks':[]},self.u)
    def test_explicit_chapters_include_warning_before_next_heading(self):
        es=[{'ref':str(i),'text':t,'start':float(i),'end':float(i+1)} for i,t in enumerate(
            ['第一个 Alpha 面板，','注意不要超过三项。','第二个 Beta 面板，','只放一项。'])]
        ps=packets(es,{'bvid':'toy'},'toy')
        self.assertEqual([e['ref'] for e in ps[0]['speech']],['0','1'])
    def test_long_procedure_has_no_eight_source_limit(self):
        us=[dict(unit_id=str(i),quote='执行。',start=float(i),end=float(i+1)) for i in range(12)]
        aa=[dict(unit_id=str(i),role='action') for i in range(12)]
        steps=[dict(action_refs=[str(i)],context_refs=[],condition_refs=[],result_refs=[],after=[i-1] if i else []) for i in range(12)]
        self.assertEqual(len(build_steps({'steps':steps},us,aa)['steps']),12)

class ExportContracts(unittest.TestCase):
    def fixture(self,root):
        speech=[{'ref':'A','text':'关闭窗口。','start':1.,'end':2.}]
        m={'source_identity':{'bvid':'toy'},'title':'toy','source_url':'https://example.test',
           'domain':'test','transcript_review_basis':'fixture','legacy_root':None,'source_files':{}}
        ps=packets(speech,m['source_identity'],m['title'])
        aa=[{'unit_id':'A.0','role':'action'}]
        fs=materialize_units(ps[0],aa)
        c={'packet_id':'P00','assignments':aa,'procedure':{'steps':[dict(action_refs=['A.0'],context_refs=[],condition_refs=[],result_refs=[],after=[])]},'alarms':[],'link_error':None}
        for name,data in [('input-manifest',m)]:dump(root/'sources'/(name+'.json'),data)
        for name,rows in [('speech',speech),('packets',ps),('cards',[c]),('units.classified',fs)]:
            write_jsonl(root/'sources'/(name+'.jsonl'),rows)
        return c,fs[0]
    def test_model_output_cannot_publish_without_source_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);export(root)
            self.assertEqual((root/'sources/units.exported.jsonl').read_text(),'')
            self.assertNotIn('1. 关闭窗口',(root/'knowledge/test/toy.md').read_text())
    def test_stale_review_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,f=self.fixture(root)
            write_jsonl(root/'sources/source-reviews.jsonl',[dict(fact_id=f['fact_id'],fact_sha256='old',status='approve')])
            with self.assertRaisesRegex(ValueError,'stale unit'):export(root)
    def test_unit_review_alone_cannot_publish_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,f=self.fixture(root)
            write_jsonl(root/'sources/source-reviews.jsonl',[dict(fact_id=f['fact_id'],fact_sha256=hash_object(f),status='approve')])
            export(root)
            self.assertNotIn('1. 关闭窗口',(root/'knowledge/test/toy.md').read_text())
    def test_graph_review_bound_to_exact_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,f=self.fixture(root)
            write_jsonl(root/'sources/source-reviews.jsonl',[dict(fact_id=f['fact_id'],fact_sha256=hash_object(f),status='approve')])
            write_jsonl(root/'sources/procedure-reviews.jsonl',[dict(packet_id='P00',card_sha256=hash_object(c),status='approve')])
            export(root)
            self.assertIn('1. 关闭窗口',(root/'knowledge/test/toy.md').read_text())
            c['procedure']['steps'][0]['context_refs']=['A.0']
            write_jsonl(root/'sources/cards.jsonl',[c])
            with self.assertRaisesRegex(ValueError,'future context|stale procedure'):export(root)

class ContextExportRegression(unittest.TestCase):
    def test_intro_quoted_command_does_not_require_procedural_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            speech=[{'ref':str(i),'text':t,'start':float(i),'end':float(i+1)} for i,t in enumerate(['给我展示内容。','第一个 Alpha 面板，','第二个 Beta 面板，'])]
            identity={'bvid':'toy'};ps=packets(speech,identity,'toy');cards=[];facts=[]
            for p in ps:
                aa=[dict(unit_id=u['unit_id'],role='action' if p['mode']=='context' else 'identity') for u in p['units']]
                facts+=materialize_units(p,aa)
                cards.append(dict(packet_id=p['packet_id'],assignments=aa,procedure={'steps':[]},alarms=[],link_error=None))
            m=dict(source_identity=identity,title='toy',source_url='https://example.test',domain='test',transcript_review_basis='fixture',legacy_root=None,source_files={})
            dump(root/'sources/input-manifest.json',m)
            for name,rows in [('speech',speech),('packets',ps),('cards',cards),('units.classified',facts)]:write_jsonl(root/'sources'/(name+'.jsonl'),rows)
            export(root)
            self.assertEqual((root/'sources/units.exported.jsonl').read_text(),'')

class CorrectionContracts(unittest.TestCase):
    fixture = ExportContracts.fixture
    def test_explicit_role_correction_preserves_original(self):
        from finalize_learning_v3 import finalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,f=self.fixture(root)
            write_jsonl(root/'sources/source-reviews.jsonl',[dict(fact_id=f['fact_id'],fact_sha256=hash_object(f),status='reject')])
            correction=dict(fact_id=f['fact_id'],fact_sha256=hash_object(f),status='approve',corrected_role='usage',reason='source review')
            write_jsonl(root/'sources/source-corrections.jsonl',[correction])
            finalize(root)
            original=json.loads((root/'sources/units.classified.jsonl').read_text())
            reviewed=json.loads((root/'sources/units.reviewed.jsonl').read_text())
            self.assertEqual(original['role'],'action');self.assertEqual(reviewed['role'],'usage')
            self.assertEqual(original['quote'],reviewed['quote'])
            self.assertNotEqual(original['fact_id'],reviewed['fact_id'])
    def test_stale_correction_refused(self):
        from finalize_learning_v3 import finalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,f=self.fixture(root)
            write_jsonl(root/'sources/source-reviews.jsonl',[dict(fact_id=f['fact_id'],fact_sha256=hash_object(f),status='reject')])
            write_jsonl(root/'sources/source-corrections.jsonl',[dict(fact_id=f['fact_id'],fact_sha256='changed',status='approve',corrected_role='usage',reason='x')])
            with self.assertRaisesRegex(ValueError,'stale'):finalize(root)
    def test_correction_cannot_add_unknown_action(self):
        from finalize_learning_v3 import finalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,f=self.fixture(root)
            write_jsonl(root/'sources/source-reviews.jsonl',[dict(fact_id=f['fact_id'],fact_sha256=hash_object(f),status='approve')])
            c2=copy.deepcopy(c);c2['procedure']['steps'][0]['action_refs']=['unknown']
            write_jsonl(root/'sources/procedure-corrections.jsonl',[dict(packet_id='P00',original_card_sha256=hash_object(c),status='approve',procedure=c2['procedure'],reason='x')])
            with self.assertRaisesRegex(ValueError,'unknown'):finalize(root)

if __name__=='__main__':unittest.main()
