import unittest
from run_video_understanding_v4 import validate_visual,validate_semantic,semantic_units,visual_schema,prepare_images,KINDS

class VideoExperimentContracts(unittest.TestCase):
    def setUp(self):
        self.case={'questions':[{'question_id':'x','choices':['checked','unknown']} ]}
        self.answer={'answers':{'x':{'value':'checked','positions':[0]}},'observation':'visible'}
    def test_text_cannot_cite_unseen_images(self):
        with self.assertRaisesRegex(ValueError,'no visual'):validate_visual(self.answer,self.case,'text_only')
    def test_out_of_range_reference_rejected(self):
        self.answer['answers']['x']['positions']=[4]
        with self.assertRaisesRegex(ValueError,'positions'):validate_visual(self.answer,self.case,'aligned')
    def test_unknown_cannot_fabricate_evidence(self):
        self.answer['answers']['x']['value']='unknown'
        with self.assertRaisesRegex(ValueError,'unknown should'):validate_visual(self.answer,self.case,'aligned')
    def test_duplicate_evidence_rejected(self):
        self.answer['answers']['x']['positions']=[0,0]
        with self.assertRaisesRegex(ValueError,'positions'):validate_visual(self.answer,self.case,'aligned')
    def test_semantic_sources_cannot_be_invented(self):
        d={'fields':{k:[] for k in KINDS}};d['fields']['definition']=['fake']
        with self.assertRaisesRegex(ValueError,'unknown'):validate_semantic(d,[{'ref':'A'}])
    def test_representation_preserves_source_text(self):
        p={'units':[{'unit_id':'A.0','quote':'不要超过五项，'},{'unit_id':'A.1','quote':'否则换布局。'}],
           'speech':[{'ref':'A','text':'不要超过五项，否则换布局。'}]}
        self.assertEqual(''.join(u['text'] for u in semantic_units(p,'clauses')),semantic_units(p,'sentences')[0]['text'])
    def test_text_schema_explicitly_disallows_image_evidence(self):
        self.assertEqual(visual_schema(self.case,'text_only')['properties']['answers']['properties']['x']['properties']['positions']['maxItems'],0)
    def test_reversal_keeps_identical_crops_and_relabels_positions(self):
        # Test the real image preparation path, with independently identifiable pixels.
        import tempfile
        from pathlib import Path
        from PIL import Image
        from ocr_routing import digest
        with tempfile.TemporaryDirectory() as tmp:
            frames=[]
            for i in range(4):
                path=Path(tmp)/f'{i}.png';Image.new('RGB',(8,8),(i*60,0,0)).save(path)
                frames.append(dict(crop=str(path),crop_sha256=digest(path),frame_id=f'F{i}',time=20.+i))
            case={'frames':frames,'permutation':[3,2,1,0]}
            ims,es,mapping=prepare_images(None,case,'reversed',None)
            self.assertEqual([im.getpixel((0,0))[0] for im in ims],[180,180,120,120,60,60,0,0])
            self.assertEqual([e['time'] for e in es],[0.,0.,1.,1.,2.,2.,3.,3.])
            self.assertEqual(mapping[0]['source_time'],23.)

class ScoringContracts(unittest.TestCase):
    def test_reversed_first_frame_supports_position_zero(self):
        from evaluate_video_understanding_v4 import expected_support
        self.assertEqual(expected_support({'question_id':'first','support_positions':[0]},True),{0})
        self.assertEqual(expected_support({'question_id':'order','support_positions':[0,1,3]},True),{0,2,3})
    def fixture(self):
        from run_video_understanding_v4 import MODES
        cases=[{'case_id':'toy'}]
        gold={'cases':[{'case_id':'toy','questions':[dict(question_id='first',answer='on',reversed_answer='off',support_positions=[0])]}]}
        records={('toy',mode):{'output':{'answers':{'first':dict(value='on',positions=[0])}}} for mode in MODES}
        return cases,gold,records
    def test_changing_answer_is_not_enough_to_pass_reversal(self):
        from evaluate_video_understanding_v4 import score_visual
        c,g,r=self.fixture();r[('toy','reversed')]['output']['answers']['first']['value']='unknown'
        summary,_,_=score_visual(c,g,r)
        self.assertEqual(summary['answers_changed'],1)
        self.assertEqual(summary['both_directions_correct'],0)
    def test_correct_guess_without_evidence_is_not_grounded(self):
        from evaluate_video_understanding_v4 import score_visual
        c,g,r=self.fixture();r[('toy','aligned')]['output']['answers']['first']['positions']=[]
        summary,_,_=score_visual(c,g,r)
        self.assertEqual(summary['modes']['aligned']['content_correct'],1)
        self.assertEqual(summary['modes']['aligned']['correct_with_required_positions'],0)
    def test_text_abstention_scored_separately_from_content(self):
        from evaluate_video_understanding_v4 import score_visual
        c,g,r=self.fixture();r[('toy','text_only')]['output']['answers']['first']=dict(value='unknown',positions=[])
        summary,_,_=score_visual(c,g,r)
        self.assertEqual(summary['modes']['text_only']['appropriate_text_abstentions'],1)
        self.assertEqual(summary['modes']['text_only']['content_correct'],0)

class SentenceGroupContracts(unittest.TestCase):
    def test_fragments_join_through_sentence_end(self):
        from run_semantic_groups_v4 import sentence_groups
        p={'packet_id':'P','speech':[{'ref':'A','text':'如果有数据，'},{'ref':'B','text':'就显示列表。'},{'ref':'C','text':'否则显示空状态。'}]}
        us=sentence_groups(p)
        self.assertEqual([u['text'] for u in us],['如果有数据，就显示列表。','否则显示空状态。'])
        self.assertEqual(us[0]['segment_refs'],['A','B'])
    def test_unfinished_tail_is_preserved(self):
        from run_semantic_groups_v4 import sentence_groups
        p={'packet_id':'P','speech':[{'ref':'A','text':'注意不能超过五项，'}]}
        self.assertEqual(sentence_groups(p)[0]['text'],p['speech'][0]['text'])
    def test_grouping_is_local_to_source_topic(self):
        from run_semantic_groups_v4 import sentence_groups
        a=sentence_groups({'packet_id':'A','speech':[{'ref':'a','text':'甲，'}]})
        b=sentence_groups({'packet_id':'B','speech':[{'ref':'b','text':'乙。'}]})
        self.assertNotEqual(a[0]['ref'],b[0]['ref'])
        self.assertEqual(a[0]['segment_refs'],['a'])

if __name__=='__main__':unittest.main()
