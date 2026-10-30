import json,unittest,os
from copy import deepcopy
from pathlib import Path
from object_evidence_v6 import (bind_observation,start_record,append_evidence,summarize_record,
    validate_response,needs_reread)


def task(properties=None):
    return {'task_id':'C-2-content','object_scope':'demonstration_content',
            'source':{'frame_id':'F180','pts':96000,'time_base':'1/16000','time':6.,'crop_sha256':'source'},
            'properties':properties or {'pieces':'碎片','blue_circle':'蓝色圆形'}}


def obs(values=None,name='initial',spec=None):
    return bind_observation(spec or task(),values or {'pieces':'present','blue_circle':'present'},name,'native',{'view':name})


class PropertyEvidenceContracts(unittest.TestCase):
    def test_real_native_and_temporal_paths_consume_same_pixel_budget(self):
        import sys,importlib.util
        from types import SimpleNamespace
        from PIL import Image
        model_dir=os.environ.get('VIDEO_LEARNING_MODEL_DIR')
        if not model_dir:
            self.skipTest('Optional MLX/model check: set VIDEO_LEARNING_MODEL_DIR on Apple Silicon')
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        from run_object_properties_v6 import prepare_view
        project=Path(__file__).resolve().parent.parent
        processor=Qwen3VLProcessor.from_pretrained(model_dir,local_files_only=True)
        helper_path=project/'support/pilot/vision.py'
        sys.path.insert(0,str(helper_path.parent))
        spec=importlib.util.spec_from_file_location('v6_test_temporal_helper',helper_path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        model=SimpleNamespace(config={'model_type':'qwen3_vl'})
        for size in [(1405,775),(315,230),(1920,1080)]:
            image=Image.new('RGB',size,(30,60,90))
            _,_,native=prepare_view(model,processor,helper,image,'观察图片','native',524288)
            _,_,temporal=prepare_view(model,processor,helper,image,'观察图片','temporal',524288)
            self.assertEqual(native['grid'],temporal['grid'])
            self.assertEqual(native['media_tokens'],temporal['media_tokens'])
            self.assertLessEqual(native['resized_height']*native['resized_width'],524288)

    def test_coexisting_content_is_not_forced_into_one_label(self):
        record=start_record(obs())
        self.assertEqual(record['initial']['values'],{'pieces':'present','blue_circle':'present'})

    def test_missing_attribute_is_not_silently_absent(self):
        with self.assertRaisesRegex(ValueError,'scope'):
            validate_response({'pieces':'present'},task())

    def test_unknown_stays_distinct_from_absence(self):
        summary=summarize_record(start_record(obs({'pieces':'unknown','blue_circle':'absent'})))
        self.assertEqual(summary['properties']['pieces']['candidate_values'],[])
        self.assertEqual(summary['properties']['blue_circle']['candidate_values'],['absent'])

    def test_captured_v5_overwrite_is_prevented_without_correct_answer_lookup(self):
        from visual_ledger_v5 import finish_review
        path=Path(__file__).resolve().parent.parent/'tests/fixtures/v5-overwrite-case.json'
        event=json.loads(path.read_text())
        old=finish_review(event['initial'],event['attempts'],'corroborated')
        self.assertNotEqual(old['observation']['state'],event['initial']['observation']['state'])
        spec=task({'pieces':'碎片'});spec['source']=event['initial']['source']
        encode=lambda row:{'pieces':'present' if row['observation']['state']=='scattered_fragments' else 'absent'}
        initial=bind_observation(spec,encode(event['initial']),event['initial']['receipt'],'native',{})
        record=start_record(initial)
        for row in event['attempts']:
            record=append_evidence(record,bind_observation(spec,encode(row),row['receipt'],'native',row['view']),'targeted_reread')
        self.assertEqual(record['initial'],initial)
        summary=summarize_record(record)['properties']['pieces']
        self.assertEqual(summary['status'],'unreviewed_conflict')
        self.assertIsNone(summary['automatic_value'])
        self.assertEqual(summary['candidate_values'],['absent','present'])

    def test_append_does_not_mutate_original_record(self):
        initial=start_record(obs());before=deepcopy(initial)
        appended=append_evidence(initial,obs({'pieces':'absent','blue_circle':'present'},'reread'),'targeted_reread')
        self.assertEqual(initial,before);self.assertEqual(len(appended['additional']),1)

    def test_duplicate_receipt_does_not_add_votes(self):
        first=obs();record=start_record(first)
        record=append_evidence(record,first,'reread')
        self.assertEqual(len(record['additional']),0)
        second=obs(name='second')
        record=append_evidence(record,second,'reread')
        record=append_evidence(record,second,'reread')
        self.assertEqual(len(record['additional']),1)
        self.assertIsNone(summarize_record(record)['properties']['pieces']['automatic_value'])

    def test_agreement_never_marks_semantic_approval(self):
        record=append_evidence(start_record(obs()),obs(name='another'),'reread')
        self.assertFalse(summarize_record(record)['semantic_approval'])

    def test_new_information_does_not_overwrite_initial_unknown(self):
        record=start_record(obs({'pieces':'unknown','blue_circle':'unknown'}))
        record=append_evidence(record,obs(name='new'),'reread')
        field=summarize_record(record)['properties']['pieces']
        self.assertEqual(field['initial_value'],'unknown')
        self.assertEqual(field['status'],'unreviewed_additional_candidate')

    def test_other_frame_cannot_be_appended_to_same_source(self):
        spec=task();spec['source']['frame_id']='F181'
        with self.assertRaisesRegex(ValueError,'source/object'):
            append_evidence(start_record(obs()),obs(spec=spec),'reread')

    def test_content_and_window_identity_are_not_interchangeable(self):
        spec=task();spec['object_scope']='foreground_window'
        with self.assertRaisesRegex(ValueError,'source/object'):
            append_evidence(start_record(obs()),obs(spec=spec),'reread')

    def test_changed_bound_value_is_detected(self):
        observation=obs();observation['values']['pieces']='absent'
        with self.assertRaisesRegex(ValueError,'altered'):
            start_record(observation)

    def test_pts_and_time_must_agree(self):
        spec=task();spec['source']['time']=7
        with self.assertRaisesRegex(ValueError,'PTS'):
            obs(spec=spec)

    def test_conflicts_and_unknowns_trigger_but_agreement_cannot_prove_truth(self):
        self.assertEqual(needs_reread(obs(),obs(name='other')),[])
        self.assertEqual(needs_reread(obs(),obs({'pieces':'absent','blue_circle':'present'},'other')),['pieces'])
        self.assertEqual(needs_reread(obs({'pieces':'unknown','blue_circle':'present'}),obs({'pieces':'unknown','blue_circle':'present'},'other')),['pieces'])


if __name__=='__main__':unittest.main()
