import unittest
from copy import deepcopy
from visual_ledger_v5 import (validate_observation, bind_observation, effective_state, derive_answers,
    render_description, state_changes, review_reasons, review_decision, finish_review, ReviewBudget, observation_schema)


SPEC = {'states':['settings','running_game','editor','unknown'],
        'labels':{'settings':'项目设置','running_game':'运行窗口','editor':'编辑器','unknown':'无法确定'},
        'endpoint_choices':['settings','running_game','editor','unknown'], 'adapter':'sequence',
        'object_requirements':{s:{'kind':[s],'layer':['foreground']} for s in ['settings','running_game','editor']},
        'sequence_rules':[{'states':['settings','running_game'],'value':'settings_to_game'},
                          {'states':['running_game','settings'],'value':'game_to_settings'}]}


def make_row(state='settings', position=0, kind=None, layer='foreground', reason='none', time=None):
    obj = {'kind':kind or state,'layer':layer,'bbox':[10,20,900,950],'cue':'visible title'}
    obs = {'state':state,'target_object':0,'reason':reason,'objects':[obj]}
    if state=='unknown':
        obs = {'state':'unknown','target_object':-1,'reason':'ambiguous','objects':[]}
    sec = float(position if time is None else time)
    return bind_observation(obs,{'pts':int(sec*100),'time_base':'1/100','time':sec,'frame_id':f'F{sec}'},position,'receipt',SPEC)


class SourceAndConsistency(unittest.TestCase):
    def test_generation_contract_cannot_emit_known_state_without_an_object(self):
        branches=observation_schema(SPEC['states'])['anyOf']
        known,unknown_branch=branches
        self.assertNotIn('unknown',known['properties']['state']['enum'])
        self.assertEqual(known['properties']['objects']['minItems'],1)
        self.assertEqual(known['properties']['target_object']['enum'],[0])
        self.assertEqual(known['properties']['reason']['enum'],['none'])
        self.assertEqual(unknown_branch['properties']['state']['enum'],['unknown'])
        self.assertEqual(unknown_branch['properties']['target_object']['enum'],[-1])
        self.assertNotIn('none',unknown_branch['properties']['reason']['enum'])

    def test_source_pts_cannot_be_relabelled_as_presentation_position(self):
        row = make_row(position=0,time=23)
        self.assertEqual(row['source']['time'],23)
        bad = deepcopy(row['source']); bad['pts'] = 2200
        with self.assertRaisesRegex(ValueError,'PTS'):
            bind_observation(row['observation'],bad,0,'r',SPEC)

    def test_missing_object_reference_rejected(self):
        obs = make_row()['observation']; obs['target_object'] = 2
        with self.assertRaisesRegex(ValueError,'missing target'):
            validate_observation(obs,SPEC['states'])

    def test_invalid_crop_box_is_not_accepted_as_evidence(self):
        obs = make_row()['observation']; obs['objects'][0]['bbox'] = [100,0,20,900]
        with self.assertRaisesRegex(ValueError,'region'):
            validate_observation(obs,SPEC['states'])

    def test_background_editor_cannot_support_foreground_running_state(self):
        row = make_row('running_game',kind='editor',layer='background')
        self.assertIn('target_kind_conflict',row['issues'])
        self.assertIn('target_layer_conflict',row['issues'])
        self.assertEqual(effective_state(row),'unknown')

    def test_known_value_with_uncertain_basis_is_blocked(self):
        row = make_row(reason='too_small')
        self.assertEqual(effective_state(row),'unknown')

    def test_duplicate_positions_do_not_silently_replace_a_frame(self):
        with self.assertRaisesRegex(ValueError,'position'):
            derive_answers([make_row(position=i) for i in [0,1,2,2]],SPEC)

    def test_middle_unknown_cannot_be_hidden_by_correct_endpoints(self):
        rows = [make_row(s,i) for i,s in enumerate(['settings','unknown','running_game','running_game'])]
        answers = derive_answers(rows,SPEC)
        self.assertEqual(answers['first']['value'],'settings')
        self.assertEqual(answers['last']['value'],'running_game')
        self.assertEqual(answers['order'],{'value':'unknown','positions':[]})
        self.assertIn('位置1，源时间1秒：无法确定',render_description(rows,SPEC))

    def test_reverse_presentation_does_not_sort_by_original_time(self):
        rows = [make_row(s,i,time=23-i) for i,s in enumerate(['running_game','running_game','settings','settings'])]
        self.assertEqual(derive_answers(rows,SPEC)['order']['value'],'game_to_settings')
        self.assertFalse(state_changes(rows)[0]['chronological'])

    def test_state_change_has_interval_and_no_invented_action(self):
        rows = [make_row(s,i,time=21+i) for i,s in enumerate(['settings','settings','running_game','running_game'])]
        change = state_changes(rows)[0]
        self.assertEqual(change['source_time_bounds'],[22,23])
        self.assertIsNone(change['action']); self.assertIsNone(change['cause'])

    def test_checkbox_transition_does_not_prove_a_click(self):
        spec = {**SPEC,'adapter':'checkbox','object_requirements':{},'endpoint_choices':['checked','unchecked','unknown']}
        rows = []
        for i,state in enumerate(['checked','checked','unchecked','unchecked']):
            row = make_row(position=i); row['observation']['state'] = state; rows.append(row)
        self.assertEqual(derive_answers(rows,spec)['change']['value'],'unknown')
        for row in rows: row['observation']['state'] = 'checked'
        self.assertEqual(derive_answers(rows,spec)['change']['value'],'static_checked')


class BoundedReview(unittest.TestCase):
    def test_native_pixel_coordinates_are_converted_before_reread_prompt(self):
        from run_visual_ledger_v5 import target_hint_in_view
        self.assertEqual(target_hint_in_view([400,130,1805,905],[0,0,1920,1080]),[208,120,940,838])
        self.assertEqual(target_hint_in_view([400,130,1805,905],[500,150,1700,900]),[0,0,1000,1000])

    def test_disagreement_triggers_without_reference_answer(self):
        self.assertIn('single_multi_disagreement',review_reasons(make_row(),make_row('editor')))

    def test_identical_known_wrong_observations_are_not_detectable(self):
        # The contract must not pretend agreement detects visual errors.
        self.assertEqual(review_reasons(make_row('editor'),make_row('editor')),[])

    def test_no_new_view_does_not_consume_another_request(self):
        b = ReviewBudget(); self.assertIsNone(b.reserve('a','pixels1'))
        self.assertEqual(b.reserve('a','pixels1'),'no_new_view'); self.assertEqual(b.used,1)

    def test_per_frame_and_total_limits(self):
        b = ReviewBudget(max_rounds=2,max_calls=3)
        self.assertIsNone(b.reserve('a','one')); self.assertIsNone(b.reserve('a','two'))
        self.assertEqual(b.reserve('a','three'),'round_limit')
        self.assertIsNone(b.reserve('b','one')); self.assertEqual(b.reserve('c','one'),'total_limit')

    def test_disagreeing_rereads_do_not_choose_latest_answer(self):
        initial = make_row('settings')
        attempts = [make_row('editor'),make_row('running_game')]
        self.assertIsNone(review_decision(initial,attempts))
        final = finish_review(initial,attempts,'round_limit')
        self.assertEqual(effective_state(final),'unknown')
        self.assertEqual(initial['observation']['state'],'settings')

    def test_two_agreeing_rereads_can_replace_initial_but_not_approve_it(self):
        final = finish_review(make_row('editor'),[make_row('running_game'),make_row('running_game')],'corroborated')
        self.assertEqual(effective_state(final),'running_game')
        self.assertFalse(final['review']['semantic_approval'])

    def test_full_context_then_roi_stay_within_native_image(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from run_visual_ledger_v5 import reread_view, verify_source
        from ocr_routing import digest
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory)/'full.png'; crop = Path(directory)/'crop.png'
            im = Image.new('RGB',(200,100),(10,20,30)); im.save(full); im.crop([50,10,190,90]).save(crop)
            source = dict(source_image=str(full),source_image_sha256=digest(full),crop=str(crop),crop_sha256=digest(crop),bbox_top_left=[50,10,190,90])
            verify_source(source)
            self.assertEqual(reread_view(source,make_row()['observation'],0),[0,0,200,100])
            box = reread_view(source,make_row()['observation'],1)
            self.assertTrue(0<=box[0]<box[2]<=200 and 0<=box[1]<box[3]<=100)
            source['bbox_top_left']=[0,0,100,90]
            with self.assertRaisesRegex(ValueError,'pixels'):
                verify_source(source)


if __name__=='__main__':
    unittest.main()
