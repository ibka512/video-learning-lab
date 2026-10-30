import copy
import json
from pathlib import Path
import tempfile
import unittest

from fact_ledger import hash_object
from knowledge_schema import (validate_topics, materialize, validate_judgments, export_decision)
from ocr_routing import dump, write_jsonl
from run_knowledge_pipeline import export, source_container_plan
from recover_knowledge_duplicates import remove_exact_duplicates, normalize_recorded_classification


class KnowledgeContracts(unittest.TestCase):
    def setUp(self):
        self.speech = [
            {'ref': 'A', 'text': '面板X 用来分组。', 'start': 1.0, 'end': 2.0, 'modality': 'speech'},
            {'ref': 'B', 'text': '如果内容超过五组，不要全部展开。', 'start': 2.0, 'end': 3.0, 'modality': 'speech'},
            {'ref': 'C', 'text': '点击才显示内容。', 'start': 3.0, 'end': 4.0, 'modality': 'speech'},
        ]
        self.plan = {'topics': [{'title': '面板X', 'title_ref': 'A', 'start_ref': 'A',
                                'end_ref': 'C', 'mode': 'concept'}], 'overflow': False}
        self.topic = validate_topics(self.plan, self.speech)[0]
        self.packet = {'source_identity': {'bvid': 'fixture'}, 'topic': self.topic, 'evidence': self.speech}
        self.data = {'fields': [{'kind': 'definition', 'refs': ['A']},
                                {'kind': 'limitation', 'refs': ['B']},
                                {'kind': 'interaction', 'refs': ['C']}],
                     'omitted_speech': [], 'overflow': False}

    def test_cross_sentence_subject_keeps_restriction_and_negation(self):
        facts, _ = materialize(self.data, self.packet)
        self.assertEqual(facts[1]['subject'], '面板X')
        self.assertEqual(facts[1]['value'], '如果内容超过五组，不要全部展开。')
        self.assertEqual(facts[1]['source_spans'][0]['char_start'], 0)
        self.assertEqual(facts[1]['source_locations'][0]['start'], 2.0)

    def test_overlapping_topic_is_rejected(self):
        self.plan['topics'].append(self.plan['topics'][0].copy())
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            validate_topics(self.plan, self.speech)

    def test_invented_subject_rejected(self):
        self.plan['topics'][0]['title'] = '面板Y'
        with self.assertRaisesRegex(ValueError, 'verbatim'):
            validate_topics(self.plan, self.speech)

    def test_unchaptered_input_uses_reference_container_without_guessing_task(self):
        plan = source_container_plan(self.speech)
        self.assertEqual(plan['topics'][0]['mode'], 'reference')
        self.assertEqual(validate_topics(plan, self.speech)[0]['speech_refs'], ['A', 'B', 'C'])
        chaptered = copy.deepcopy(self.speech)
        chaptered[0]['text'] = '第一个 测试对象。'
        chaptered[1]['text'] = '第二个 另一个对象。'
        self.assertIsNone(source_container_plan(chaptered))

    def test_title_outside_interval_rejected(self):
        self.plan['topics'][0]['start_ref'] = 'B'
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_topics(self.plan, self.speech)

    def test_unknown_reference_rejected(self):
        self.data['fields'][0]['refs'] = ['UNKNOWN']
        with self.assertRaisesRegex(ValueError, 'unknown'):
            materialize(self.data, self.packet)

    def test_missing_speech_requires_explicit_omission(self):
        self.data['fields'].pop()
        with self.assertRaisesRegex(ValueError, 'unaccounted'):
            materialize(self.data, self.packet)
        self.data['omitted_speech'] = [{'ref': 'C', 'reason': 'insufficient_context'}]
        self.assertEqual(len(materialize(self.data, self.packet)[0]), 2)

    def test_omitted_and_used_cannot_overlap(self):
        self.data['omitted_speech'] = [{'ref': 'C', 'reason': 'housekeeping'}]
        with self.assertRaisesRegex(ValueError, 'omission'):
            materialize(self.data, self.packet)

    def test_no_current_state_type_from_narration(self):
        self.data['fields'][0]['kind'] = 'observed_state'
        with self.assertRaises(ValueError):
            materialize(self.data, self.packet)

    def test_no_free_text_to_invent_a_paraphrase(self):
        self.data['fields'][0]['summary'] = '这是硬性标准'
        with self.assertRaises(ValueError):
            materialize(self.data, self.packet)

    def test_ocr_cannot_masquerade_as_author_experience(self):
        self.packet['topic']['mode'] = 'procedure'
        self.packet['evidence'] = copy.deepcopy(self.speech)
        self.packet['evidence'][0]['modality'] = 'ocr'
        self.data['fields'][0]['kind'] = 'author_experience'
        with self.assertRaisesRegex(ValueError, 'requires speech'):
            materialize(self.data, self.packet)

    def test_concept_usage_cannot_be_labeled_completed_procedure(self):
        self.data['fields'][0]['kind'] = 'success_criterion'
        with self.assertRaisesRegex(ValueError, 'concept card'):
            materialize(self.data, self.packet)

    def test_overflow_not_silently_accepted(self):
        self.data['overflow'] = True
        with self.assertRaisesRegex(ValueError, 'overflow'):
            materialize(self.data, self.packet)

    def test_duplicate_recovery_cannot_change_kind_refs_or_add_facts(self):
        original = copy.deepcopy(self.data)
        original['fields'].append(copy.deepcopy(original['fields'][0]))
        normalized, count = remove_exact_duplicates(original)
        self.assertEqual(count, 1)
        self.assertEqual(normalized, self.data)
        self.assertEqual(len(original['fields']), 4)
        self.assertEqual(len(materialize(normalized, self.packet)[0]), 3)

    def test_cleanup_does_not_convert_missing_speech_into_knowledge(self):
        self.data['fields'].pop()
        normalized, removed, missing = normalize_recorded_classification(self.data, self.packet)
        self.assertEqual(missing, ['C'])
        self.assertEqual(removed, 0)
        facts, omitted = materialize(normalized, self.packet)
        self.assertEqual(len(facts), 2)
        self.assertEqual(omitted, [{'ref': 'C', 'reason': 'insufficient_context'}])

    def test_partial_judge_result_is_rejected(self):
        facts, _ = materialize(self.data, self.packet)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            validate_judgments({'decisions': []}, facts)

    def test_judge_support_does_not_publish(self):
        fact = materialize(self.data, self.packet)[0][0]
        ok, _ = export_decision(fact, {fact['fact_id']: {'verdict': 'supported'}}, {})
        self.assertFalse(ok)

    def test_digest_bound_review_and_rejection(self):
        fact = materialize(self.data, self.packet)[0][0]
        review = {'status': 'approve', 'fact_sha256': hash_object(fact), 'reason': 'source checked'}
        self.assertTrue(export_decision(fact, {}, {fact['fact_id']: review})[0])
        review['status'] = 'reject'
        self.assertFalse(export_decision(fact, {}, {fact['fact_id']: review})[0])
        fact['value'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'stale'):
            export_decision(fact, {}, {fact['fact_id']: review})

    def test_export_keeps_unreviewed_fields_out_and_preserves_approved_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            r = Path(temporary)
            facts, _ = materialize(self.data, self.packet)
            dump(r / 'sources/input-manifest.json', {'source_files': {}, 'legacy_root': None,
                 'domain': '测试', 'title': '演示', 'source_url': 'https://example.com/',
                 'source_identity': {'bvid': 'fixture'}, 'transcript_review_basis': 'test fixture'})
            dump(r / 'packets' / (self.topic['topic_id'] + '.json'), self.packet)
            write_jsonl(r / 'sources/cards.jsonl', [{'topic': self.topic,
                        'raw_classification': self.data, 'fact_ids': [f['fact_id'] for f in facts]}])
            write_jsonl(r / 'sources/knowledge.candidates.jsonl', facts)
            write_jsonl(r / 'sources/semantic-judgments.jsonl', [])
            export(r)
            self.assertEqual((r / 'sources/knowledge.exported.jsonl').read_text(), '')
            write_jsonl(r / 'sources/source-reviews.jsonl', [{'fact_id': facts[1]['fact_id'],
                        'fact_sha256': hash_object(facts[1]), 'status': 'approve', 'reason': 'source checked'}])
            export(r)
            md = (r / 'knowledge/测试/fixture.md').read_text()
            self.assertIn('如果内容超过五组，不要全部展开。', md)
            self.assertNotIn('点击才显示内容。', md)
            altered = copy.deepcopy(facts)
            altered[1]['value'] = '可以全部展开'
            write_jsonl(r / 'sources/knowledge.candidates.jsonl', altered)
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                export(r)


if __name__ == '__main__':
    unittest.main()
