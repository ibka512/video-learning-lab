"""Recover exact repeated entries only; never invent or relabel knowledge."""
import argparse
import json
from pathlib import Path

from knowledge_schema import CardExtraction, materialize
from ocr_routing import digest, dump, read_jsonl, write_jsonl
from fact_ledger import hash_object
import re
from run_knowledge_pipeline import export


def remove_exact_duplicates(data):
    parsed = CardExtraction.model_validate(data)
    seen, fields, removed = set(), [], 0
    for f in parsed.fields:
        key = (f.kind, tuple(f.refs))
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            fields.append(f.model_dump())
    return {**parsed.model_dump(), 'fields': fields}, removed


def normalize_recorded_classification(data, packet):
    normalized, removed = remove_exact_duplicates(data)
    referenced = {ref for f in normalized['fields'] for ref in f['refs']}
    omitted = {o['ref'] for o in normalized['omitted_speech']}
    missing = [e['ref'] for e in packet['evidence']
               if e['modality'] == 'speech' and e['ref'] not in referenced | omitted]
    # Missing speech stays explicitly unclassified. No knowledge is inferred.
    normalized['omitted_speech'] += [{'ref': ref, 'reason': 'insufficient_context'} for ref in missing]
    return normalized, removed, missing


def recover(root):
    cards = read_jsonl(root / 'sources/cards.jsonl')
    facts = read_jsonl(root / 'sources/knowledge.candidates.jsonl')
    topics = json.loads((root / 'sources/topics.json').read_text())
    prior = root / 'sources/duplicate-recovery.json'
    records = json.loads(prior.read_text())['recovered'] if prior.exists() else []
    done = {c['topic']['topic_id'] for c in cards}
    for topic in topics:
        tid = topic['topic_id']
        if tid in done:
            continue
        packet = json.loads((root / 'packets' / (tid + '.json')).read_text())
        for receipt in sorted((root / 'runs').glob(tid + '-extract-attempt-*.json'), reverse=True):
            record = json.loads(receipt.read_text())
            if record.get('finish_reason') == 'length' or record.get('output') is None:
                continue
            try:
                normalized, removed, missing = normalize_recorded_classification(record['output'], packet)
                if not removed and not missing:
                    continue
                fs, omitted = materialize(normalized, packet)
            except (ValueError, TypeError, KeyError):
                continue
            cards.append({'topic': topic, 'raw_classification': normalized,
                          'fact_ids': [f['fact_id'] for f in fs], 'omitted_speech': omitted,
                          'classification_normalization': 'recorded_output_cleanup_after_bounded_failure'})
            facts.extend(fs)
            records.append({'topic_id': tid, 'receipt': str(receipt.relative_to(root)),
                            'receipt_sha256': digest(receipt), 'duplicates_removed': removed,
                            'explicitly_unclassified_speech': missing,
                            'recovered_fields': len(fs), 'new_model_calls': 0,
                            'semantic_status': 'pending source review; not model validation success'})
            break
    # A warning in an interior gap is retained as an explicitly rule-linked
    # candidate. Its topic assignment still requires semantic source review.
    all_speech = read_jsonl(root / 'sources/speech.jsonl')
    assigned = {ref for t in topics for ref in t['speech_refs']}
    ordered = sorted(topics, key=lambda t: t['start'])
    gap_records = []
    existing_refs = {s['ref'] for f in facts for s in f['source_spans']}
    for previous, following in zip(ordered, ordered[1:]):
        for e in all_speech:
            if e['ref'] in assigned or e['ref'] in existing_refs:
                continue
            if not previous['end'] <= e['start'] < following['start']:
                continue
            if not re.match(r'^(?:注意|不要|不能|避免|切勿)', e['text']):
                continue
            topic = {**previous, 'topic_id': 'G' + hash_object([previous['topic_id'], e])[:12],
                     'speech_refs': [e['ref']], 'start': e['start'], 'end': e['end'],
                     'parent_topic_id': previous['topic_id'], 'assignment_status': 'pending_source_review'}
            packet = {'schema_version': 2, 'source_identity': json.loads((root / 'sources/input-manifest.json').read_text())['source_identity'],
                      'topic': topic, 'evidence': [e], 'classification_origin': 'source_rule_interior_gap_warning',
                      'binding_basis': 'warning in interior gap tentatively attached to preceding topic; requires source review'}
            raw = {'fields': [{'kind': 'limitation', 'refs': [e['ref']]}], 'omitted_speech': [], 'overflow': False}
            fs, omitted = materialize(raw, packet)
            dump(root / 'packets' / (topic['topic_id'] + '.json'), packet)
            cards.append({'topic': topic, 'raw_classification': raw,
                          'fact_ids': [f['fact_id'] for f in fs], 'omitted_speech': omitted,
                          'classification_normalization': 'source_rule_gap_warning_not_model_extraction'})
            facts.extend(fs)
            gap_records.append({'ref': e['ref'], 'parent_topic_id': previous['topic_id'],
                                'fact_ids': [f['fact_id'] for f in fs], 'new_model_calls': 0})
    cards.sort(key=lambda c: c['topic']['start'])
    write_jsonl(root / 'sources/cards.jsonl', cards)
    write_jsonl(root / 'sources/knowledge.candidates.jsonl', facts)
    dump(prior, {'recovered': records, 'operations': ['exact_duplicate_removal', 'explicit_unclassified_speech_accounting'],
                 'no_new_semantic_content_generated': True})
    gap_path = root / 'sources/gap-warning-recovery.json'
    if gap_records or not gap_path.exists():
        dump(gap_path, {'candidates': gap_records, 'assignment_requires_source_review': True})
    coverage = json.loads((root / 'sources/coverage.json').read_text())
    initial = root / 'sources/coverage.before-recovery.json'
    if not initial.exists():
        dump(initial, coverage)
    recovered_ids = {r['topic_id'] for r in records}
    used = {s['ref'] for f in facts for s in f['source_spans']}
    coverage['completed_topics'] = len(cards)
    coverage['assigned_speech_without_candidate'] = [ref for ref in coverage['assigned_speech_without_candidate'] if ref not in used]
    history = coverage.get('historical_failures', []) + coverage['failures']
    coverage['historical_failures'] = list({hash_object(f): f for f in history}.values())
    coverage['failures'] = [f for f in coverage['failures'] if f['topic_id'] not in recovered_ids]
    coverage['duplicate_recovered_topics'] = len(records)
    coverage['program_unclassified_speech'] = [ref for record in records for ref in record.get('explicitly_unclassified_speech', [])]
    coverage['gap_warning_refs'] = [r['ref'] for r in json.loads(gap_path.read_text())['candidates']]
    coverage['omissions'] = [{'topic_id': c['topic']['topic_id'], **o} for c in cards for o in c['omitted_speech']]
    coverage['completed_original_topics'] = sum(c['topic']['topic_id'] in {t['topic_id'] for t in topics} for c in cards)
    coverage['supplementary_cards'] = len(cards) - coverage['completed_original_topics']
    coverage['unique_subjects'] = len({c['topic']['title'] for c in cards})
    dump(root / 'sources/coverage.json', coverage)
    export(root)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(recover(a.root.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
