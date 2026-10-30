"""Read-only regression scoring; target lists never enter extraction requests."""
import argparse
import json
import re
from pathlib import Path

from fact_ledger import hash_object
from ocr_routing import digest, dump, read_jsonl


def compact(s):
    return re.sub(r'\s+', '', s).casefold()


def resolve(root, path):
    mapping = root.parent.parent / 'sources/portable-paths.json'
    if mapping.exists():
        table = json.loads(mapping.read_text())
        if str(path) in table:
            return mapping.parent.parent / table[str(path)]
    return Path(path)


def audit(root):
    manifest = json.loads((root / 'sources/input-manifest.json').read_text())
    for path, sha in manifest['source_files'].items():
        assert digest(resolve(root, path)) == sha, path
    facts = read_jsonl(root / 'sources/knowledge.candidates.jsonl')
    exported = read_jsonl(root / 'sources/knowledge.exported.jsonl')
    reviews = {r['fact_id']: r for r in read_jsonl(root / 'sources/source-reviews.jsonl')}
    judgments = {r['fact_id']: r for r in read_jsonl(root / 'sources/semantic-judgments.jsonl')}
    checked = 0
    ocr_refs = set()
    for fact in facts:
        packet = json.loads((root / 'packets' / (fact['topic_id'] + '.json')).read_text())
        assert fact['packet_sha256'] == hash_object(packet)
        assert fact['source_identity'] == manifest['source_identity']
        es = {e['ref']: e for e in packet['evidence']}
        assert fact['subject'] == packet['topic']['title']
        assert len(fact['source_spans']) == len(fact['source_locations'])
        for span, loc in zip(fact['source_spans'], fact['source_locations']):
            e = es[span['ref']]
            assert e['text'][span['char_start']:span['char_end']] == span['quote'] == e['text']
            assert loc['ref'] == span['ref']
            assert (loc['start'], loc['end']) == (e['start'], e['end'])
            if e['modality'] == 'ocr':
                ocr_refs.add(e['ref'])
                assert digest(resolve(root, e['source_image'])) == e['source_image_sha256']
            checked += 1
        assert fact['value'] == '\n'.join(s['quote'] for s in fact['source_spans'])
    original = {f['fact_id']: f for f in facts}
    for f in exported:
        r = reviews[f['fact_id']]
        assert r['status'] == 'approve' and r['fact_sha256'] == hash_object(original[f['fact_id']])
    return {'candidate_fields': len(facts), 'exported_fields': len(exported),
            'exact_source_spans_checked': checked, 'cited_ocr_observations': len(ocr_refs),
            'judgments_received': len(judgments), 'reviewed_fields': len(reviews),
            'all_exported_fields_have_digest_bound_review': True}, facts, exported


def frontend(root, targets_path):
    integrity, facts, exported = audit(root)
    targets = json.loads(targets_path.read_text())
    # These acceptable source field types are an evaluator policy, never model input.
    types = {'definition': {'definition', 'interaction'}, 'limitation': {'limitation', 'author_experience'},
             'comparison': {'comparison'}, 'usage': {'usage'}, 'term_identity': set()}
    cards_raw = read_jsonl(root / 'sources/cards.jsonl')
    direct_ids = {fid for c in cards_raw if not c.get('classification_normalization') for fid in c['fact_ids']}
    direct = [f for f in facts if f['fact_id'] in direct_ids]
    after_cleanup = [f for f in facts if f['origin'] != 'source_rule_interior_gap_warning']
    scores = []
    for target in targets['targets']:
        term = target['id'].split('-')[0].casefold()
        a, b = target['window']
        def check(source, typed):
            related = [f for f in source if term in f['subject'].casefold()
                       and any(x['start'] < b and x['end'] > a for x in f['source_locations'])
                       and (not typed or not types[target['kind']] or f['kind'] in types[target['kind']])]
            text = compact(' '.join(f['subject'] + ' ' + f['value'] for f in related))
            missing = [x for x in target['all_text'] if compact(x) not in text]
            return not missing, missing
        candidate_pass, candidate_missing = check(facts, True)
        published_pass, missing = check(exported, True)
        scores.append({'target': target['id'], 'candidate_typed_wording_retained': candidate_pass,
                       'direct_model_typed_wording_retained': check(direct, True)[0],
                       'after_output_cleanup_typed_wording_retained': check(after_cleanup, True)[0],
                       'candidate_missing': candidate_missing, 'exported_typed_wording_retained': published_pass,
                       'exported_missing': missing})
    cards = []
    for term in targets['structured_term_targets']:
        fs = [f for f in facts if term.casefold() in f['subject'].casefold()]
        approved = [f for f in exported if term.casefold() in f['subject'].casefold()]
        cards.append({'term': term, 'automatic_fields': len(fs), 'reviewed_fields': len(approved),
                      'automatic_has_definition': any(f['kind'] == 'definition' for f in fs),
                      'reviewed_has_definition': any(f['kind'] == 'definition' for f in approved)})
    result = {**integrity, 'benchmark': 'frontend_regression_not_new_holdout', 'targets': scores,
              'target_count': len(scores),
              'direct_model_validated_fields': len(direct),
              'direct_model_typed_wording_retained': sum(s['direct_model_typed_wording_retained'] for s in scores),
              'after_output_cleanup_typed_wording_retained': sum(s['after_output_cleanup_typed_wording_retained'] for s in scores),
              'candidate_typed_wording_retained': sum(s['candidate_typed_wording_retained'] for s in scores),
              'exported_typed_wording_retained': sum(s['exported_typed_wording_retained'] for s in scores),
              'component_cards': cards, 'automatic_objects': sum(c['automatic_fields'] > 0 for c in cards),
              'reviewed_objects_with_definition': sum(c['reviewed_has_definition'] for c in cards),
              'target_sha256': digest(targets_path),
              'scoring_limit': 'Typed wording checks plus Codex source review; not independent human semantic gold.'}
    dump(root / 'sources/regression-evaluation.json', result)
    return {k: v for k, v in result.items() if k not in ['targets', 'component_cards']}


def godot(root, previous):
    integrity, facts, exported = audit(root)
    legacy = read_jsonl(root / 'sources/legacy-reviewed-fields.jsonl')
    before = [f for f in read_jsonl(previous / 'sources/facts.exported.jsonl') if f['kind'] != 'author_statement']
    assert legacy == before
    assert all(f['kind'] not in ['observed_state', 'help_default'] for f in facts)
    result = {**integrity, 'benchmark': 'Godot_regression', 'legacy_fields_preserved': len(legacy),
              'legacy_exact_match': True, 'new_narration_not_promoted_to_current_state': True,
              'new_unsupported_ocr_not_promoted_to_legacy_default': True,
              'procedure_fields': sum(f['kind'] == 'procedure' for f in exported)}
    dump(root / 'sources/regression-evaluation.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--frontend-targets', type=Path)
    p.add_argument('--godot-previous', type=Path)
    a = p.parse_args()
    if bool(a.frontend_targets) == bool(a.godot_previous):
        p.error('select one evaluation dataset')
    r = frontend(a.root, a.frontend_targets) if a.frontend_targets else godot(a.root, a.godot_previous)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
