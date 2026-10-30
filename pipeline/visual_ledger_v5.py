"""Source-bound visual observations. No reference answers or model runtime imports."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

REASONS = ['none', 'not_visible', 'too_small', 'occluded', 'ambiguous', 'conflicting_observations', 'invalid_output']
KINDS = ['settings', 'running_game', 'editor', 'menu', 'content', 'other', 'unknown']
LAYERS = ['foreground', 'background', 'unknown']


def observation_schema(states):
    obj = {'type': 'object', 'properties': {
        'kind': {'type': 'string', 'enum': KINDS},
        'layer': {'type': 'string', 'enum': LAYERS},
        'bbox': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4},
        'cue': {'type': 'string', 'maxLength': 100}},
        'required': ['kind', 'layer', 'bbox', 'cue'], 'additionalProperties': False}
    # Structurally distinct alternatives avoid asking a small model to maintain
    # redundant state/reason/index relationships across independently sampled fields.
    def branch(values, target, reasons, minimum):
        return {'type': 'object', 'properties': {
            'state': {'type': 'string', 'enum': values},
            'target_object': {'type': 'integer', 'enum': [target]},
            'reason': {'type': 'string', 'enum': reasons},
            'objects': {'type': 'array', 'items': obj, 'minItems': minimum, 'maxItems': 4}},
            'required': ['state', 'target_object', 'reason', 'objects'], 'additionalProperties': False}
    return {'anyOf': [branch([s for s in states if s != 'unknown'], 0, ['none'], 1),
                      branch(['unknown'], -1, ['not_visible','too_small','occluded','ambiguous'], 0)]}


def validate_observation(obs, states):
    if not isinstance(obs, dict) or set(obs) != {'state', 'target_object', 'reason', 'objects'}:
        raise ValueError('observation scope mismatch')
    if obs['state'] not in states or obs['reason'] not in REASONS:
        raise ValueError('unknown state/reason')
    if type(obs['target_object']) is not int or not -1 <= obs['target_object'] < 4:
        raise ValueError('invalid object reference')
    objects = obs['objects']
    if not isinstance(objects, list) or len(objects) > 4:
        raise ValueError('invalid objects')
    for obj in objects:
        if set(obj) != {'kind', 'layer', 'bbox', 'cue'} or obj['kind'] not in KINDS or obj['layer'] not in LAYERS:
            raise ValueError('invalid object')
        box = obj['bbox']
        if (not isinstance(box, list) or len(box) != 4 or
                any(type(x) is not int or not 0 <= x <= 1000 for x in box) or
                not (box[0] < box[2] and box[1] < box[3])):
            raise ValueError('invalid region')
        if not isinstance(obj['cue'], str) or len(obj['cue']) > 100:
            raise ValueError('invalid cue')
    if obs['target_object'] >= len(objects):
        raise ValueError('missing target object')
    return obs


def unknown(reason='invalid_output'):
    return dict(state='unknown', target_object=-1, reason=reason, objects=[])


def consistency_issues(obs, spec):
    """Check logical constraints, not visual truth. Does not inspect free-text cues."""
    issues = []
    if obs['state'] == 'unknown':
        if obs['reason'] == 'none':
            issues.append('unknown_without_reason')
    else:
        if obs['reason'] != 'none':
            issues.append('certain_state_with_uncertainty')
        index = obs['target_object']
        if index < 0:
            issues.append('state_without_object')
        else:
            obj = obs['objects'][index]
            required = spec.get('object_requirements', {}).get(obs['state'], {})
            for key, allowed in required.items():
                if obj[key] not in allowed:
                    issues.append('target_' + key + '_conflict')
    return issues


def bind_observation(obs, source, position, receipt, spec):
    if type(position) is not int or position not in range(4):
        raise ValueError('invalid presentation position')
    if abs(float(Fraction(source['time_base']) * source['pts']) - source['time']) > 1e-6:
        raise ValueError('source PTS mismatch')
    return {'position': position, 'source': deepcopy(source), 'receipt': receipt,
            'observation': deepcopy(obs), 'issues': consistency_issues(obs, spec)}


def effective_state(row):
    return row['observation']['state'] if not row['issues'] else 'unknown'


def ordered_rows(rows):
    positions = [r['position'] for r in rows]
    if len(rows) != 4 or sorted(positions) != [0, 1, 2, 3]:
        raise ValueError('missing/duplicate presentation position')
    return sorted(rows, key=lambda r: r['position'])


def derive_answers(rows, spec):
    """A small adapter for the public v4 questions, not a general video reasoning model."""
    rows = ordered_rows(rows)
    states = [effective_state(r) for r in rows]
    answer = lambda value, positions: {'value': value, 'positions': [] if value == 'unknown' else positions}
    endpoints = set(spec['endpoint_choices'])
    result = {k: answer(states[p] if states[p] in endpoints else 'unknown', [p])
              for k, p in [('first', 0), ('last', 3)]}
    if spec['adapter'] == 'checkerboard':
        middle = states[1:3]
        value = ('unknown' if 'unknown' in middle else
                 'both_checkerboard' if middle.count('checkerboard') == 2 else
                 'neither_checkerboard' if middle.count('not_checkerboard') == 2 else 'only_one')
        result['middle'] = answer(value, [1, 2])
    elif spec['adapter'] == 'checkbox':
        # Only sampled states. Never infer a mouse/keyboard action, even on a state transition.
        value = 'static_checked' if states == ['checked'] * 4 else 'unknown'
        result['change'] = answer(value, [0, 1, 2, 3])
    else:
        value = 'unknown'; support = []
        compact = []
        for pos, state in enumerate(states):
            if not compact or state != compact[-1][0]:
                compact.append((state, pos))
        if 'unknown' not in states:
            for rule in spec['sequence_rules']:
                seq = [(s, p) for s, p in compact if s not in rule.get('ignore_states', [])]
                if [s for s, _ in seq] == rule['states']:
                    value = rule['value']; support = sorted({0, 3, *(p for _, p in seq)}); break
        result['order'] = answer(value, support)
    return result


def render_description(rows, spec):
    parts = []
    for row in ordered_rows(rows):
        state = effective_state(row)
        why = ','.join(row['issues']) or row['observation']['reason']
        suffix = f'（{why}）' if state == 'unknown' else ''
        parts.append(f"位置{row['position']}，源时间{row['source']['time']:g}秒：{spec['labels'][state]}{suffix}")
    return '；'.join(parts) + '。仅描述所采样画面，不证明未观测区间的点击或因果关系。'


def state_changes(rows):
    rows = ordered_rows(rows); changes = []
    for a, b in zip(rows, rows[1:]):
        x, y = effective_state(a), effective_state(b)
        if 'unknown' not in (x, y) and x != y:
            forward = a['source']['time'] < b['source']['time']
            changes.append({'positions': [a['position'], b['position']], 'states': [x, y],
                            'source_time_bounds': sorted([a['source']['time'], b['source']['time']]),
                            'chronological': forward, 'localization': 'sample_bracket_only',
                            'action': None, 'cause': None})
    return changes


def review_reasons(single, joint):
    reasons = list(single['issues'])
    x, y = effective_state(single), effective_state(joint)
    if x == 'unknown':
        reasons.append('unresolved_single_observation')
    if x != y:
        reasons.append('single_multi_disagreement')
    return sorted(set(reasons))


def review_decision(initial, attempts):
    """A corroboration policy, not semantic approval or an independent vote."""
    if not attempts:
        return None
    base = effective_state(initial)
    first = effective_state(attempts[0])
    if first != 'unknown' and first == base:
        return 0
    if len(attempts) >= 2 and first != 'unknown' and effective_state(attempts[1]) == first:
        return 1
    return None


def finish_review(initial, attempts, reason):
    selected = review_decision(initial, attempts)
    result = deepcopy(attempts[selected] if selected is not None else initial)
    if selected is None:
        result['observation'] = unknown('conflicting_observations' if attempts else 'ambiguous')
        result['issues'] = []
    result['review'] = {'status': 'corroborated_candidate' if selected is not None else 'unresolved',
                        'stop_reason': reason, 'attempts': [a['receipt'] for a in attempts],
                        'initial_receipt': initial['receipt'], 'semantic_approval': False}
    return result


class ReviewBudget:
    def __init__(self, max_rounds=2, max_calls=48):
        self.max_rounds = max_rounds; self.max_calls = max_calls
        self.used = 0; self.seen = {}; self.rounds = {}

    def reserve(self, key, view_identity):
        if self.rounds.get(key, 0) >= self.max_rounds:
            return 'round_limit'
        if self.used >= self.max_calls:
            return 'total_limit'
        if view_identity in self.seen.get(key, set()):
            return 'no_new_view'
        self.seen.setdefault(key, set()).add(view_identity)
        self.rounds[key] = self.rounds.get(key, 0) + 1; self.used += 1
        return None
