"""Version 2 source-linked knowledge. Classification is a semantic hypothesis."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from fact_ledger import hash_object


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Topic(StrictModel):
    title: str = Field(min_length=1)
    title_ref: str
    start_ref: str
    end_ref: str
    mode: Literal['concept', 'procedure', 'reference']


class TopicPlan(StrictModel):
    topics: list[Topic] = Field(min_length=1, max_length=24)
    overflow: bool


KINDS = ['definition', 'usage', 'interaction', 'procedure', 'limitation',
         'comparison', 'example', 'author_experience', 'success_criterion']


class FieldLink(StrictModel):
    kind: Literal['definition', 'usage', 'interaction', 'procedure', 'limitation',
                  'comparison', 'example', 'author_experience', 'success_criterion']
    refs: list[str] = Field(min_length=1, max_length=8)


class Omission(StrictModel):
    ref: str
    reason: Literal['housekeeping', 'outside_topic', 'insufficient_context']


class CardExtraction(StrictModel):
    fields: list[FieldLink] = Field(max_length=20)
    omitted_speech: list[Omission]
    overflow: bool


class Judgment(StrictModel):
    fact_id: str
    verdict: Literal['supported', 'unsupported', 'uncertain']
    reason: str = Field(min_length=1, max_length=160)


class Verification(StrictModel):
    decisions: list[Judgment]


def validate_topics(data, evidence):
    plan = TopicPlan.model_validate(data)
    if plan.overflow:
        raise ValueError('topic plan overflow: explicitly increase scope in another run')
    refs = {e['ref']: e for e in evidence}
    order = {e['ref']: i for i, e in enumerate(evidence)}
    used = set()
    result = []
    for topic in plan.topics:
        t = topic.model_dump()
        if any(t[k] not in refs for k in ['title_ref', 'start_ref', 'end_ref']):
            raise ValueError('unknown topic reference')
        a, b = order[t['start_ref']], order[t['end_ref']]
        if a > b or not a <= order[t['title_ref']] <= b:
            raise ValueError('invalid topic interval or title outside interval: ' + str(t))
        if t['title'] not in refs[t['title_ref']]['text']:
            matches = [ref for ref, e in refs.items() if t['title'] in e['text']]
            raise ValueError('topic title must be verbatim source text: ' + str({
                'title': t['title'], 'wrong_title_ref': t['title_ref'],
                'actual_text': refs[t['title_ref']]['text'], 'exact_matching_title_refs': matches,
                'repair': 'repair title_ref and interval; previous topic must end before this topic starts'}))
        interval = set(range(a, b + 1))
        if used & interval:
            raise ValueError('overlapping topic intervals: ' + str(t))
        used |= interval
        result.append({**t, 'topic_id': 'T' + hash_object(t)[:12],
                       'speech_refs': [e['ref'] for e in evidence[a:b + 1]],
                       'start': evidence[a]['start'], 'end': evidence[b]['end']})
    return sorted(result, key=lambda t: t['start'])


def materialize(data, packet):
    """Keep complete source units, including conditions, negations and quantities."""
    parsed = CardExtraction.model_validate(data)
    if parsed.overflow:
        raise ValueError('card overflow; cannot claim complete extraction')
    evidence = {e['ref']: e for e in packet['evidence']}
    speech = {e['ref'] for e in packet['evidence'] if e['modality'] == 'speech'}
    used, seen, facts = set(), set(), []
    for field in parsed.fields:
        if packet['topic']['mode'] == 'concept' and field.kind in ['procedure', 'success_criterion', 'author_experience']:
            raise ValueError('concept card cannot assert procedure, success criterion or personal experience')
        if len(set(field.refs)) != len(field.refs):
            raise ValueError('duplicate evidence reference')
        if set(field.refs) - set(evidence):
            raise ValueError('unknown field reference')
        key = (field.kind, tuple(field.refs))
        if key in seen:
            raise ValueError('duplicate fact')
        seen.add(key)
        selected = sorted((evidence[ref] for ref in field.refs),
                          key=lambda e: (e['start'], e['ref']))
        if field.kind == 'author_experience' and any(e['modality'] != 'speech' for e in selected):
            raise ValueError('author experience requires speech evidence')
        used |= set(field.refs) & speech
        core = {'schema_version': 2, 'source_identity': packet['source_identity'],
                'topic_id': packet['topic']['topic_id'], 'subject': packet['topic']['title'],
                'topic_mode': packet['topic']['mode'], 'kind': field.kind,
                'value': '\n'.join(e['text'] for e in selected),
                'source_spans': [{'ref': e['ref'], 'char_start': 0,
                                  'char_end': len(e['text']), 'quote': e['text']} for e in selected],
                'source_locations': [{k: v for k, v in e.items() if k != 'text'} for e in selected],
                'packet_sha256': hash_object(packet), 'origin': packet.get('classification_origin', 'model_classification_verbatim_evidence'),
                'semantic_status': 'pending', 'paraphrase_generated': False}
        facts.append({**core, 'fact_id': 'N' + hash_object(core)[:18]})
    omitted = [o.ref for o in parsed.omitted_speech]
    if len(set(omitted)) != len(omitted) or set(omitted) - speech or set(omitted) & used:
        raise ValueError('invalid or duplicate speech omission')
    if used | set(omitted) != speech:
        raise ValueError('unaccounted speech: ' + ','.join(sorted(speech - used - set(omitted))))
    return facts, [o.model_dump() for o in parsed.omitted_speech]


def validate_judgments(data, facts):
    v = Verification.model_validate(data)
    ids = [d.fact_id for d in v.decisions]
    if len(set(ids)) != len(ids) or set(ids) != {f['fact_id'] for f in facts}:
        raise ValueError('verification must cover each fact exactly once')
    return [d.model_dump() for d in v.decisions]


def export_decision(fact, judgments, reviews):
    """A model judgment is advisory; a separate digest-bound review authorizes export."""
    review = reviews.get(fact['fact_id'])
    if not review:
        return False, 'needs independent source/assignment review'
    if review['fact_sha256'] != hash_object(fact):
        raise ValueError('stale source review')
    if review['status'] != 'approve':
        return False, review['reason']
    return True, review['reason']
