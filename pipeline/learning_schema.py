"""Clause evidence and procedural dependencies. No generated factual prose."""
from __future__ import annotations
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from fact_ledger import hash_object

ROLES = ['identity','definition','usage','behavior','limitation','comparison','analogy',
         'condition','action','result','overview','housekeeping','unresolved']
class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid', strict=True)
class Assignment(Strict):
    unit_id: str
    role: Literal['identity','definition','usage','behavior','limitation','comparison','analogy',
                  'condition','action','result','overview','housekeeping','unresolved']
class Classification(Strict):
    assignments: list[Assignment]
class Step(Strict):
    action_refs: list[str] = Field(min_length=1, max_length=4)
    context_refs: list[str] = Field(max_length=4)
    condition_refs: list[str] = Field(max_length=4)
    result_refs: list[str] = Field(max_length=4)
    after: list[int] = Field(max_length=4)
class Procedure(Strict):
    steps: list[Step] = Field(max_length=40)
class Check(Strict):
    unit_id: str
    verdict: Literal['supported','wrong_role','missing_context','uncertain']
class Checks(Strict):
    checks: list[Check]


def split_units(speech):
    """Punctuation boundaries retain exact offsets; times remain parent segment times."""
    units=[]
    for e in speech:
        for i, m in enumerate(re.finditer(r'[^，。；：！？,;!?]+[，。；：！？,;!?]*', e['text'])):
            if not m.group().strip(): continue
            units.append({'unit_id': f"{e['ref']}.{i}", 'segment_ref': e['ref'],
                          'char_start': m.start(), 'char_end': m.end(), 'quote': m.group(),
                          'start': e['start'], 'end': e['end'],
                          'timing_precision': 'parent_segment_not_clause_alignment'})
    return units


def classify(data, units):
    rows=[r.model_dump() for r in Classification.model_validate(data).assignments]
    ids=[r['unit_id'] for r in rows]
    expected=[u['unit_id'] for u in units]
    if ids != expected:
        raise ValueError('assign every unit exactly once in input order: '+str(expected))
    for row,u in zip(rows,units):
        if row['role']=='condition' and not re.search(r'如果|若|当|时候|时[，。；]|前提|只有|只要|除非|一旦',u['quote']):
            raise ValueError('condition needs an actual conditional predicate, not a setting path, analogy or filler: '+str(u))
        if row['role']=='action' and re.match(r'^(?:使|达成|达到)',u['quote']):
            raise ValueError('purpose/result fragment is not an independent action: '+str(u))
    return rows


def build_steps(data, units, assignments):
    raw=Procedure.model_validate(data).model_dump()
    by_id={u['unit_id']:u for u in units}
    roles={r['unit_id']:r['role'] for r in assignments}
    order={u['unit_id']:i for i,u in enumerate(units)}
    used=set(); previous=-1
    for i,s in enumerate(raw['steps']):
        for field in ['action_refs','context_refs','condition_refs','result_refs']:
            rs=s[field]
            if len(set(rs))!=len(rs) or set(rs)-set(by_id): raise ValueError('unknown or duplicate refs')
            if rs != sorted(rs,key=order.get): raise ValueError('reference order changed')
        if any(roles[r]!='action' for r in s['action_refs']): raise ValueError('action must use action-classified source')
        if set(s['action_refs']) & used: raise ValueError('action used twice')
        used.update(s['action_refs'])
        if min(order[r] for r in s['action_refs']) <= previous: raise ValueError('step chronology reversed')
        previous=max(order[r] for r in s['action_refs'])
        if any(roles[r]!='condition' for r in s['condition_refs']): raise ValueError('branch condition must use condition source')
        if any(roles[r]!='result' for r in s['result_refs']): raise ValueError('result must use result source')
        if any(order[r]>previous for r in s['context_refs']+s['condition_refs']): raise ValueError('future context or condition')
        if any(j<0 or j>=i for j in s['after']) or len(set(s['after']))!=len(s['after']): raise ValueError('invalid step dependency')
        if i and not s['after']: raise ValueError('later steps need explicit dependency; no invented parallel sequence')
    missing={r for r,k in roles.items() if k=='action'}-used
    if missing: raise ValueError('unrepresented actions: '+str(sorted(missing)))
    return raw


def audit_procedure(units, assignments, procedure):
    """Conservative alarms only; flags never manufacture actions or conditions."""
    roles={a['unit_id']:a['role'] for a in assignments}
    steps=procedure['steps']; represented={r for s in steps for r in s['action_refs']}
    alarms=[]
    for n,u in enumerate(units):
        r=u['unit_id']; text=u['quote']; role=roles[r]
        if role=='action' and r not in represented:
            alarms.append({'unit_id':r,'issue':'unrepresented_action'})
        if re.search(r'^(?:然后|再|那么|接着|随后).*(?:启动|重启|点击|取消|保存|运行|打开|关闭|设置)',text) and role not in ['action','condition']:
            alarms.append({'unit_id':r,'issue':'possible_missing_continuation_action'})
        if re.search(r'如果|若是|当.*时',text) and role not in ['condition','housekeeping','overview','unresolved']:
            alarms.append({'unit_id':r,'issue':'possible_misclassified_condition'})
    # A branch condition applies to subsequent continuation actions until a result/new task.
    active=[]
    step_by_action={r:s for s in steps for r in s['action_refs']}
    for u in units:
        r=u['unit_id']; role=roles[r]
        if role=='condition': active.append(r)
        elif role in ['result','overview','housekeeping']: active=[]
        elif role=='action' and r in step_by_action:
            if set(active)-set(step_by_action[r]['condition_refs']):
                alarms.append({'unit_id':r,'issue':'branch_condition_not_carried','condition_refs':active.copy()})
    return alarms


def materialize_units(packet, assignments):
    classify({'assignments':assignments},packet['units'])
    rows=[]
    for u,a in zip(packet['units'],assignments):
        core={'schema_version':3,'source_identity':packet['source_identity'],
              'subject':packet['subject'],'packet_id':packet['packet_id'],
              'packet_sha256':hash_object(packet),**u,'role':a['role'],
              'origin':'model_role_exact_source_span','observed_execution':False}
        rows.append({**core,'fact_id':'U'+hash_object(core)[:18]})
    return rows


def validate_checks(data, units):
    rows=[r.model_dump() for r in Checks.model_validate(data).checks]
    if [r['unit_id'] for r in rows] != [u['unit_id'] for u in units]:
        raise ValueError('check every source unit once in order')
    return rows
