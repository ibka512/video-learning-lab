"""Independent object properties and append-only observation history."""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction
from fact_ledger import hash_object

VISIBILITY = ['present', 'absent', 'unknown']
WINDOWS = ['settings', 'running_game', 'editor_only', 'unknown']


def response_schema(task):
    properties = {key: {'type':'string','enum': WINDOWS if key=='window_kind' else VISIBILITY}
                  for key in task['properties']}
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def validate_response(data, task):
    if not isinstance(data,dict) or set(data)!=set(task['properties']):
        raise ValueError('property scope mismatch')
    for key,value in data.items():
        if value not in (WINDOWS if key=='window_kind' else VISIBILITY):
            raise ValueError('invalid property value')
    return data


def bind_observation(task, values, receipt, method, view):
    validate_response(values,task)
    source=task['source']
    if abs(float(Fraction(source['time_base'])*source['pts'])-source['time'])>1e-6:
        raise ValueError('source PTS mismatch')
    obs={'task_id':task['task_id'],'object_scope':task['object_scope'],
         'source':deepcopy(source),'values':deepcopy(values),'receipt':receipt,
         'method':method,'view':deepcopy(view),'semantic_approval':False}
    obs['observation_id']=hash_object(obs)
    return obs


def validate_binding(observation):
    body={k:v for k,v in observation.items() if k!='observation_id'}
    if observation.get('observation_id')!=hash_object(body):
        raise ValueError('observation altered after binding')


def start_record(initial):
    validate_binding(initial)
    return {'initial':deepcopy(initial),'additional':[], 'semantic_approval':False}


def append_evidence(record, observation, purpose):
    """Preserve initial values even if many later observations agree on another answer."""
    validate_binding(record['initial']); validate_binding(observation)
    original=record['initial']
    if any(observation[key]!=original[key] for key in ['task_id','object_scope','source']):
        raise ValueError('evidence source/object scope mismatch')
    if set(observation['values'])!=set(original['values']):
        raise ValueError('evidence property mismatch')
    result=deepcopy(record)
    ids={original['observation_id'], *(e['observation']['observation_id'] for e in record['additional'])}
    if observation['observation_id'] in ids:
        return result
    result['additional'].append({'purpose':purpose,'observation':deepcopy(observation)})
    return result


def summarize_record(record):
    observations=[record['initial']]+[e['observation'] for e in record['additional']]
    for obs in observations: validate_binding(obs)
    fields={}
    for key,initial in record['initial']['values'].items():
        candidates=sorted({o['values'][key] for o in observations if o['values'][key]!='unknown'})
        if len(candidates)>1:
            status='unreviewed_conflict'
        elif initial=='unknown' and candidates:
            status='unreviewed_additional_candidate'
        elif candidates:
            status='unreviewed_candidate'
        else:
            status='unknown'
        fields[key]={'initial_value':initial,'candidate_values':candidates,'status':status,
                     'automatic_value':None, 'support':{v:[o['observation_id'] for o in observations if o['values'][key]==v] for v in candidates}}
    return {'task_id':record['initial']['task_id'],'properties':fields,'semantic_approval':False,
            'observations_are_not_independent_validators':True}


def needs_reread(a,b):
    return [key for key in a['values'] if a['values'][key]!=b['values'][key] or a['values'][key]=='unknown']


def render_properties(values,task):
    words={'present':'可见','absent':'未见','unknown':'无法确定','settings':'项目设置窗口',
           'running_game':'游戏运行窗口','editor_only':'编辑器本体'}
    return '；'.join(f"{task['properties'][key]}：{words[value]}" for key,value in values.items())
