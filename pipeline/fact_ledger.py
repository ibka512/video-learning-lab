"""Source-grounded atomic facts. Rule validation is not semantic approval."""
from __future__ import annotations
import hashlib,json,re
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,ValidationError

PATH=re.compile(r'(?<![A-Za-z0-9_/])[A-Za-z][A-Za-z0-9_]*(?:/[A-Za-z0-9_]+){2,}(?![A-Za-z0-9_/])')
HEADER=re.compile(r'(?:设置\s*)?('+PATH.pattern+r')\s*:\s*(bool|int|float|string|String)\s*=\s*(.+)$')
KINDS=('setting_key','help_default','help_statement','author_statement','observed_state')
class Citation(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    ref:str=Field(min_length=1)
    quote:str=Field(min_length=1)
class Fact(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    kind:Literal['setting_key','help_default','help_statement','author_statement','observed_state']
    value:str=Field(min_length=1)
    citations:list[Citation]=Field(min_length=1,max_length=6)
class Extraction(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    facts:list[Fact]=Field(max_length=8)
    overflow:bool

def hash_object(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def validate_fact(data,packet):
    """Return errors and exact offsets; never repair punctuation or paraphrase quotes."""
    try:f=Fact.model_validate(data)
    except ValidationError as e:return ['schema: '+str(e)],[]
    evidence={e['ref']:e for e in packet['evidence']};errors=[];spans=[];used=[]
    if f.kind not in packet['allowed_kinds']:errors.append('fact_type_not_allowed_for_packet')
    for c in f.citations:
        e=evidence.get(c.ref)
        if not e:errors.append('unknown_reference:'+c.ref);continue
        start=e['text'].find(c.quote)
        if start<0:errors.append('quote_not_exact:'+c.ref);continue
        if e['text'].find(c.quote,start+1)>=0:errors.append('ambiguous_quote:'+c.ref);continue
        spans.append({'ref':c.ref,'char_start':start,'char_end':start+len(c.quote),'quote':c.quote})
        used.append((e,c.quote))
    quotes='\n'.join(q for _,q in used)
    if f.kind=='setting_key':
        if not PATH.fullmatch(f.value) or f.value!=packet.get('subject') or not any(f.value in q for _,q in used):errors.append('setting_key_not_exact_or_wrong_subject')
    elif f.kind=='help_default':
        matches=[HEADER.search(q.strip()) for e,q in used if e['modality']=='ocr_help'];matches=[m for m in matches if m]
        if not any(m.group(1)==packet.get('subject') and m.group(3).strip()==f.value for m in matches):errors.append('default_not_bound_to_exact_header')
        if any(m.group(2)=='bool' for m in matches) and f.value not in ('true','false'):errors.append('invalid_boolean_not_auto_corrected')
    elif f.kind=='help_statement':
        # A full help body is an atomic evidence unit: do not drop the sentence ending.
        bodies=[e['text'] for e,_ in used if e['modality']=='ocr_help' and e.get('role')=='body']
        if f.value not in bodies or not any(f.value==q for _,q in used):errors.append('help_body_must_be_complete_verbatim')
    elif f.kind=='author_statement':
        if not used or any(e['modality']!='speech' for e,_ in used):errors.append('author_statement_requires_speech')
        if f.value not in [e['text'] for e,_ in used] or f.value not in [q for _,q in used]:errors.append('author_statement_must_preserve_complete_source_segment')
    elif f.kind=='observed_state':
        if not any(e['modality']=='reviewed_state' and e.get('subject')==packet.get('subject') and e.get('observed_value')==f.value and e.get('source_image_sha256') and e.get('frame_id') and q==e['text'] for e,q in used):errors.append('state_requires_object_specific_reviewed_frame')
    if f.kind.startswith('help_') and any(e.get('subject')!=packet.get('subject') for e,_ in used):errors.append('help_subject_mismatch')
    return sorted(set(errors)),spans

def evaluate_response(response,packet):
    try:parsed=Extraction.model_validate(response)
    except ValidationError as e:return [],[{'errors':['response_schema:'+str(e)]}],{'missing_kinds':packet['required_kinds'],'overflow':False}
    valid=[];invalid=[];seen=set()
    for f in parsed.facts:
        d=f.model_dump();errors,spans=validate_fact(d,packet);identity=(d['kind'],d['value'])
        if identity in seen:errors.append('duplicate_fact')
        seen.add(identity)
        if errors:invalid.append({'fact':d,'errors':errors})
        else:
            refs={c['ref'] for c in d['citations']};used=[e for e in packet['evidence'] if e['ref'] in refs]
            trusted=all(e.get('review_status')=='source_reviewed' for e in used)
            valid.append({**d,'fact_id':'K'+hash_object([packet['packet_id'],d])[:18],
                          'packet_id':packet['packet_id'],'subject':packet.get('subject'),
                          'source_identity':packet['source_identity'],'source_spans':spans,
                          'source_locations':[{k:e[k] for k in ('ref','start','end','frame_id','bbox','source_image','source_image_sha256','review_status') if k in e} for e in used],
                          'review_status':'source_supported_quote' if trusted else 'pending_source_review',
                          'validation_status':'exact_evidence_rules_passed','semantic_paraphrase_generated':False})
    missing=sorted(set(packet['required_kinds'])-{f['kind'] for f in valid})
    return valid,invalid,{'missing_kinds':missing,'overflow':parsed.overflow}
