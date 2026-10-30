"""Existing MLX runtime, bounded repair, immutable stage receipts."""
import json
import os
import time
from pathlib import Path

from fact_ledger import hash_object
from ocr_routing import digest, dump


class Runtime:
    def __init__(self, model_path, out):
        self.path, self.out = model_path.resolve(), out.resolve()
        self.loaded = None
        self.fresh = self.hits = 0
        self.wall = 0.0
        self.revision = json.loads((self.path / 'pilot-download-receipt.json').read_text())['revision']

    def call(self, stage, instruction, payload, schema, validate, max_tokens=2200):
        feedback = None
        for attempt in range(1, 3):
            request = {'instruction': instruction, 'input': payload, 'repair_errors': feedback}
            identity = {'request_sha256': hash_object(request), 'schema_sha256': hash_object(schema),
                        'model_path': str(self.path), 'model_revision': self.revision,
                        'temperature': 0, 'max_tokens': max_tokens,
                        'code': {p.name: digest(p) for p in [Path(__file__),
                            Path(__file__).with_name('knowledge_schema.py'),
                            Path(__file__).with_name('run_knowledge_pipeline.py')]}}
            receipt = self.out / 'runs' / f'{stage}-attempt-{attempt}.json'
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record['identity'] != identity:
                    raise ValueError('changed inference identity: use a new output directory')
                self.hits += 1
            else:
                if self.loaded is None:
                    os.environ['HF_HOME'] = str(self.path.parent.parent / 'cache/huggingface')
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    import mlx.core as mx
                    from mlx_vlm import load, generate
                    from mlx_vlm.structured import build_json_schema_logits_processor
                    model, processor = load(str(self.path), trust_remote_code=False)
                    self.loaded = model, processor, generate, build_json_schema_logits_processor, mx
                model, processor, generate, builder, mx = self.loaded
                prompt = processor.apply_chat_template([{'role': 'user', 'content': json.dumps(request, ensure_ascii=False)}],
                                                        tokenize=False, add_generation_prompt=True)
                tick = time.monotonic()
                try:
                    result = generate(model, processor, prompt=prompt, temperature=0,
                                      max_tokens=max_tokens, verbose=False,
                                      logits_processors=[builder(processor.tokenizer, schema)])
                    raw, finish = result.text, result.finish_reason
                    record = {'identity': identity, 'request': request, 'raw_output': raw,
                              'finish_reason': finish, 'output': json.loads(raw) if finish != 'length' else None}
                    del result
                    mx.clear_cache()
                except Exception as exc:
                    record = {'identity': identity, 'request': request, 'output': None,
                              'error': type(exc).__name__ + ': ' + str(exc)}
                record['wall_seconds'] = time.monotonic() - tick
                self.wall += record['wall_seconds']
                dump(receipt, record)
                self.fresh += 1
            try:
                value = validate(record['output'])
                print(json.dumps({'stage': stage, 'attempt': attempt, 'valid': True}, ensure_ascii=False), flush=True)
                return value
            except (ValueError, TypeError, KeyError) as exc:
                feedback = str(exc)
                dump(self.out / 'validation' / f'{stage}-attempt-{attempt}.json',
                     {'receipt_sha256': digest(receipt), 'error': feedback})
                print(json.dumps({'stage': stage, 'attempt': attempt, 'valid': False, 'error': feedback}, ensure_ascii=False), flush=True)
        raise ValueError(f'{stage}: bounded repair exhausted: {feedback}')
