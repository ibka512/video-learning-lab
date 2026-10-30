"""Finish this pilot after the OCR worker exits successfully. Does not launch OCR."""
from pathlib import Path
import json,subprocess
project=Path(__file__).resolve().parents[1];root=project/'outputs/OCR局部复核接入-2026-10-28';base=project/'outputs/第二轮视频验证-2026-10-27';bench=project/'outputs/PaddleOCR对照验证-2026-10-28'
plan=json.loads((root/'sources/route-plan.json').read_text());metrics=json.loads((root/'sources/metrics.json').read_text())
assert metrics['jobs']==len(plan['jobs']) and metrics['all_planned_regions_processed'],'OCR worker must finish first'
for job in plan['jobs']:
 receipt=json.loads((root/'receipts'/(job['job_id']+'.json')).read_text());assert receipt['status']=='success' and receipt['identity']['image_sha256']==job['image_sha256']
subprocess.run(['python3',str(project/'pipeline/build_refined_evidence.py'),'--root',str(root),'--baseline',str(base),'--benchmark',str(bench)],check=True)
common=[str(project/'work/pilot-20261026/.venv/bin/python'),str(project/'pipeline/extract_knowledge.py'),'--root',str(root),'--source',str(base),'--refinement-root',str(root),'--model',str(project/'work/pilot-20261026/models/Qwen3-VL-4B-Instruct-4bit')]
subprocess.run(common+['--prepare-only'],check=True)
print('All source interval contexts compiled; now one affected interval end-to-end',flush=True)
subprocess.run(common+['--segment-id','SC0004'],check=True)
for script in ['evaluate_ocr_review.py','audit_ocr_review.py']:
 flag='--benchmark' if script.startswith('evaluate') else '--baseline';path=bench if script.startswith('evaluate') else base
 subprocess.run(['python3',str(project/'pipeline'/script),'--root',str(root),flag,str(path)],check=True)
print('COMPLETE',flush=True)
