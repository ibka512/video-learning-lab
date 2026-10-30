"""Freeze property targets before inference. References are never imported by the runner."""
import json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path('pipeline').resolve()))
from ocr_routing import dump,digest,read_jsonl,write_jsonl

root=Path('outputs/对象属性验证-v6-2026-10-30')
if (root/'references/freeze.json').exists(): raise SystemExit('Already frozen')
old=Path('outputs/声画理解对照-v4-2026-10-29')
cases=read_jsonl(old/'sources/cases.public.jsonl'); tasks=[]; targets=[]
definitions={
 'window_kind':'中央最前方的主要软件窗口类型。settings=项目设置；running_game=有独立运行窗口标题栏的游戏；editor_only=编辑器本体，没有覆盖其上的独立运行/设置窗口。背后编辑器仍可见不等于前景是编辑器。',
 'checkerboard':'中央演示区域内密集灰白棋盘格背景。',
 'pieces':'中央演示区域中分离的、散落的不规则碎片。',
 'hand':'中央演示区域中真实手掌/手指形状，部分被遮挡仍可记可见；忽略编辑器侧栏小图标。',
 'blue_circle':'中央演示区域中蓝色发光圆形/圆环效果，部分被碎片遮挡仍可记可见。',
 'godot_logo':'中央主内容中Godot启动页的大标志与GODOT文字；忽略工具栏或标题栏的小图标。',
 'solid_triangles':'中央演示区域中明显实心深色三角形；只有细淡轮廓不算。',
 'first_checked':'菜单第一项“下次运行时嵌入游戏”的勾选标记。只读当前状态，不推断取消动作。'}
visibility={
 'B':[{'checkerboard':x} for x in ['absent','present','present','absent']],
 'F':[{'pieces':p,'hand':h} for p,h in [('absent','absent'),('absent','absent'),('present','present'),('present','present')]],
 'C':[{'pieces':p,'blue_circle':c} for p,c in [('absent','absent'),('absent','absent'),('present','present'),('present','present')]],
 'T':[{'godot_logo':l,'solid_triangles':t} for l,t in [('absent','absent'),('present','absent'),('absent','absent'),('absent','present')]],
 'M':[{'first_checked':'present'} for _ in range(4)]}
window={'W':['settings','settings','running_game','running_game'],
        'T':['editor_only','running_game','running_game','running_game']}
for case in cases:
    cid=case['case_id']
    for pos,source in enumerate(case['frames']):
        groups=[]
        if cid in window: groups.append(('foreground_window',{'window_kind':window[cid][pos]}))
        if cid in visibility: groups.append(('menu_first_item' if cid=='M' else 'demonstration_content',visibility[cid][pos]))
        for scope,expected in groups:
            task_id=f'{cid}-{pos}-{scope}'
            task={'task_id':task_id,'case_id':cid,'position':pos,'object_scope':scope,'source':source,
                  'properties':{key:definitions[key] for key in expected},'cross_frame_identity_verified':False}
            tasks.append(task);targets.append({'task_id':task_id,'expected':expected,'crop_sha256':source['crop_sha256']})
write_jsonl(root/'sources/tasks.public.jsonl',tasks)
shutil.copy2(old/'sources/source-identity.json',root/'sources/source-identity.json')
dump(root/'references/properties.json',{'reviewer':'Codex exact-source image review before inference; development data, not independent human gold',
 'annotation_notes':['All targets are observable attributes, not mutually exclusive primary states.',
                     'F-2 includes a partly obscured hand; C-2 includes partly obscured blue circle.',
                     'T-2 has faint triangle outlines, not solid dark triangles.',
                     'Scope labels are task-defined search scopes, not automatically discovered or tracked objects.'],
 'targets':targets})
dump(root/'protocol.json',{'status':'frozen before inference','date':'2026-10-30','model':'existing Qwen3-VL-4B-Instruct-4bit',
 'tasks':len(tasks),'property_targets':sum(len(t['properties']) for t in tasks),'source_positions':24,
 'arms':['temporal: one still repeated twice with existing video encoder','native: same still through native image encoder'],
 'control':'Same task text, schema, image pixels, pixel budget 524288, no transcript or previous outputs. Assert matching resized geometry; record actual pixel tensor differences.',
 'task_decomposition':'8 one-property window tasks plus 20 content/menu tasks with at most two independent attributes. No generated boxes, titles, descriptions or redundant certainty labels.',
 'hypotheses':['Independent properties retain coexistence that primary-state choices lose.',
               'Smaller tasks separate content perception from window identity.',
               'Native single-image encoding may differ from the existing repeated-still video path.'],
 'reread':'For disagreements/unknown between the two encodings, append one native full-frame read with the target area normalized to current image. Maximum 12 tasks, in frozen task order. No voting or overwriting.',
 'base_calls':2*len(tasks),'max_reread_calls':12,'semantic_approval':False,
 'gates':['Initial evidence and raw model output never replaced by rereads.','Unknown and absence distinguished.','Coexisting positive properties remain simultaneously representable.',
          'Native/temporal comparison uses identical prompt/schema/crop and logs consumed tensors.','Report positive recall, false positives, unknowns, identity accuracy and cost separately.'],
 'limits':['Existing six cases from one development video; no new-video accuracy.',
           'v5 versus v6 changes tasks and schema together; do not attribute difference to a single factor.',
           'No automatic object discovery, precise spatial detection, full-video event or action verification.']})
tracked=[root/'protocol.json',root/'references/properties.json',*sorted((root/'sources').glob('*'))]
dump(root/'references/freeze.json',{str(p.relative_to(root)):digest(p) for p in tracked})
print({'root':str(root),'tasks':len(tasks),'targets':sum(len(t['properties']) for t in tasks)})
