"""Freeze source-backed development references BEFORE inference, never import from runner."""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path('pipeline').resolve()))
from ocr_routing import dump, digest, read_jsonl, write_jsonl

ROOT = Path('outputs/逐帧证据验证-v5-2026-10-29')
OLD = Path('outputs/声画理解对照-v4-2026-10-29')
if (ROOT / 'references/freeze.json').exists():
    raise SystemExit('Already frozen; do not overwrite an experiment.')
cases = read_jsonl(OLD / 'sources/cases.public.jsonl')
write_jsonl(ROOT / 'sources/cases.public.jsonl', cases)
labels = dict(settings='项目设置窗口', running_game='游戏运行窗口', editor='编辑器',
              checkerboard='运行区域显示棋盘格', not_checkerboard='运行区域未显示棋盘格',
              intact_background='背景图主体完整', scattered_fragments='背景主体呈散落碎片',
              hand_revealed='主要露出手部内容', blue_circular_effect='主要露出蓝色圆形效果',
              editor_preview='编辑器内预览', godot_splash='Godot启动画面',
              running_background='运行窗口内的背景图', running_triangular_shapes='运行区域出现明显三角形形状',
              checked='菜单第一项已勾选', unchecked='菜单第一项未勾选', unknown='无法确定')
targets = {
    'W': ('中央最前方的主要软件窗口。区分前景运行窗口与背后的编辑器，按可见标题/边框判断。', ['settings','running_game','editor']),
    'B': ('主要运行区域是否显示密集灰白棋盘格。', ['checkerboard','not_checkerboard']),
    'F': ('主要演示内容的状态。碎片和手可能共存，按当前主体状态选择；完整背景不等于编辑器小预览。', ['intact_background','scattered_fragments','hand_revealed']),
    'T': ('主要软件内容：编辑器预览、Godot启动画面、运行窗口背景、或明显三角形形状。', ['editor_preview','godot_splash','running_background','running_triangular_shapes']),
    'M': ('菜单第一项“下次运行时嵌入游戏”的勾选状态。只读当前状态。', ['checked','unchecked']),
    'C': ('主要演示内容的状态。碎片与蓝色图形可能共存，按当前主体状态选择。', ['intact_background','scattered_fragments','blue_circular_effect']),
}
sequence_rules = {
    'W': [('settings_to_game',['settings','running_game']), ('game_to_settings',['running_game','settings']),
          *[('no_change',[s]) for s in targets['W'][1]]],
    'F': [('intact_fragments_hand',['intact_background','scattered_fragments','hand_revealed']),
          ('hand_fragments_intact',['hand_revealed','scattered_fragments','intact_background']), ('intact_only',['intact_background'])],
    'C': [('intact_fragments_blue_circle',['intact_background','scattered_fragments','blue_circular_effect']),
          ('blue_circle_fragments_intact',['blue_circular_effect','scattered_fragments','intact_background']), ('intact_only',['intact_background'])],
    'T': [('editor_splash_triangles',['editor_preview','godot_splash','running_triangular_shapes']),
          ('triangles_splash_editor',['running_triangular_shapes','godot_splash','editor_preview']), ('splash_only',['godot_splash'])],
}
specs = []
for case in cases:
    cid = case['case_id']; task, states = targets[cid]
    requirements = {}
    if cid == 'W':
        requirements = {s: {'kind':[s], 'layer':['foreground']} for s in states}
    if cid == 'T':
        requirements = {'editor_preview': {'kind':['editor'], 'layer':['foreground']},
                        **{s: {'kind':['running_game','content'], 'layer':['foreground']} for s in states if s != 'editor_preview'}}
    spec = {'case_id':cid, 'target':task, 'states':states+['unknown'],
            'labels':{s:labels[s] for s in states+['unknown']},
            'adapter': {'B':'checkerboard','M':'checkbox'}.get(cid,'sequence'),
            'endpoint_choices':case['questions'][0]['choices'], 'object_requirements':requirements,
            'sequence_rules':[{'value':v,'states':ss, 'ignore_states':['running_background'] if cid=='T' else []}
                              for v,ss in sequence_rules.get(cid,[])]}
    specs.append(spec)
write_jsonl(ROOT / 'sources/tasks.public.jsonl', specs)
shutil.copy2(OLD / 'sources/source-identity.json', ROOT / 'sources/source-identity.json')

# Reviewed against all six exact crop contact sheets before any v5 model invocation.
gold = {'W':['settings','settings','running_game','running_game'],
        'B':['not_checkerboard','checkerboard','checkerboard','not_checkerboard'],
        'F':['intact_background','intact_background','scattered_fragments','hand_revealed'],
        'T':['editor_preview','godot_splash','running_background','running_triangular_shapes'],
        'M':['checked']*4,
        'C':['intact_background','intact_background','scattered_fragments','blue_circular_effect']}
dump(ROOT / 'references/frame-states.json', {
    'reviewer':'Codex source-image review, not independent human annotation',
    'scope':'Primary target state on the exact v4 crop; fragments/hand/circle may coexist. No claim of absence of other objects.',
    'cases':[{'case_id':c['case_id'], 'frames':[{'position':i, 'state':gold[c['case_id']][i],
                 'crop_sha256':f['crop_sha256']} for i,f in enumerate(c['frames'])]} for c in cases]})
shutil.copy2(OLD / 'references/visual-gold.json', ROOT / 'references/v4-questions.json')
dump(ROOT / 'protocol.json', {
    'status':'frozen before v5 inference', 'development_regression':True,
    'model':'existing Qwen3-VL-4B-Instruct-4bit', 'max_pixels_per_still':524288,
    'arms':{'A':'Reuse immutable v4 outputs.',
            'B':'Joint four stills: visual_only, aligned, reversed(aligned), unified observations.',
            'C':'24 independently observed still positions; no transcript. Same repeated-still temporal encoding as B.',
            'D':'C with triggered source rereads. Uses B visual_only only to detect disagreements.'},
    'base_calls':42, 'max_review_calls':48, 'max_rounds_per_frame':2,
    'review_trigger':['single unknown','program-checkable state/object conflict','B visual_only versus C state disagreement'],
    'review_views':['original full frame at up to 1048576 pixels, target area described by coordinates',
                    'initial single-image selected object region + 15 percent margin, mapped to native source; original crop fallback'],
    'review_accept':'R1 agrees with usable C, or R1 and R2 agree with each other; otherwise unknown. Agreement is not semantic approval.',
    'stop':['corroboration','two rounds','total 48 requests','no new view'],
    'no_gold_in_inference':True,
    'evaluation':['raw and gated frame state correctness/error/unknown','v4 question compatibility','required and extra evidence positions',
                  'B aligned/reversed paired accuracy','C/D reversed sorting is program-only, excluded from model temporal score',
                  'trigger coverage against observed errors, resolved/unresolved, wrong corroborations',
                  'call counts, actual tensor dimensions, generation time, review changes'],
    'limits':['Source crops and cases are manually selected.','Task choices/sequence adapter are benchmark specific.',
              'B versus A changes schema, prompt and task decomposition together.',
              'D full frame adds context including peripheral subtitles; not a pure resolution comparison.',
              'No new-video, continuous-action, whole-favorites or downstream application claim.']})
tracked = [ROOT/'protocol.json', *sorted((ROOT/'sources').glob('*.json*')), *sorted((ROOT/'references').glob('*.json'))]
dump(ROOT/'references/freeze.json', {str(p.relative_to(ROOT)):digest(p) for p in tracked})
print(ROOT)
