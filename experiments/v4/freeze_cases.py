import sys,json
from pathlib import Path
sys.path.insert(0,'pipeline')
from ocr_routing import dump,read_jsonl,write_jsonl,digest
r=Path('outputs/声画理解对照-v4-2026-10-29')
assets=read_jsonl(r/'sources/cases.assets.jsonl')
speech=read_jsonl(r/'sources/speech.jsonl')
def q(qid,text,choices,answer,reverse,support):
 return dict(question_id=qid,text=text,choices=choices+['unknown'],answer=answer,reversed_answer=reverse,support_positions=support)
scene=['settings','running_game','editor']
configs={
'W':[
 q('first','第0个画面中央最前面的主要窗口是什么？settings项目设置；running_game游戏运行窗口；editor编辑器。',scene,'settings','running_game',[0]),
 q('last','第3个画面中央最前面的主要窗口是什么？',scene,'running_game','settings',[3]),
 q('order','按呈现顺序，主要窗口是怎样变化的？',['settings_to_game','game_to_settings','no_change'],'settings_to_game','game_to_settings',[0,3])],
'B':[
 q('first','第0个画面的运行区域是否显示密集灰白棋盘格？',['checkerboard','not_checkerboard'],'not_checkerboard','not_checkerboard',[0]),
 q('middle','第1和第2个画面的运行区域是否显示密集灰白棋盘格？',['both_checkerboard','neither_checkerboard','only_one'],'both_checkerboard','both_checkerboard',[1,2]),
 q('last','第3个画面的运行区域是否显示密集灰白棋盘格？',['checkerboard','not_checkerboard'],'not_checkerboard','not_checkerboard',[3])],
'F':[
 q('first','第0个画面主要演示处于什么状态？',['intact_background','scattered_fragments','hand_revealed'],'intact_background','hand_revealed',[0]),
 q('last','第3个画面主要演示处于什么状态？',['intact_background','scattered_fragments','hand_revealed'],'hand_revealed','intact_background',[3]),
 q('order','按呈现顺序，演示的变化是什么？',['intact_fragments_hand','hand_fragments_intact','intact_only'],'intact_fragments_hand','hand_fragments_intact',[0,2,3])],
'T':[
 q('first','第0个画面的主要内容是什么？',['editor_preview','godot_splash','running_triangular_shapes'],'editor_preview','running_triangular_shapes',[0]),
 q('last','第3个画面的主要内容是什么？',['editor_preview','godot_splash','running_triangular_shapes'],'running_triangular_shapes','editor_preview',[3]),
 q('order','按呈现顺序，哪种变化与画面一致？',['editor_splash_triangles','triangles_splash_editor','splash_only'],'editor_splash_triangles','triangles_splash_editor',[0,1,3])],
'M':[
 q('first','第0个画面菜单第一项“下次运行时嵌入游戏”是否勾选？',['checked','unchecked'],'checked','checked',[0]),
 q('last','第3个画面菜单第一项是否勾选？',['checked','unchecked'],'checked','checked',[3]),
 q('change','仅凭所给四个画面，能否确认取消勾选已经发生？static_checked仅见仍勾选；cancel_observed看到了取消前后变化。',['static_checked','cancel_observed'],'static_checked','static_checked',[0,3])],
'C':[
 q('first','第0个画面演示区域主要是什么？',['intact_background','scattered_fragments','blue_circular_effect'],'intact_background','blue_circular_effect',[0]),
 q('last','第3个画面露出的主要图形是什么？',['intact_background','scattered_fragments','blue_circular_effect'],'blue_circular_effect','intact_background',[3]),
 q('order','按呈现顺序，演示如何变化？',['intact_fragments_blue_circle','blue_circle_fragments_intact','intact_only'],'intact_fragments_blue_circle','blue_circle_fragments_intact',[0,2,3])]
}
public=[];gold=[]
for a in assets:
 cid=a['case_id'];lo,hi=a['source_interval']
 segments=[s for s in speech if s['end']>=lo and s['start']<=hi]
 public.append({**a,'questions':[{k:q[k] for k in ['question_id','text','choices']} for q in configs[cid]],
                'transcript':[{'ref':s['ref'],'text':s['text']} for s in segments],
                'frames_selected_by':'Codex direct source-image review','permutation':[3,2,1,0]})
 gold.append({'case_id':cid,'questions':configs[cid]})
write_jsonl(r/'sources/cases.public.jsonl',public)
dump(r/'references/visual-gold.json',{'review_basis':'Codex viewed exact source crops before model calls; development regression, not independent human gold',
     'cases':gold,'scope':'Only selected four images, not proof of unseen clicks or full interval','frozen_before_inference':True})
# Fixed semantic groups: explicit source headings; no target text or prior roles supplied.
old=Path('outputs/步骤完整性-v3-2026-10-29/r5/BV1Xato6REfH')
packets=read_jsonl(old/'sources/packets.jsonl')
ps=[{k:p[k] for k in ['packet_id','source_identity','subject','speech','units']} for p in packets if p['mode']=='concept']
write_jsonl(r/'sources/semantic.public.jsonl',ps)
for source,name in [(Path('outputs/步骤完整性-v3-2026-10-29/references/frontend.reference.json'),'semantic-gold.json'),
                    (Path('outputs/独立视频验证-BV1Xato6REfH/sources/holdout-targets.frozen.json'),'semantic-targets.json')]:
 (r/'references'/name).write_bytes(source.read_bytes())
dump(r/'references/freeze.json',{p.name:digest(p) for p in (r/'references').glob('*.json')})
