"""Check the small public checkout, never the maintainer's private source workspace."""
from pathlib import Path
import json,re,subprocess,sys

root=Path(__file__).resolve().parents[1]
patterns={
 'machine_path':re.compile(r'/(?:Users|Volumes)/|/(?:private/)?var/folders/|[A-Z]:\\Users\\'),
 'credential':re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}'),
 'private_assignment':re.compile(r'(?i)(?:SESSDATA|bili_jct|authorization|password|api_key)\s*[=:]\s*[\"\x27]?[A-Za-z0-9_%+/.-]{12,}'),
 'personal_tracking':re.compile(r'[?&](?:vd_source|spm_id_from|access_key|deadline|upsig|wsSecret|wsTime)=')}
blocked_suffixes={'.safetensors','.gguf','.m4s','.mp4','.mp3','.wav','.zip','.png','.jpg','.jpeg','.webp','.pem','.key'}
blocked_dirs={'.venv','work','outputs','local-data','models','media','cache'}
failures=[];count=0;total=0
for path in root.rglob('*'):
    relative=path.relative_to(root)
    if any(part in {'.git','__pycache__'} for part in relative.parts) or not path.is_file():continue
    count+=1;total+=path.stat().st_size
    if any(part in blocked_dirs for part in relative.parts) or path.suffix.lower() in blocked_suffixes:
        failures.append((str(relative),'excluded artifact'));continue
    if path.stat().st_size>2_000_000:failures.append((str(relative),'unexpected large file'))
    try:text=path.read_text()
    except UnicodeError:failures.append((str(relative),'binary file'));continue
    # The checker contains its own patterns, never an actual credential.
    if path.name=='check_publication.py':continue
    for key,pattern in patterns.items():
        for match in pattern.finditer(text):
            failures.append((str(relative),key+':line '+str(text.count('\n',0,match.start())+1)))
    if path.suffix=='.md':
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=link.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith('mailto:'):continue
            if not (path.parent/target).exists():failures.append((str(relative),'broken local link: '+target))
if failures:
    for path,reason in failures:print(path+': '+reason)
    sys.exit(1)
print(json.dumps({'status':'passed','text_files':count,'bytes':total,'media_and_weights':0},ensure_ascii=False))
