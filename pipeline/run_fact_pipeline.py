"""Active fact-ledger entry point. Run in the existing MLX/Pydantic environment."""
from pathlib import Path
import argparse,subprocess,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--ocr',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--domain',required=True);p.add_argument('--model',required=True,type=Path);p.add_argument('--state-reviews',type=Path);p.add_argument('--max-attempts',type=int,default=2);a=p.parse_args();scripts=Path(__file__).resolve().parent
    def run(name,*args):subprocess.run([sys.executable,str(scripts/name),*map(str,args)],check=True)
    extra=['--state-reviews',a.state_reviews] if a.state_reviews else []
    run('prepare_fact_packets.py','--baseline',a.baseline,'--ocr',a.ocr,'--out',a.out,'--domain',a.domain,*extra)
    run('run_fact_extraction.py','--root',a.out,'--model',a.model,'--max-attempts',a.max_attempts)
    run('finalize_fact_ledger.py','--root',a.out)
    run('export_fact_markdown.py','--root',a.out)
if __name__=='__main__':main()
