"""One entry: classify, check, recover exact duplicate output, and render."""
import sys
from pathlib import Path

from run_knowledge_pipeline import main
from recover_knowledge_duplicates import recover


if __name__ == '__main__':
    main()
    # The underlying argument parser already validates --out.
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else Path(
        next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--out=')))
    if '--export-only' not in sys.argv:
        recover(out.resolve())
