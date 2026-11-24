"""Run missing analysis stages, or rebuild everything with --retrain."""
from pathlib import Path
import argparse,subprocess,sys

ROOT=Path(__file__).resolve().parent

def command(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def main():
    p=argparse.ArgumentParser(description='BAF Base full temporal analysis')
    p.add_argument('--retrain',action='store_true',help='Reproduce training and final scoring from scratch, overwriting artifacts')
    p.add_argument('--skip-notebook',action='store_true')
    args=p.parse_args()
    command('scripts/download_data.py')
    stages=[('src.eda','outputs/eda_summary.json'),('src.train','outputs/models/frozen_manifest.json'),
            ('src.evaluate','outputs/evaluation.json'),('src.explain','outputs/explanation.json'),
            ('src.extensions','outputs/extensions.json'),
            ('src.rolling_backtest','outputs/rolling_backtest.json'),
            ('src.coursework','outputs/coursework.json'),
            ('src.feature_study','outputs/feature_study.json')]
    for module,artifact in stages:
        if args.retrain or not (ROOT/artifact).exists(): command('-m',module)
        else: print(f'Using completed stage: {module}',flush=True)
    command('-m','src.verify')
    command('-m','src.report','--require-complete')
    if not args.skip_notebook: command('-m','src.notebook')

if __name__=='__main__': main()
