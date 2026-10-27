"""Populate the NeurIPS report from observed JSON outputs, then compile with Tectonic."""
import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from data import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('runs/vae_cpu'))
    p.add_argument('--tectonic',default='tectonic')
    p.add_argument('--observation',required=True,help='Plain English visual observation after reviewing generated images')
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    paper=root/'report'/'paper'
    h=json.loads((a.run/'history.json').read_text())
    summary=json.loads((a.run/'training_summary.json').read_text())
    metrics=json.loads((a.run/'metrics.json').read_text())
    if len(h)!=30 or metrics['real_count']!=624 or metrics['generated_count']!=624:
        raise ValueError('Report prose assumes 30 epochs and 624 test/generated samples; update prose for a different protocol')
    if metrics['checkpoint_sha256']!=file_hash(a.run/'best.pt') or metrics['checkpoint_epoch']!=summary['best_epoch']:
        raise ValueError('Checkpoint/evaluation mismatch')
    vals={'BestVal':f"{summary['best_val_total']:.2f}",'BestEpoch':str(summary['best_epoch']),
          'FIDValue':f"{metrics['fid_2048']:.2f}",'ISValue':f"{metrics['inception_score']:.3f}",
          'TrainMinutes':f"{summary['elapsed_seconds']/60:.1f}",
          'FirstTrain':f"{h[0]['train']['total']:.2f}",'LastTrain':f"{h[-1]['train']['total']:.2f}",
          'LastKL':f"{h[-1]['train']['kl']:.2f}"}
    if not all(math.isfinite(float(v)) for v in vals.values()): raise ValueError('Non-finite results')
    if any(c in a.observation for c in r'\{}%&#_$^~'): raise ValueError('Observation must be plain text without TeX special characters')
    vals['VisualObservation']=a.observation
    (paper/'results.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in vals.items()))
    (paper/'images').mkdir(exist_ok=True)
    for name in ['loss_curves.png','samples16.png']:
        shutil.copy2(a.run/name,paper/'images'/name)
    subprocess.run([a.tectonic,'main.tex','--keep-logs'],cwd=paper,check=True)
    print(paper/'main.pdf')


if __name__=='__main__': main()
