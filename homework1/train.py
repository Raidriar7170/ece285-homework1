"""Train a VAE; no test data are used for training or checkpoint selection."""
import argparse
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data import ChestDataset, file_hash
from model import VAE, loss_terms


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(3)
    n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x in loader:
            x = x.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            # Stochastic ELBO in both training and validation.
            logits, mu, logvar = model(x)
            losses = loss_terms(logits, x, mu, logvar)
            if not all(torch.isfinite(v) for v in losses):
                raise RuntimeError('Non-finite loss; aborting rather than reporting invalid results')
            if optimizer:
                losses[0].backward()
                optimizer.step()
            totals += np.array([v.item() for v in losses]) * len(x)
            n += len(x)
    return dict(zip(['total', 'reconstruction', 'kl'], (totals / n).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('runs/vae_cpu'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=285)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        p.error('epochs and batch size must be positive')
    if a.out.exists():
        p.error('Output directory already exists; use a new path to preserve prior results')
    a.out.mkdir(parents=True)
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    train, val = (ChestDataset(a.data, s) for s in ['train','val'])
    manifest = {'train': train.manifest(), 'val': val.manifest()}
    train_hashes = {r['sha256'] for r in manifest['train']}
    if train_hashes & {r['sha256'] for r in manifest['val']}:
        raise RuntimeError('Exact image duplicate across train and validation splits')
    (a.out/'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    config.update(train_count=len(train), val_count=len(val), torch=torch.__version__,
                  platform=platform.platform(), preprocessing='PIL grayscale, bilinear resize 64x64, [0,1]',
                  objective='pixel-summed BCE + latent-summed KL; beta=1; batch mean',
                  source_hashes={f: file_hash(Path(__file__).parent/f) for f in ['model.py','data.py','train.py']})
    (a.out/'config.json').write_text(json.dumps(config, indent=2))
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0)
    valid = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=0)
    model = VAE(a.latent_dim).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, history = float('inf'), []
    print(json.dumps({'train':len(train),'val':len(val),'device':a.device,'parameters':sum(v.numel() for v in model.parameters())}),flush=True)
    for epoch in range(1, a.epochs+1):
        t = time.perf_counter()
        tr = run_epoch(model, loader, a.device, optimizer)
        # Fixed validation epsilon gives consistent checkpoint comparisons without
        # changing the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(a.seed + 1)
            va = run_epoch(model, valid, a.device)
        row = {'epoch':epoch,'train':tr,'val':va,'seconds':time.perf_counter()-t}
        history.append(row)
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'epoch':epoch,'config':config,'val_total':va['total']}
        if va['total'] < best:
            best = va['total']; torch.save(state, a.out/'best.pt')
        torch.save(state, a.out/'last.pt')
        (a.out/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row),flush=True)
    checkpoint = torch.load(a.out/'best.pt', map_location=a.device, weights_only=False)
    model.load_state_dict(checkpoint['model']); model.eval()
    torch.manual_seed(a.seed + 2)
    with torch.no_grad():
        samples = model.decode(torch.randn(64,a.latent_dim,device=a.device)).sigmoid().cpu()
        x = val.images[:16].to(a.device)
        reconstruction = model(x, sample=False)[0].sigmoid().cpu()
    save_image(samples, a.out/'samples.png', nrow=8, padding=2, normalize=False)
    save_image(torch.cat([x.cpu(),reconstruction]),a.out/'reconstructions.png',nrow=len(x),padding=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(10,2.8))
    for ax,key,title in zip(axes,['total','reconstruction','kl'],['Negative ELBO','Reconstruction BCE','KL divergence']):
        for split in ['train','val']:
            ax.plot([r['epoch'] for r in history],[r[split][key] for r in history],label=split)
        ax.set(xlabel='Epoch',ylabel='Nats / image',title=title); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(a.out/'loss_curves.png',dpi=200); plt.close(fig)
    summary = {'best_epoch':checkpoint['epoch'],'best_val_total':checkpoint['val_total'],
               'elapsed_seconds':time.perf_counter()-start,'checkpoint_sha256':file_hash(a.out/'best.pt')}
    (a.out/'training_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
