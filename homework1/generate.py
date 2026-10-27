"""Generate unconditional images from a saved, trusted VAE checkpoint."""
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from model import VAE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=64)
    p.add_argument('--nrow',type=int,default=8)
    p.add_argument('--seed',type=int,default=287)
    a=p.parse_args()
    if a.count<1 or a.nrow<1: p.error('count and nrow must be positive')
    if a.output.exists(): p.error('Output already exists; choose a new filename')
    torch.set_num_threads(6)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=VAE(ckpt['config']['latent_dim']); model.load_state_dict(ckpt['model']); model.eval()
    generator=torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        samples=model.decode(torch.randn(a.count,model.latent_dim,generator=generator)).sigmoid()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    save_image(samples,a.output,nrow=a.nrow,padding=2,normalize=False)


if __name__=='__main__': main()
