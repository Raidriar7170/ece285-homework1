"""CPU FID/IS evaluation with Ignite accumulation and ImageNet Inception-v3."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from PIL import Image
from ignite.metrics import FID, InceptionScore
import ignite
from data import ChestDataset, file_hash
from model import VAE


def preprocess(batch):
    # Identical uint8 PIL resizing for real and generated images, as in the
    # assignment's Ignite tutorial. Explicit ImageNet normalization is used.
    images = [TF.to_tensor(TF.to_pil_image(x.cpu()).convert('RGB').resize(
        (299,299),Image.Resampling.BILINEAR)) for x in batch]
    return transforms.Normalize((.485,.456,.406),(.229,.224,.225))(torch.stack(images))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=1285)
    p.add_argument('--threads',type=int,default=6)
    a = p.parse_args()
    if (a.run/'metrics.json').exists():
        p.error('metrics.json already exists; preserve the existing evaluation')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    start = time.perf_counter()
    ckpt_path = a.run/'best.pt'
    checkpoint_hash = file_hash(ckpt_path)
    ckpt = torch.load(ckpt_path,map_location='cpu',weights_only=False)
    model = VAE(ckpt['config']['latent_dim']); model.load_state_dict(ckpt['model']); model.eval()
    test = ChestDataset(a.data,'test')
    manifest = test.manifest()  # Freshly hash frozen evaluation inputs.
    previous = json.loads((a.run/'data_manifest.json').read_text())
    development = {r['sha256'] for s in ['train','val'] for r in previous[s]}
    duplicates = [r['path'] for r in manifest if r['sha256'] in development]
    if duplicates:
        raise RuntimeError(f'Exact train/validation-test leakage: {duplicates}')
    (a.run/'test_manifest.json').write_text(json.dumps(manifest,indent=2))
    net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).eval()
    classifier = net.fc
    net.fc = nn.Identity()
    fid = FID(num_features=2048,feature_extractor=nn.Identity())
    score = InceptionScore(num_features=1000,feature_extractor=nn.Identity())
    fid.reset(); score.reset()
    generator = torch.Generator().manual_seed(a.seed)
    count = 0
    with torch.no_grad():
        for real in DataLoader(test,batch_size=a.batch_size,shuffle=False):
            z = torch.randn(len(real),model.latent_dim,generator=generator)
            fake = model.decode(z).sigmoid()
            real_features = net(preprocess(real))
            fake_features = net(preprocess(fake))
            fid.update((fake_features,real_features))
            score.update(classifier(fake_features).softmax(dim=1))
            count += len(real)
            print(f'Evaluated {count}/{len(test)} real and generated images',flush=True)
    result = {'fid_2048':float(fid.compute()),'inception_score':float(score.compute()),
              'real_count':count,'generated_count':count,'seed':a.seed,
              'checkpoint_epoch':ckpt['epoch'],'checkpoint_sha256':checkpoint_hash,
              'feature_extractor':'torchvision ImageNet Inception_V3_Weights.IMAGENET1K_V1, pool3 2048',
              'preprocessing':'grayscale 64x64 [0,1] -> uint8 RGB repeat -> PIL bilinear 299x299 -> ImageNet normalization',
              'is_protocol':'single global score over all generated samples; no split standard deviation',
              'fid_protocol':'Ignite FID with explicit 2048-dimensional pooled features, not default 1000 probabilities; not Clean-FID',
              'inception_weights_sha256':file_hash(Path(torch.hub.get_dir())/'checkpoints'/'inception_v3_google-0cc3c7bd.pth'),
              'ignite_version':ignite.__version__,'device':'cpu','elapsed_seconds':time.perf_counter()-start,
              'source_sha256':file_hash(Path(__file__)),
              'limitations':'ImageNet features are not clinical validation; small-sample FID is biased; supplied split does not establish patient independence'}
    if not all(math.isfinite(result[k]) for k in ['fid_2048','inception_score']):
        raise RuntimeError('Non-finite evaluation metric')
    if file_hash(ckpt_path) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation')
    (a.run/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
