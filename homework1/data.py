"""Original Kaggle split, deterministic grayscale preprocessing, file provenance."""
import hashlib
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ChestDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.files = sorted(p for p in (self.root / split).rglob('*')
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and not p.name.startswith('._'))
        if not self.files:
            raise ValueError(f'No images found in {self.root / split}')
        # Cache only 64 x 64 tensors; avoids repeatedly decoding large source JPEGs.
        images = []
        for p in self.files:
            with Image.open(p) as im:
                im = im.convert('L').resize((64, 64), Image.Resampling.BILINEAR)
                images.append(TF.to_tensor(im))
        self.images = torch.stack(images)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.images[i]

    def manifest(self):
        return [{'path': str(p.relative_to(self.root)), 'sha256': file_hash(p)} for p in self.files]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
