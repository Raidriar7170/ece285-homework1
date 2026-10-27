import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from model import VAE, loss_terms
from data import ChestDataset


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)

    def test_forward_loss_and_gradient(self):
        model = VAE(8)
        x = torch.rand(3,1,64,64)
        logits,mu,logvar = model(x)
        self.assertEqual(tuple(logits.shape),tuple(x.shape))
        self.assertEqual(tuple(mu.shape),(3,8))
        loss,bce,kl = loss_terms(logits,x,mu,logvar)
        self.assertTrue(torch.allclose(loss,bce+kl))
        self.assertGreaterEqual(kl.item(),0)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.mu.weight.grad.abs().sum().item(),0)

    def test_analytic_loss(self):
        x = torch.zeros(2,1,64,64); z = torch.zeros(2,8)
        total,bce,kl = loss_terms(x,x,z,z)
        self.assertAlmostEqual(bce.item(),4096*__import__('math').log(2),places=2)
        self.assertEqual(kl.item(),0)

    def test_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'train'/'NORMAL'; p.mkdir(parents=True)
            Image.new('RGB',(90,120),(255,255,255)).save(p/'a.png')
            ds=ChestDataset(d,'train')
            self.assertEqual(tuple(ds[0].shape),(1,64,64))
            self.assertEqual(ds[0].min().item(),1)
            self.assertEqual(ds.manifest()[0]['path'],'train/NORMAL/a.png')

    def test_ignite_metric_oracles(self):
        from ignite.metrics import FID,InceptionScore
        f=FID(num_features=4,feature_extractor=torch.nn.Identity())
        x=torch.randn(20,4); f.update((x,x))
        self.assertAlmostEqual(f.compute(),0,places=6)
        s=InceptionScore(num_features=4,feature_extractor=torch.nn.Identity())
        s.update(torch.ones(20,4)/4)
        self.assertAlmostEqual(s.compute(),1,places=6)


if __name__=='__main__':
    unittest.main()
