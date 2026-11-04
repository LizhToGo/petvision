from __future__ import annotations

import unittest

import torch
from torch import nn

from petvision.models import build_classifier


class ResNetTrainingTest(unittest.TestCase):
    def test_frozen_backbone_keeps_batchnorm_state_until_unfrozen(self) -> None:
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = build_classifier(3, "resnet50")
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                images = torch.randn(2, 3, 64, 64)
                labels = torch.tensor([0, 2])
                criterion = nn.CrossEntropyLoss()
                initial_conv = model.backbone.conv1.weight.detach().clone()
                initial_mean = model.backbone.bn1.running_mean.clone()
                initial_head = model.classifier[-1].weight.detach().clone()

                model.set_backbone_trainable(False)
                model.train()
                criterion(model(images), labels).backward()
                optimizer.step()
                self.assertFalse(model.backbone.training)
                self.assertTrue(model.classifier.training)
                self.assertIsNone(model.backbone.conv1.weight.grad)
                self.assertTrue(torch.equal(initial_conv, model.backbone.conv1.weight))
                self.assertTrue(torch.equal(initial_mean, model.backbone.bn1.running_mean))
                self.assertFalse(torch.equal(initial_head, model.classifier[-1].weight))

                model.set_backbone_trainable(True)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                criterion(model(images), labels).backward()
                optimizer.step()
                self.assertTrue(model.backbone.training)
                self.assertIsNotNone(model.backbone.conv1.weight.grad)
                self.assertFalse(torch.equal(initial_conv, model.backbone.conv1.weight))
                self.assertFalse(torch.equal(initial_mean, model.backbone.bn1.running_mean))
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
