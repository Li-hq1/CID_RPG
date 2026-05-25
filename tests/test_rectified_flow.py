import unittest

import torch

from genrec.models.diffusion.diffloss import DiffLoss
from genrec.models.diffusion.rectified_flow import RectifiedFlow


class ConstantVelocity(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.timesteps = []

    def forward(self, x, t, **kwargs):
        self.timesteps.append(t.detach().clone())
        return torch.full_like(x, self.value)


class RectifiedFlowTest(unittest.TestCase):
    def test_sample_ode_integrates_from_noise_to_data_direction(self):
        flow = RectifiedFlow(num_timesteps=1000)
        model = ConstantVelocity(value=2.0)
        noise = torch.tensor([[3.0]])

        sample = flow.sample_ode(
            model=model,
            shape=noise.shape,
            solver="euler",
            num_steps=2,
            noise=noise,
        )

        self.assertTrue(torch.allclose(sample, torch.tensor([[1.0]])))
        self.assertEqual(model.timesteps[0].item(), 999)

    def test_sample_ode_uses_training_timestep_range_with_few_solver_steps(self):
        flow = RectifiedFlow(num_timesteps=1000)
        model = ConstantVelocity(value=0.0)

        flow.sample_ode(
            model=model,
            shape=(1, 1),
            solver="euler",
            num_steps=5,
            noise=torch.zeros(1, 1),
        )

        timesteps = torch.cat(model.timesteps)
        self.assertEqual(timesteps.max().item(), 999)
        self.assertGreaterEqual(timesteps.min().item(), 0)
        self.assertGreater(timesteps[-1].item(), 100)


class DiffLossRectifiedFlowTest(unittest.TestCase):
    def test_rectified_flow_generation_keeps_training_time_range(self):
        loss = DiffLoss(
            target_channels=2,
            z_channels=3,
            depth=1,
            width=8,
            num_sampling_steps=5,
            use_rectified_flow=True,
            rectified_flow_steps=1000,
        )

        self.assertEqual(loss.train_diffusion.num_timesteps, 1000)
        self.assertEqual(loss.gen_diffusion.num_timesteps, 1000)
        self.assertEqual(loss.num_sampling_steps, 5)

    def test_rectified_flow_sample_reuses_cfg_noise_pair(self):
        loss = DiffLoss(
            target_channels=2,
            z_channels=3,
            depth=1,
            width=8,
            num_sampling_steps=5,
            use_rectified_flow=True,
            rectified_flow_steps=1000,
        )
        base_noise = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        captured = {}

        def fake_sample_noise(z, batch_size):
            self.assertEqual(batch_size, 2)
            return base_noise.to(device=z.device, dtype=z.dtype)

        def fake_sample_ode(model, shape, solver, model_kwargs, num_steps, device, noise):
            captured["noise"] = noise.detach().clone()
            captured["shape"] = shape
            captured["num_steps"] = num_steps
            return noise

        loss._sample_noise = fake_sample_noise
        loss.gen_diffusion.sample_ode = fake_sample_ode

        z = torch.zeros(4, 3)
        sample = loss.sample(z, temperature=0.5, cfg=3.0)

        expected = torch.cat([base_noise, base_noise], dim=0) * 0.5
        self.assertTrue(torch.equal(captured["noise"], expected))
        self.assertEqual(captured["shape"], expected.shape)
        self.assertEqual(captured["num_steps"], 5)
        self.assertTrue(torch.equal(sample, expected))


if __name__ == "__main__":
    unittest.main()
