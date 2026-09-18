import unittest

import numpy as np

from obs_noise import (
    ObservationNoiseConfig,
    ObservationNoiseProcessor,
    append_observation_frame,
    split_policy_and_rnd_frame,
)


def make_processor(seed=123, **kwargs):
    config = ObservationNoiseConfig(**kwargs)
    processor = ObservationNoiseProcessor(config)
    processor.seed(seed)
    return processor


class ObservationNoiseTests(unittest.TestCase):
    def test_disabled_noise_returns_clean_observation(self):
        obs = np.arange(16, dtype=np.float32).reshape(4, 4)
        processor = make_processor(enabled=False, std_fraction=0.50)

        noisy = processor.apply(obs)

        np.testing.assert_array_equal(noisy, obs)

    def test_zero_standard_deviation_returns_clean_observation(self):
        obs = np.arange(16, dtype=np.float32).reshape(4, 4)
        processor = make_processor(enabled=True, std_fraction=0.0)

        noisy = processor.apply(obs)

        np.testing.assert_array_equal(noisy, obs)

    def test_shape_is_preserved_for_batched_observations(self):
        obs = np.full((64, 84, 84), 127.5, dtype=np.float32)
        processor = make_processor(enabled=True, std_fraction=0.05)

        noisy = processor.apply(obs, batched=True)

        self.assertEqual(noisy.shape, obs.shape)

    def test_range_is_preserved(self):
        obs = np.full((64, 84, 84), 250.0, dtype=np.float32)
        processor = make_processor(enabled=True, std_fraction=1.0)

        noisy = processor.apply(obs, batched=True)

        self.assertGreaterEqual(float(noisy.min()), 0.0)
        self.assertLessEqual(float(noisy.max()), 255.0)

    def test_seed_reproducibility(self):
        sequence = [
            np.full((8, 8), 120.0 + idx, dtype=np.float32)
            for idx in range(4)
        ]
        first = make_processor(seed=7, enabled=True, std_fraction=0.05)
        second = make_processor(seed=7, enabled=True, std_fraction=0.05)
        different = make_processor(seed=8, enabled=True, std_fraction=0.05)

        first_outputs = [first.apply(obs) for obs in sequence]
        second_outputs = [second.apply(obs) for obs in sequence]
        different_outputs = [different.apply(obs) for obs in sequence]

        for first_output, second_output in zip(first_outputs, second_outputs):
            np.testing.assert_array_equal(first_output, second_output)

        self.assertTrue(any(
            not np.array_equal(first_output, different_output)
            for first_output, different_output in zip(first_outputs, different_outputs)
        ))

    def test_gaussian_statistics_before_clipping(self):
        obs = np.full((200000,), 127.5, dtype=np.float32)
        std_fraction = 0.05
        processor = make_processor(enabled=True, std_fraction=std_fraction)

        _, raw_noise = processor.apply(obs, return_noise=True)

        expected_std = std_fraction * 255.0
        self.assertLess(abs(float(raw_noise.mean())), 0.15)
        self.assertLess(abs(float(raw_noise.std()) - expected_std), 0.15)

    def test_both_target_uses_corrupted_observation_for_policy_and_rnd(self):
        obs = np.full((8, 8), 127.5, dtype=np.float32)
        processor = make_processor(
            enabled=True,
            std_fraction=0.05,
            target='both')

        policy_frame, rnd_frame = split_policy_and_rnd_frame(obs, processor)

        self.assertIs(policy_frame, rnd_frame)
        self.assertFalse(np.array_equal(policy_frame, obs))

    def test_rnd_only_keeps_policy_clean_and_rnd_noisy(self):
        obs = np.full((8, 8), 127.5, dtype=np.float32)
        processor = make_processor(
            enabled=True,
            std_fraction=0.05,
            target='rnd_only')

        policy_frame, rnd_frame = split_policy_and_rnd_frame(obs, processor)
        predictor_input = rnd_frame
        target_input = rnd_frame

        np.testing.assert_array_equal(policy_frame, obs)
        self.assertFalse(np.array_equal(rnd_frame, obs))
        self.assertIs(predictor_input, target_input)

    def test_frame_stack_does_not_recorrupt_old_frames(self):
        policy_history = np.zeros((4, 2, 2), dtype=np.float32)
        rnd_history = np.zeros_like(policy_history)
        processor = make_processor(
            enabled=True,
            std_fraction=0.05,
            target='both')

        append_observation_frame(
            np.full((2, 2), 100.0, dtype=np.float32),
            policy_history,
            rnd_history,
            processor)
        stored_policy_frame = policy_history[-1].copy()
        stored_rnd_frame = rnd_history[-1].copy()

        append_observation_frame(
            np.full((2, 2), 120.0, dtype=np.float32),
            policy_history,
            rnd_history,
            processor)

        np.testing.assert_array_equal(policy_history[-2], stored_policy_frame)
        np.testing.assert_array_equal(rnd_history[-2], stored_rnd_frame)


if __name__ == '__main__':
    unittest.main()

