from dataclasses import dataclass

import numpy as np


OBS_NOISE_TARGET_BOTH = 'both'
OBS_NOISE_TARGET_RND_ONLY = 'rnd_only'
OBS_NOISE_TYPE_GAUSSIAN = 'gaussian'
OBS_NOISE_SEED_OFFSET = 0x9E3779B9
MAX_UINT32 = 2 ** 32


@dataclass(frozen=True)
class ObservationNoiseConfig:
    """Configuration for the thesis observation-noise condition.

    Gaussian noise is parameterized as a fraction of the valid pixel intensity
    range. ``target`` controls whether the corrupted frame is used by both the
    policy and RND streams or only by the RND predictor/target comparison.
    """

    enabled: bool = False
    noise_type: str = OBS_NOISE_TYPE_GAUSSIAN
    std_fraction: float = 0.05
    probability: float = 1.0
    target: str = OBS_NOISE_TARGET_BOTH
    value_min: float = 0.0
    value_max: float = 255.0

    @classmethod
    def from_config(cls, config_section):
        """Build noise settings from the shared experiment configuration."""
        config = cls(
            enabled=config_section.getboolean('UseObsNoise', fallback=False),
            noise_type=config_section.get('ObsNoiseType', fallback=OBS_NOISE_TYPE_GAUSSIAN),
            std_fraction=config_section.getfloat('ObsNoiseStd', fallback=0.05),
            probability=config_section.getfloat('ObsNoiseProbability', fallback=1.0),
            target=config_section.get('ObsNoiseTarget', fallback=OBS_NOISE_TARGET_BOTH),
        )
        config.validate()
        return config

    def validate(self):
        noise_type = self.noise_type.lower()
        if noise_type != OBS_NOISE_TYPE_GAUSSIAN:
            raise ValueError(
                "ObsNoiseType must be '{}', got '{}'".format(
                    OBS_NOISE_TYPE_GAUSSIAN,
                    self.noise_type))

        target = self.target.lower()
        if target not in (OBS_NOISE_TARGET_BOTH, OBS_NOISE_TARGET_RND_ONLY):
            raise ValueError(
                "ObsNoiseTarget must be '{}' or '{}', got '{}'".format(
                    OBS_NOISE_TARGET_BOTH,
                    OBS_NOISE_TARGET_RND_ONLY,
                    self.target))

        if self.std_fraction < 0:
            raise ValueError('ObsNoiseStd must be >= 0, got {}'.format(self.std_fraction))

        if self.probability < 0 or self.probability > 1:
            raise ValueError('ObsNoiseProbability must be between 0 and 1, got {}'.format(self.probability))

        if self.value_max <= self.value_min:
            raise ValueError(
                'Observation noise value_max must be greater than value_min, got {} <= {}'.format(
                    self.value_max,
                    self.value_min))

    @property
    def normalized_noise_type(self):
        return self.noise_type.lower()

    @property
    def normalized_target(self):
        return self.target.lower()

    @property
    def intensity_range(self):
        return self.value_max - self.value_min

    @property
    def absolute_std(self):
        """Noise standard deviation in pixel-value units, not a normalized scale."""
        return self.std_fraction * self.intensity_range

    @property
    def requires_separate_rnd_observation(self):
        return self.enabled and self.normalized_target == OBS_NOISE_TARGET_RND_ONLY

    def log_lines(self):
        lines = ['Observation noise enabled: {}'.format(self.enabled)]
        if self.enabled:
            lines.extend([
                'Observation noise type: {}'.format(self.normalized_noise_type),
                'Observation noise std: {}'.format(self.std_fraction),
                'Observation noise probability: {}'.format(self.probability),
                'Observation noise target: {}'.format(self.normalized_target),
            ])
        return lines


def derive_observation_noise_seed(seed):
    """Derive a separate but reproducible RNG seed for observation noise."""
    return (int(seed) + OBS_NOISE_SEED_OFFSET) % MAX_UINT32


class ObservationNoiseProcessor:
    """Applies configured observation corruption without mutating inputs.

    The environment wrappers call this after environment-specific preprocessing
    has produced an 84x84 grayscale frame and before the frame is inserted into
    the four-frame history stack.
    """

    def __init__(self, config=None, rng=None):
        self.config = config if config is not None else ObservationNoiseConfig()
        self.config.validate()
        self.rng = rng if rng is not None else np.random.RandomState()

    def seed(self, seed):
        """Seed only the noise RNG, offset from the environment seed."""
        self.rng.seed(derive_observation_noise_seed(seed))

    def get_rng_state(self):
        return self.rng.get_state()

    def set_rng_state(self, rng_state):
        if rng_state is not None:
            self.rng.set_state(rng_state)

    def apply(self, observation, batched=False, return_noise=False):
        """Return a clean copy or a clipped float32 observation with Gaussian noise."""
        observation = np.asarray(observation)

        if (not self.config.enabled or
                self.config.std_fraction == 0 or
                self.config.probability == 0):
            clean = observation.copy()
            if return_noise:
                return clean, np.zeros_like(clean, dtype=np.float32)
            return clean

        if self.config.normalized_noise_type == OBS_NOISE_TYPE_GAUSSIAN:
            return self._apply_gaussian(observation, batched, return_noise)

        raise ValueError(
            "Unsupported ObsNoiseType '{}'".format(self.config.noise_type))

    def _apply_gaussian(self, observation, batched, return_noise):
        clean_float = observation.astype(np.float32, copy=True)
        # Noise is sampled in the same pixel-value scale as the preprocessed
        # frame, then clipped to the configured valid observation range.
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.absolute_std,
            size=clean_float.shape).astype(np.float32)

        if self.config.probability < 1.0:
            noise = noise * self._corruption_mask(clean_float.shape, batched)

        noisy = np.clip(
            clean_float + noise,
            self.config.value_min,
            self.config.value_max).astype(np.float32, copy=False)

        if return_noise:
            return noisy, noise
        return noisy

    def _corruption_mask(self, shape, batched):
        # Probability gates whole frames/samples, not individual pixels. This
        # keeps partial corruption controlled at the observation level.
        if batched:
            if len(shape) == 0:
                return np.float32(self.rng.rand() < self.config.probability)
            mask_shape = (shape[0],) + (1,) * (len(shape) - 1)
            return (self.rng.rand(shape[0]).reshape(mask_shape) < self.config.probability).astype(np.float32)
        return np.float32(self.rng.rand() < self.config.probability)


def split_policy_and_rnd_frame(clean_frame, noise_processor):
    """Create the policy and RND frames for a single preprocessed observation.

    ``both`` sends the same noisy frame to policy and RND. ``rnd_only`` keeps the
    policy frame clean while feeding the same noisy frame to both RND networks,
    isolating the noise perturbation to the intrinsic-reward pathway.
    """
    clean_frame = np.asarray(clean_frame)
    config = noise_processor.config

    if not config.enabled:
        return clean_frame.copy(), clean_frame.copy()

    noisy_frame = noise_processor.apply(clean_frame)

    if config.normalized_target == OBS_NOISE_TARGET_BOTH:
        return noisy_frame, noisy_frame
    if config.normalized_target == OBS_NOISE_TARGET_RND_ONLY:
        return clean_frame.copy(), noisy_frame

    raise ValueError(
        "ObsNoiseTarget must be '{}' or '{}', got '{}'".format(
            OBS_NOISE_TARGET_BOTH,
            OBS_NOISE_TARGET_RND_ONLY,
            config.target))


def fill_observation_histories(clean_frame, policy_history, rnd_history, noise_processor):
    """Initialize policy and RND frame stacks from one preprocessed frame."""
    policy_frame, rnd_frame = split_policy_and_rnd_frame(clean_frame, noise_processor)
    for idx in range(policy_history.shape[0]):
        policy_history[idx, :, :] = policy_frame
        rnd_history[idx, :, :] = rnd_frame


def append_observation_frame(clean_frame, policy_history, rnd_history, noise_processor):
    """Append one new preprocessed frame without re-corrupting older history."""
    policy_frame, rnd_frame = split_policy_and_rnd_frame(clean_frame, noise_processor)
    policy_history[:-1, :, :] = policy_history[1:, :, :]
    rnd_history[:-1, :, :] = rnd_history[1:, :, :]
    policy_history[-1, :, :] = policy_frame
    rnd_history[-1, :, :] = rnd_frame


def env_result_has_rnd_state(result):
    if len(result) <= 5:
        return False
    last = result[-1]
    first = result[0]
    return isinstance(last, np.ndarray) and isinstance(first, np.ndarray) and last.shape == first.shape


def split_env_result_rnd_state(result):
    """Return the policy result tuple and the RND observation stack.

    Older clean-observation runs return only one state stack; in that case the
    same stack is used for policy actions and RND reward computation.
    """
    result = list(result)
    if env_result_has_rnd_state(result):
        return result[:-1], result[-1]
    return result, result[0]

