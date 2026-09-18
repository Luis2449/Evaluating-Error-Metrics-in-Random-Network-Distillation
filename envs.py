import gym
import cv2
import os

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

try:
    import gymnasium
except ImportError:
    gymnasium = None

try:
    import minigrid
except ImportError:
    minigrid = None

try:
    import gym_minigrid
except ImportError:
    gym_minigrid = None

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from obs_noise import (
    ObservationNoiseConfig,
    ObservationNoiseProcessor,
    append_observation_frame,
    fill_observation_histories,
)
from PIL import Image

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])
episode_log_interval = default_config.getint('EpisodeLogInterval', fallback=10)


class VideoRecorder:
    def __init__(self, path=None, fps=60):
        self.path = path
        self.fps = fps
        self.writer = None

    def write(self, frame):
        if not self.path:
            return

        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Video frames must have shape (height, width, 3)')

        if self.writer is None:
            video_dir = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(video_dir, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError('Could not open video writer for {}'.format(self.path))

        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


def render_env(env):
    raw_env = unwrap(env)
    if hasattr(raw_env, "ale"):
        frame = raw_env.ale.getScreenRGB()
        window_name = getattr(raw_env, "_game", "Atari")
        cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return None

    try:
        return env.render()
    except TypeError as exc:
        if "mode" not in str(exc):
            raise
        return env.render(mode="human")


def enable_atari_human_render(env):
    raw_env = unwrap(env)
    if hasattr(raw_env, "ale"):
        if hasattr(raw_env, "_render_mode"):
            raw_env._render_mode = "human"
        raw_env.ale.setBool("display_screen", True)
        raw_env.ale.setBool("sound", True)


def make_atari_env(env_id, is_render):
    env = gym.make(env_id)
    if is_render:
        enable_atari_human_render(env)
    return env


def make_minigrid_env(env_id, is_render):
    """Create MiniGrid with Gymnasium when available, otherwise legacy Gym."""
    if gymnasium is not None and minigrid is not None:
        render_mode = "human" if is_render else "rgb_array"
        return gymnasium.make(env_id, render_mode=render_mode)

    if gym_minigrid is not None:
        return gym.make(env_id)

    raise ImportError(
        "MiniGrid environments require the 'minigrid' package. "
        "Install it with `pip install minigrid`."
    )


def make_configured_env(env_type, env_id, is_render=False):
    if env_type == 'mario':
        return JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    if env_type == 'atari':
        return make_atari_env(env_id, is_render)
    if env_type == 'minigrid':
        return make_minigrid_env(env_id, is_render)
    raise NotImplementedError(
        "Unsupported EnvType '{}'. Expected one of: atari, mario, minigrid.".format(env_type)
    )


def get_environment_class(env_type):
    if env_type == 'atari':
        return AtariEnvironment
    if env_type == 'mario':
        return MarioEnvironment
    if env_type == 'minigrid':
        return MiniGridEnvironment
    raise NotImplementedError(
        "Unsupported EnvType '{}'. Expected one of: atari, mario, minigrid.".format(env_type)
    )


def get_observation_shape(env):
    """Return an image-like observation shape for Atari, Mario, and MiniGrid."""
    shape = getattr(env.observation_space, 'shape', None)
    if shape is not None:
        return shape

    spaces = getattr(env.observation_space, 'spaces', {})
    image_space = spaces.get('image') if spaces else None
    shape = getattr(image_space, 'shape', None)
    if shape is not None:
        return shape

    return (84, 84, 3)


def unpack_reset_result(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0]
    return result


def unpack_step_result(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    return result


def seed_env(env, seed):
    """Seed the environment and spaces when the installed Gym API supports it."""
    if hasattr(env, 'seed'):
        try:
            env.seed(seed)
        except TypeError:
            pass

    for space_name in ('action_space', 'observation_space'):
        space = getattr(env, space_name, None)
        if hasattr(space, 'seed'):
            space.seed(seed)


def reset_env(env, seed=None):
    if seed is not None:
        try:
            return unpack_reset_result(env.reset(seed=seed))
        except TypeError:
            pass
    return unpack_reset_result(env.reset())


def build_observation_noise_processor(obs_noise_config=None):
    """Construct the per-worker noise processor from the shared config."""
    config = obs_noise_config if obs_noise_config is not None else ObservationNoiseConfig.from_config(default_config)
    return ObservationNoiseProcessor(config)


def get_environment_rng_state(obs_noise_processor):
    """Capture worker RNG state needed to resume training reproducibly."""
    if not obs_noise_processor.config.enabled:
        return np.random.get_state()

    return {
        'numpy': np.random.get_state(),
        'obs_noise': obs_noise_processor.get_rng_state(),
    }


def set_environment_rng_state(rng_state, obs_noise_processor):
    """Restore worker RNG state, including observation-noise RNG when present."""
    if isinstance(rng_state, dict) and 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])
        obs_noise_processor.set_rng_state(rng_state.get('obs_noise'))
        return

    # Backward compatibility for checkpoints that only stored NumPy state.
    np.random.set_state(rng_state)


def append_rnd_state_if_needed(result, rnd_history, obs_noise_processor):
    """Attach a separate RND frame stack only for ``ObsNoiseTarget=rnd_only``.

    Clean and ``both`` noise runs use the policy stack as the RND stack. The
    extra return value lets training keep policy observations clean while still
    computing RND rewards from corrupted observations.
    """
    if obs_noise_processor.config.requires_separate_rnd_observation:
        result = list(result)
        result.append(rnd_history[:, :, :])
    return result


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4, video_path=None, video_fps=60):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render
        self.video_recorder = VideoRecorder(video_path, video_fps)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = unpack_step_result(self.env.step(action))
            self.video_recorder.write(obs)
            if self.is_render:
                render_env(self.env)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return unpack_reset_result(self.env.reset(**kwargs))

    def close(self):
        self.video_recorder.close()
        return self.env.close()


class MontezumaInfoWrapper(gym.Wrapper):
    """Adds room-coverage metadata for Montezuma's Revenge evaluation.

    The room id is read directly from ALE RAM address 3, matching the common
    Montezuma room-counting convention used for exploration benchmarks.
    """

    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25,
            video_path=None,
            video_fps=60,
            obs_noise_config=None):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = None
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.video_path = video_path
        self.video_fps = video_fps

        self.history_size = history_size
        # ``history`` feeds the policy/value network. ``rnd_history`` is usually
        # identical, but can carry the noisy stream for ``ObsNoiseTarget=rnd_only``.
        self.history = np.zeros([history_size, h, w])
        self.rnd_history = np.zeros_like(self.history)
        self.obs_noise = build_observation_noise_processor(obs_noise_config)
        self.h = h
        self.w = w

    def _ensure_env(self):
        if self.env is not None:
            return

        self.env = MaxAndSkipEnv(
            make_atari_env(self.env_id, self.is_render),
            self.is_render,
            video_path=self.video_path,
            video_fps=self.video_fps)
        if 'Montezuma' in self.env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3)

    def run(self):
        super(AtariEnvironment, self).run()
        try:
            self._ensure_env()
            self.reset()
            while True:
                try:
                    action = self.child_conn.recv()
                except (EOFError, BrokenPipeError, OSError):
                    break

                if isinstance(action, dict):
                    if action.get('cmd') == 'get_rng_state':
                        self.child_conn.send(get_environment_rng_state(self.obs_noise))
                    elif action.get('cmd') == 'set_rng_state':
                        set_environment_rng_state(action['state'], self.obs_noise)
                        self.child_conn.send(True)
                    elif action.get('cmd') == 'set_seed':
                        self.set_seed(action['seed'])
                        self.child_conn.send(True)
                    elif action.get('cmd') == 'close':
                        self.child_conn.send(True)
                        break
                    continue

                if 'Breakout' in self.env_id:
                    action += 1

                # sticky action
                if self.sticky_action:
                    if np.random.rand() <= self.p:
                        action = self.last_action
                    self.last_action = action

                s, reward, done, info = self.env.step(action)

                log_reward = reward
                force_done = done

                # Noise is applied after Atari preprocessing and before the
                # four-frame stacks are shifted.
                append_observation_frame(self.pre_proc(s), self.history, self.rnd_history, self.obs_noise)

                self.rall += reward
                self.steps += 1

                if self.steps >= max_step_per_episode:
                    done = True
                    force_done = True

                if done:
                    self.recent_rlist.append(self.rall)
                    if episode_log_interval > 0 and self.episode % episode_log_interval == 0:
                        print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                            self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                            info.get('episode', {}).get('visited_rooms', {})))

                    self.history = self.reset()

                try:
                    visited_rooms = sorted(info.get('episode', {}).get('visited_rooms', []))
                    self.child_conn.send(
                        append_rnd_state_if_needed(
                            [self.history[:, :, :], reward, force_done, done, log_reward, visited_rooms],
                            self.rnd_history,
                            self.obs_noise))
                except (BrokenPipeError, OSError):
                    break
        finally:
            if self.env is not None:
                self.env.close()
            if self.is_render:
                cv2.destroyAllWindows()
            self.child_conn.close()

    def set_seed(self, seed):
        seed = int(seed)
        np.random.seed(seed)
        self.obs_noise.seed(seed)
        seed_env(self.env, seed)
        self.last_action = 0
        self.steps = 0
        self.rall = 0
        s = reset_env(self.env, seed)
        self.get_init_state(
            self.pre_proc(s))

    def reset(self):
        self._ensure_env()
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        fill_observation_histories(self.pre_proc(s), self.history, self.rnd_history, self.obs_noise)


class MarioEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            life_done=False,
            h=84,
            w=84, movement=COMPLEX_MOVEMENT, sticky_action=True,
            p=0.25,
            video_path=None,
            video_fps=60,
            obs_noise_config=None):
        super(MarioEnvironment, self).__init__()
        self.daemon = True
        self.env = None

        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.life_done = life_done
        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.video_recorder = VideoRecorder(video_path, video_fps)
        self.movement = movement

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.rnd_history = np.zeros_like(self.history)
        self.obs_noise = build_observation_noise_processor(obs_noise_config)
        self.h = h
        self.w = w

    def _ensure_env(self):
        if self.env is not None:
            return

        self.env = JoypadSpace(
            gym_super_mario_bros.make(self.env_id), self.movement)

    def run(self):
        super(MarioEnvironment, self).run()
        try:
            self._ensure_env()
            self.reset()
            while True:
                try:
                    action = self.child_conn.recv()
                except (EOFError, BrokenPipeError, OSError):
                    break

                if isinstance(action, dict):
                    if action.get('cmd') == 'get_rng_state':
                        self.child_conn.send(get_environment_rng_state(self.obs_noise))
                    elif action.get('cmd') == 'set_rng_state':
                        set_environment_rng_state(action['state'], self.obs_noise)
                        self.child_conn.send(True)
                    elif action.get('cmd') == 'set_seed':
                        self.set_seed(action['seed'])
                        self.child_conn.send(True)
                    elif action.get('cmd') == 'close':
                        self.child_conn.send(True)
                        break
                    continue

                if self.is_render:
                    render_env(self.env)

                # sticky action
                if self.sticky_action:
                    if np.random.rand() <= self.p:
                        action = self.last_action
                    self.last_action = action

                # 4 frame skip
                reward = 0.0
                done = None
                for i in range(4):
                    obs, r, done, info = unpack_step_result(self.env.step(action))
                    self.video_recorder.write(obs)
                    if self.is_render:
                        render_env(self.env)
                    reward += r
                    if done:
                        break

                # when Mario loses life, changes the state to the terminal
                # state.
                if self.life_done:
                    if self.lives > info['life'] and info['life'] > 0:
                        force_done = True
                        self.lives = info['life']
                    else:
                        force_done = done
                        self.lives = info['life']
                else:
                    force_done = done

                # reward range -15 ~ 15
                log_reward = reward / 15
                self.rall += log_reward

                r = int(info.get('flag_get', False))

                append_observation_frame(self.pre_proc(obs), self.history, self.rnd_history, self.obs_noise)

                self.steps += 1

                if done:
                    self.recent_rlist.append(self.rall)
                    if episode_log_interval > 0 and self.episode % episode_log_interval == 0:
                        print(
                            "[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Stage: {} current x:{}   max x:{}".format(
                                self.episode,
                                self.env_idx,
                                self.steps,
                                self.rall,
                                np.mean(
                                    self.recent_rlist),
                                info['stage'],
                                info['x_pos'],
                                self.max_pos))

                    self.history = self.reset()

                try:
                    self.child_conn.send(
                        append_rnd_state_if_needed(
                            [self.history[:, :, :], r, force_done, done, log_reward, []],
                            self.rnd_history,
                            self.obs_noise))
                except (BrokenPipeError, OSError):
                    break
        finally:
            self.video_recorder.close()
            if self.env is not None:
                self.env.close()
            if self.is_render:
                cv2.destroyAllWindows()
            self.child_conn.close()

    def set_seed(self, seed):
        seed = int(seed)
        np.random.seed(seed)
        self.obs_noise.seed(seed)
        seed_env(self.env, seed)
        self.last_action = 0
        self.steps = 0
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        self.get_init_state(reset_env(self.env, seed))

    def reset(self):
        self._ensure_env()
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        self.get_init_state(reset_env(self.env))
        return self.history[:, :, :]

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))

        return x

    def get_init_state(self, s):
        fill_observation_histories(self.pre_proc(s), self.history, self.rnd_history, self.obs_noise)


class MiniGridEnvironment(Environment):
    """DoorKey-compatible worker with MiniGrid-specific evaluation metadata."""

    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            life_done=False,
            h=84,
            w=84,
            sticky_action=True,
            p=0.25,
            video_path=None,
            video_fps=60,
            obs_noise_config=None):
        super(MiniGridEnvironment, self).__init__()
        self.daemon = True
        self.env = None

        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p
        self.video_recorder = VideoRecorder(video_path, video_fps)

        self.history_size = history_size
        # The noisy DoorKey experiment can keep the policy stack clean while the
        # RND stack receives Gaussian-corrupted frames.
        self.history = np.zeros([history_size, h, w])
        self.rnd_history = np.zeros_like(self.history)
        self.obs_noise = build_observation_noise_processor(obs_noise_config)
        self.h = h
        self.w = w
        self.visited_cells = set()
        self.picked_up_key = False
        self.opened_door = False
        self.reached_goal = False

    def _ensure_env(self):
        if self.env is not None:
            return

        self.env = make_minigrid_env(self.env_id, self.is_render)

    def _agent_pos(self):
        pos = getattr(unwrap(self.env), 'agent_pos', None)
        if pos is None:
            return None
        return [int(pos[0]), int(pos[1])]

    def _carrying_type(self):
        carrying = getattr(unwrap(self.env), 'carrying', None)
        if carrying is None:
            return None
        return getattr(carrying, 'type', str(carrying))

    def _grid_objects(self):
        """Collect object positions used to summarize DoorKey progress."""
        grid = getattr(unwrap(self.env), 'grid', None)
        width = getattr(grid, 'width', 0)
        height = getattr(grid, 'height', 0)
        objects = {
            'doors': [],
            'keys': [],
            'goals': [],
        }

        if grid is None:
            return objects

        for x in range(width):
            for y in range(height):
                obj = grid.get(x, y)
                obj_type = getattr(obj, 'type', None)
                if obj_type == 'door':
                    objects['doors'].append({
                        'position': [int(x), int(y)],
                        'is_open': bool(getattr(obj, 'is_open', False)),
                        'is_locked': bool(getattr(obj, 'is_locked', False)),
                    })
                elif obj_type == 'key':
                    objects['keys'].append([int(x), int(y)])
                elif obj_type == 'goal':
                    objects['goals'].append([int(x), int(y)])

        return objects

    def _reset_minigrid_tracking(self):
        """Start a fresh per-episode DoorKey exploration summary."""
        self.visited_cells = set()
        self.picked_up_key = False
        self.opened_door = False
        self.reached_goal = False
        pos = self._agent_pos()
        if pos is not None:
            self.visited_cells.add(tuple(pos))

    def _minigrid_summary(self, reward=0.0, done=False, info=None):
        """Return DoorKey metrics accumulated for the current episode.

        The summary tracks visited grid cells and thesis-relevant milestones
        such as key pickup, door opening, and reaching the goal.
        """
        info = info if isinstance(info, dict) else {}
        pos = self._agent_pos()
        if pos is not None:
            self.visited_cells.add(tuple(pos))

        carrying = self._carrying_type()
        if carrying == 'key':
            self.picked_up_key = True

        objects = self._grid_objects()
        if any(door['is_open'] for door in objects['doors']):
            self.opened_door = True

        if bool(info.get('success', False)) or (bool(done) and float(reward) > 0.0):
            self.reached_goal = True

        visited_cells = sorted([list(cell) for cell in self.visited_cells])
        return {
            'success': bool(self.reached_goal),
            'goal_reached': bool(self.reached_goal),
            'picked_up_key': bool(self.picked_up_key),
            'door_opened': bool(self.opened_door),
            'visited_cells_count': int(len(visited_cells)),
            'visited_cells': visited_cells,
            'final_agent_pos': pos,
            'final_agent_dir': int(getattr(unwrap(self.env), 'agent_dir', -1)),
            'carrying': carrying,
            'doors': objects['doors'],
            'keys': objects['keys'],
            'goals': objects['goals'],
        }

    def run(self):
        super(MiniGridEnvironment, self).run()
        try:
            self._ensure_env()
            self.reset()
            while True:
                try:
                    action = self.child_conn.recv()
                except (EOFError, BrokenPipeError, OSError):
                    break

                if isinstance(action, dict):
                    if action.get('cmd') == 'get_rng_state':
                        self.child_conn.send(get_environment_rng_state(self.obs_noise))
                    elif action.get('cmd') == 'set_rng_state':
                        set_environment_rng_state(action['state'], self.obs_noise)
                        self.child_conn.send(True)
                    elif action.get('cmd') == 'set_seed':
                        self.set_seed(action['seed'])
                        self.child_conn.send(True)
                    elif action.get('cmd') == 'close':
                        self.child_conn.send(True)
                        break
                    continue

                if self.sticky_action:
                    if np.random.rand() <= self.p:
                        action = self.last_action
                    self.last_action = action

                obs, reward, done, info = unpack_step_result(self.env.step(int(action)))
                frame = self.frame_from_obs(obs)
                self.video_recorder.write(frame)
                if self.is_render:
                    render_env(self.env)

                log_reward = reward
                force_done = done

                # For noisy DoorKey runs, corruption happens after rendering the
                # MiniGrid frame and resizing/grayscaling it in ``pre_proc``.
                append_observation_frame(self.pre_proc(frame), self.history, self.rnd_history, self.obs_noise)

                self.rall += reward
                self.steps += 1

                if self.steps >= max_step_per_episode:
                    done = True
                    force_done = True

                minigrid_summary = self._minigrid_summary(reward, done, info)

                if done:
                    self.recent_rlist.append(self.rall)
                    if episode_log_interval > 0 and self.episode % episode_log_interval == 0:
                        print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                            self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))

                    self.history = self.reset()

                try:
                    self.child_conn.send(
                        append_rnd_state_if_needed(
                            [self.history[:, :, :], reward, force_done, done, log_reward, [], minigrid_summary],
                            self.rnd_history,
                            self.obs_noise))
                except (BrokenPipeError, OSError):
                    break
        finally:
            self.video_recorder.close()
            if self.env is not None:
                self.env.close()
            if self.is_render:
                cv2.destroyAllWindows()
            self.child_conn.close()

    def set_seed(self, seed):
        seed = int(seed)
        np.random.seed(seed)
        self.obs_noise.seed(seed)
        seed_env(self.env, seed)
        self.last_action = 0
        self.steps = 0
        self.rall = 0
        self.get_init_state(reset_env(self.env, seed))
        self._reset_minigrid_tracking()

    def reset(self):
        self._ensure_env()
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.get_init_state(reset_env(self.env))
        self._reset_minigrid_tracking()
        return self.history[:, :, :]

    def frame_from_obs(self, obs):
        """Convert MiniGrid observations or renders into RGB frames."""
        frame = self.render_frame()
        if frame is not None:
            return frame

        if isinstance(obs, dict):
            obs = obs.get('image')

        if obs is None:
            raise ValueError('MiniGrid observation does not contain an image')

        frame = np.asarray(obs)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        else:
            raise ValueError('Unsupported MiniGrid observation shape {}'.format(frame.shape))

        frame = frame.astype(np.float32)
        max_value = np.max(frame) if frame.size else 0
        if max_value <= 20:
            frame *= 255.0 / max(1.0, max_value)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def render_frame(self):
        try:
            if getattr(self.env, 'render_mode', None):
                frame = self.env.render()
            else:
                frame = self.env.render(mode='rgb_array')
        except TypeError:
            frame = self.env.render()
        except Exception:
            return None

        if frame is None:
            return None

        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        else:
            return None
        return np.clip(frame, 0, 255).astype(np.uint8)

    def pre_proc(self, x):
        """Match the Atari-style grayscale 84x84 input expected by the models."""
        frame = self.frame_from_obs(x)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray.astype('float32'), (self.h, self.w))

    def get_init_state(self, s):
        initial_state = self.pre_proc(s)
        fill_observation_histories(initial_state, self.history, self.rnd_history, self.obs_noise)
