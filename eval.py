from agents import *
from envs import *
from utils import *
from config import *
from obs_noise import ObservationNoiseConfig, split_env_result_rnd_state
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter

import numpy as np
import pickle
import os
import json


def unpack_env_result(result):
    """Normalize worker responses from clean and noisy observation modes."""
    result, rnd_state = split_env_result_rnd_state(result)
    if len(result) == 5:
        s, r, d, rd, lr = result
        return s, r, d, rd, lr, [], {}, rnd_state
    if len(result) == 6:
        s, r, d, rd, lr, visited_rooms = result
        return s, r, d, rd, lr, visited_rooms, {}, rnd_state
    s, r, d, rd, lr, visited_rooms, metadata = result
    return s, r, d, rd, lr, visited_rooms, metadata, rnd_state


def merge_minigrid_summary(summary, metadata):
    """Accumulate DoorKey evaluation metadata across worker steps."""
    if not metadata:
        return summary

    if summary is None:
        summary = {
            'success': False,
            'goal_reached': False,
            'picked_up_key': False,
            'door_opened': False,
            'visited_cells': set(),
            'final_agent_pos': None,
            'final_agent_dir': None,
            'carrying': None,
            'doors': [],
            'keys': [],
            'goals': [],
        }

    for key in ('success', 'goal_reached', 'picked_up_key', 'door_opened'):
        summary[key] = bool(summary[key] or metadata.get(key, False))

    for cell in metadata.get('visited_cells', []):
        if cell is not None and len(cell) == 2:
            summary['visited_cells'].add((int(cell[0]), int(cell[1])))

    for key in ('final_agent_pos', 'final_agent_dir', 'carrying', 'doors', 'keys', 'goals'):
        if key in metadata:
            summary[key] = metadata[key]

    return summary


def finalize_minigrid_summary(summary):
    """Convert DoorKey tracking sets into JSON-serializable summary fields."""
    if summary is None:
        summary = {
            'success': False,
            'goal_reached': False,
            'picked_up_key': False,
            'door_opened': False,
            'visited_cells': set(),
            'final_agent_pos': None,
            'final_agent_dir': None,
            'carrying': None,
            'doors': [],
            'keys': [],
            'goals': [],
        }

    visited_cells = sorted([list(cell) for cell in summary['visited_cells']])
    summary = dict(summary)
    summary['visited_cells'] = visited_cells
    summary['visited_cells_count'] = int(len(visited_cells))
    return summary


def main():
    print({section: dict(config[section]) for section in config.sections()})
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']
    obs_noise_config = ObservationNoiseConfig.from_config(default_config)

    env = make_configured_env(env_type, env_id)
    try:
        input_size = get_observation_shape(env)
        output_size = env.action_space.n

        if 'Breakout' in env_id:
            output_size -= 1
    finally:
        env.close()

    is_render = False
    save_video = default_config.getboolean('SaveVideo', fallback=False)
    video_path = default_config.get('VideoPath', fallback='eval_results/gameplay.mp4')
    video_fps = default_config.getint('VideoFPS', fallback=60)
    if save_video:
        video_path = os.path.abspath(video_path)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        if os.path.exists(video_path):
            os.remove(video_path)
        print('Recording gameplay to {} at {} FPS'.format(video_path, video_fps), flush=True)
    else:
        video_path = None
    # Evaluation keeps the historical flat model paths; the selected RND loss
    # must match the model files copied or saved at these locations.
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)
    checkpoint_path = 'models/{}.checkpoint'.format(env_id)

    use_cuda = False
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    rnd_loss = default_config.get('RNDLoss', fallback='mse')

    sticky_action = False
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    agent = RNDAgent

    env_class = get_environment_class(default_config['EnvType'])

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        lam=lam,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        rnd_loss=rnd_loss
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch_load(model_path))
        agent.rnd.predictor.load_state_dict(torch_load(predictor_path))
        agent.rnd.target.load_state_dict(torch_load(target_path))
        checkpoint = torch_load(checkpoint_path, weights_only=False) if os.path.exists(checkpoint_path) else {}
    else:
        agent.model.load_state_dict(torch_load(model_path, map_location='cpu'))
        agent.rnd.predictor.load_state_dict(torch_load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch_load(target_path, map_location='cpu'))
        checkpoint = torch_load(checkpoint_path, map_location='cpu', weights_only=False) if os.path.exists(checkpoint_path) else {}
    print('End load... global_step: {}, global_update: {}'.format(
        checkpoint.get('global_step', 0),
        checkpoint.get('global_update', 0)))

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_class(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, video_path=video_path, video_fps=video_fps,
                        obs_noise_config=obs_noise_config)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])
    for parent_conn in parent_conns:
        parent_conn.send(0)

    initial_states = []
    visited_rooms = set()
    minigrid_summary = None
    initial_real_done = False
    for parent_conn in parent_conns:
        s, r, d, rd, lr, rooms, metadata, rnd_s = unpack_env_result(parent_conn.recv())
        initial_real_done = initial_real_done or bool(rd)
        # Atari workers report Montezuma room ids; MiniGrid workers report
        # DoorKey milestone/cell metadata.
        visited_rooms.update(rooms)
        minigrid_summary = merge_minigrid_summary(minigrid_summary, metadata)
        print('Initial step: reward={}, done={}, real_done={}, log_reward={}'.format(r, d, rd, lr), flush=True)
        initial_states.append(s)
    states = np.stack(initial_states)

    steps = 0
    rall = 0
    episode_done = rd
    intrinsic_reward_list = []
    last_action = None
    last_intrinsic_reward = None
    progress_interval = 100

    while not episode_done:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)
        last_action = int(actions[0]) if len(actions) > 0 else None

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd, lr, rooms, metadata, rnd_s = unpack_env_result(parent_conn.recv())
            rall += r
            next_states = s.reshape([1, 4, 84, 84])
            next_obs = rnd_s[3, :, :].reshape([1, 1, 84, 84])
            rewards.append(float(r))
            dones.append(bool(d))
            real_dones.append(bool(rd))
            log_rewards.append(float(lr))
            visited_rooms.update(rooms)
            minigrid_summary = merge_minigrid_summary(minigrid_summary, metadata)

        # Evaluation records the raw RND prediction error for the RND observation
        # stream. The saved trace is normalized after the episode for plotting.
        intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        last_intrinsic_reward = float(np.mean(intrinsic_reward))
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_states[:, :, :, :]
        episode_done = any(real_dones)

        if steps % progress_interval == 0 or episode_done:
            print(
                'Eval step: {}  action: {}  reward: {:.3f}  total_reward: {:.3f}  intrinsic: {:.6f}  done: {}'.format(
                    steps,
                    last_action,
                    float(np.mean(rewards)) if rewards else 0.0,
                    rall,
                    last_intrinsic_reward if last_intrinsic_reward is not None else 0.0,
                    episode_done),
                flush=True)

    for parent_conn in parent_conns:
        parent_conn.send({'cmd': 'close'})
    for parent_conn in parent_conns:
        parent_conn.recv()
        parent_conn.close()
    for work in works:
        work.join(timeout=10)

    intrinsic_rewards = np.asarray(intrinsic_reward_list, dtype=np.float32)
    if intrinsic_rewards.size > 0:
        # Post-hoc normalization makes intrinsic-reward traces comparable in
        # plots without changing the environment interaction above.
        reward_std = float(np.std(intrinsic_rewards))
        if reward_std > 0:
            normalized_intrinsic_rewards = (intrinsic_rewards - np.mean(intrinsic_rewards)) / reward_std
        else:
            normalized_intrinsic_rewards = intrinsic_rewards - np.mean(intrinsic_rewards)
    else:
        normalized_intrinsic_rewards = intrinsic_rewards

    result_dir = 'eval_results'
    os.makedirs(result_dir, exist_ok=True)
    int_reward_path = os.path.join(result_dir, 'int_reward.pkl')
    int_reward_csv_path = os.path.join(result_dir, 'int_reward.csv')
    summary_path = os.path.join(result_dir, 'summary.json')

    with open(int_reward_path, 'wb') as f:
        pickle.dump(normalized_intrinsic_rewards, f)
    np.savetxt(int_reward_csv_path, normalized_intrinsic_rewards.reshape(-1), delimiter=',')

    if env_type == 'minigrid':
        finalized_minigrid_summary = finalize_minigrid_summary(minigrid_summary)
        rooms_found = None
        visited_rooms_summary = None
    else:
        # For Montezuma, exploration coverage is summarized by unique room ids
        # collected from ALE RAM through MontezumaInfoWrapper.
        finalized_minigrid_summary = None
        rooms_found = int(len(visited_rooms))
        visited_rooms_summary = sorted(int(room) for room in visited_rooms)

    summary = {
        'env_id': env_id,
        'rnd_loss': rnd_loss.lower(),
        'steps': int(steps),
        'episodic_return': float(rall),
        'total_reward': float(rall),
        'rooms_found': rooms_found,
        'visited_rooms': visited_rooms_summary,
        'last_action': last_action,
        'last_intrinsic_reward': last_intrinsic_reward,
        'initial_real_done': initial_real_done,
        'intrinsic_reward_count': int(intrinsic_rewards.size),
    }
    if finalized_minigrid_summary is not None:
        summary.update({
            'success': finalized_minigrid_summary['success'],
            'goal_reached': finalized_minigrid_summary['goal_reached'],
            'picked_up_key': finalized_minigrid_summary['picked_up_key'],
            'door_opened': finalized_minigrid_summary['door_opened'],
            'visited_cells_count': finalized_minigrid_summary['visited_cells_count'],
            'visited_cells': finalized_minigrid_summary['visited_cells'],
            'minigrid': finalized_minigrid_summary,
        })
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Evaluation finished: steps={} total_reward={:.3f} intrinsic_samples={}'.format(
        summary['steps'], summary['total_reward'], summary['intrinsic_reward_count']), flush=True)
    print('Saved intrinsic rewards to {}'.format(os.path.abspath(int_reward_path)), flush=True)
    print('Saved intrinsic rewards CSV to {}'.format(os.path.abspath(int_reward_csv_path)), flush=True)
    print('Saved summary to {}'.format(os.path.abspath(summary_path)), flush=True)
    if video_path:
        print('Saved gameplay video to {}'.format(video_path), flush=True)

if __name__ == '__main__':
    main()
