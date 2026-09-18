from agents import *
from envs import *
from utils import *
from config import *
from obs_noise import ObservationNoiseConfig, split_env_result_rnd_state
from torch.multiprocessing import Pipe

from tensorboardX import SummaryWriter

import numpy as np
import os
import random
import re
import shutil
import struct

from tensorboardX.proto.event_pb2 import Event

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


MAX_SEED = 2 ** 32 - 1


def unpack_env_result(result):
    """Split a worker response into policy state, rewards, done flags, and RND state."""
    result, rnd_state = split_env_result_rnd_state(result)
    s, r, d, rd, lr = result[:5]
    return s, r, d, rd, lr, rnd_state


def get_rng_state():
    """Capture process-level RNG state for checkpointable experiment resumes."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'torch_initial_seed': torch.initial_seed(),
    }


def set_rng_state(rng_state):
    """Restore process-level RNG state saved in a training checkpoint."""
    if not rng_state:
        return

    if rng_state.get('python') is not None:
        random.setstate(rng_state['python'])
    if rng_state.get('numpy') is not None:
        np.random.set_state(rng_state['numpy'])
    if rng_state.get('torch') is not None:
        torch.set_rng_state(rng_state['torch'])
    if rng_state.get('torch_cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])


def seed_path_from_config():
    seed_dir = default_config.get('SeedDir', fallback='seeds')
    seed_file = default_config.get('SeedFile', fallback='seed.txt')
    return os.path.join(seed_dir, seed_file)


def read_seed(seed_path):
    with open(seed_path, 'r') as seed_file:
        seed_text = seed_file.read().strip()

    try:
        seed = int(seed_text)
    except ValueError:
        raise ValueError('Seed file {} must contain a single integer, got {!r}'.format(seed_path, seed_text))

    if seed < 0 or seed > MAX_SEED:
        raise ValueError('Seed in {} must be between 0 and {}, got {}'.format(seed_path, MAX_SEED, seed))

    return seed


def write_seed(seed_path, seed):
    seed_dir = os.path.dirname(seed_path)
    if seed_dir:
        os.makedirs(seed_dir, exist_ok=True)

    with open(seed_path, 'w') as seed_file:
        seed_file.write('{}\n'.format(seed))


def create_seed():
    return random.SystemRandom().randint(0, MAX_SEED)


def apply_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_start_seed(is_load_model):
    """Create or load the run seed unless a checkpoint resume will restore RNG state."""
    load_seed = default_config.getboolean('LoadSeed', fallback=False)
    seed_path = seed_path_from_config()

    if is_load_model:
        if load_seed:
            print('LoadSeed is ignored because IsLoadModel=True; checkpoint RNG state will be restored.')
        return None

    if load_seed:
        seed = read_seed(seed_path)
        print('Loaded training seed {} from {}'.format(seed, seed_path))
    else:
        seed = create_seed()
        write_seed(seed_path, seed)
        print('Created training seed {} at {}'.format(seed, seed_path))

    apply_seed(seed)
    return seed


def get_worker_rng_states(parent_conns):
    if not parent_conns:
        return None

    for parent_conn in parent_conns:
        parent_conn.send({'cmd': 'get_rng_state'})
    return [parent_conn.recv() for parent_conn in parent_conns]


def set_worker_rng_states(parent_conns, worker_rng_states):
    if not parent_conns or not worker_rng_states:
        return

    for parent_conn, rng_state in zip(parent_conns, worker_rng_states):
        parent_conn.send({'cmd': 'set_rng_state', 'state': rng_state})
    for parent_conn in parent_conns[:len(worker_rng_states)]:
        parent_conn.recv()


def set_worker_seeds(parent_conns, base_seed):
    """Give each environment worker a deterministic offset from the run seed."""
    if not parent_conns or base_seed is None:
        return

    for idx, parent_conn in enumerate(parent_conns):
        worker_seed = (base_seed + idx + 1) % (MAX_SEED + 1)
        parent_conn.send({'cmd': 'set_seed', 'seed': worker_seed})
    for parent_conn in parent_conns:
        parent_conn.recv()


def running_mean_std_state(running_mean_std):
    return {
        'mean': running_mean_std.mean,
        'var': running_mean_std.var,
        'count': running_mean_std.count,
    }


def set_running_mean_std_state(running_mean_std, state):
    if not state:
        return
    running_mean_std.mean = state['mean']
    running_mean_std.var = state['var']
    running_mean_std.count = state['count']


def copy_event_files(source_dir, target_dir):
    if os.path.abspath(source_dir) == os.path.abspath(target_dir):
        return

    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if not filename.startswith('events.out.tfevents.'):
            continue
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        if not os.path.exists(target_path):
            shutil.copy2(source_path, target_path)


def last_logged_step(log_dir, tag):
    last_step = 0
    if not os.path.isdir(log_dir):
        return last_step

    for filename in os.listdir(log_dir):
        if not filename.startswith('events.out.tfevents.'):
            continue
        event_path = os.path.join(log_dir, filename)
        try:
            with open(event_path, 'rb') as event_file:
                while True:
                    header = event_file.read(12)
                    if not header:
                        break
                    if len(header) != 12:
                        break
                    record_length = struct.unpack('<Q', header[:8])[0]
                    event_data = event_file.read(record_length)
                    footer = event_file.read(4)
                    if len(event_data) != record_length or len(footer) != 4:
                        break

                    event = Event()
                    event.ParseFromString(event_data)
                    if event.HasField('summary') and any(
                            value.tag == tag for value in event.summary.value):
                        last_step = max(last_step, event.step)
        except (OSError, ValueError):
            continue
    return last_step


def save_checkpoint(agent, model_path, predictor_path, target_path, checkpoint_path, global_step, global_update,
                    sample_episode, total_episode, reward_rms, obs_rms, discounted_reward, parent_conns=None):
    """Save model weights and the normalization/RNG state needed to resume a run."""
    torch.save(agent.model.state_dict(), model_path)
    torch.save(agent.rnd.predictor.state_dict(), predictor_path)
    torch.save(agent.rnd.target.state_dict(), target_path)
    torch.save({
        'global_step': global_step,
        'global_update': global_update,
        'sample_episode': sample_episode,
        'total_episode': total_episode,
        'reward_rms': running_mean_std_state(reward_rms),
        'obs_rms': running_mean_std_state(obs_rms),
        'discounted_reward': discounted_reward.rewems,
        'optimizer': agent.optimizer.state_dict(),
        'rng_state': get_rng_state(),
        'worker_rng_states': get_worker_rng_states(parent_conns),
    }, checkpoint_path)


def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    env_id = default_config['EnvID']
    env_type = default_config['EnvType']
    obs_noise_config = ObservationNoiseConfig.from_config(default_config)
    for line in obs_noise_config.log_lines():
        print(line)
    is_load_model = default_config.getboolean('IsLoadModel')
    start_seed = configure_start_seed(is_load_model)

    env = make_configured_env(env_type, env_id)
    try:
        input_size = get_observation_shape(env)
        output_size = env.action_space.n

        if 'Breakout' in env_id:
            output_size -= 1
    finally:
        env.close()

    is_render = False

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])
    max_episodes = default_config.getint('MaxEpisodes')
    max_rollouts_per_env = default_config.getint('MaxRolloutsPerEnv')
    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])
    update_proportion = float(default_config['UpdateProportion'])
    # ``RNDLoss`` is the loss-condition switch for the thesis experiments. The
    # surrounding PPO/RND hyperparameters are read once from the shared config.
    rnd_loss = default_config.get('RNDLoss', fallback='mse')
    safe_env_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', env_id)
    safe_rnd_loss = re.sub(r'[^A-Za-z0-9_.-]+', '_', rnd_loss.lower())
    # Training outputs are namespaced by environment and loss to keep seed/run
    # artifacts for different metric conditions separate.
    model_dir = os.path.join('models', '{}_{}'.format(safe_env_id, safe_rnd_loss))
    os.makedirs(model_dir, exist_ok=True)
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)
    checkpoint_path = 'models/{}.checkpoint'.format(env_id)
    save_model_path = os.path.join(model_dir, '{}.model'.format(env_id))
    save_predictor_path = os.path.join(model_dir, '{}.pred'.format(env_id))
    save_target_path = os.path.join(model_dir, '{}.target'.format(env_id))
    save_checkpoint_path = os.path.join(model_dir, '{}.checkpoint'.format(env_id))
    if all(os.path.exists(path) for path in [save_model_path, save_predictor_path, save_target_path]):
        model_path = save_model_path
        predictor_path = save_predictor_path
        target_path = save_target_path
        checkpoint_path = save_checkpoint_path
    log_dir = model_dir

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    # RND uses separate running statistics for observation normalization and
    # intrinsic-reward scale normalization.
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma)

    agent = RNDAgent

    env_class = get_environment_class(default_config['EnvType'])

    # The same agent architecture and optimizer are used across loss choices;
    # only ``rnd_loss`` changes the RND prediction-error metric.
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
        update_proportion=update_proportion,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        rnd_loss=rnd_loss
    )

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_class(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, obs_noise_config=obs_noise_config)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    set_worker_seeds(parent_conns, start_seed)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    total_episode = 0
    global_update = 0
    global_step = 0

    if is_load_model:
        print('load model from {}...'.format(model_dir if model_path == save_model_path else 'models'))
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
        global_step = checkpoint.get('global_step', global_step)
        global_update = checkpoint.get('global_update', global_update)
        sample_episode = checkpoint.get('sample_episode', sample_episode)
        total_episode = checkpoint.get('total_episode', total_episode)
        set_running_mean_std_state(reward_rms, checkpoint.get('reward_rms'))
        set_running_mean_std_state(obs_rms, checkpoint.get('obs_rms'))
        discounted_reward.rewems = checkpoint.get('discounted_reward')
        if 'optimizer' in checkpoint:
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        set_rng_state(checkpoint.get('rng_state'))
        set_worker_rng_states(parent_conns, checkpoint.get('worker_rng_states'))
        copy_event_files(os.path.dirname(model_path), log_dir)
        sample_episode = max(
            sample_episode, last_logged_step(log_dir, 'data/reward_per_epi'))
        logged_update = max(
            last_logged_step(log_dir, 'data/reward_per_rollout'),
            last_logged_step(log_dir, 'data/int_reward_per_rollout'))
        if logged_update > global_update:
            print('Event history is ahead of the checkpoint; continuing counters from update {}.'.format(
                logged_update))
            global_update = logged_update
            global_step = max(global_step, global_update * num_worker * num_step)
        print('load finished! global_step: {}, global_update: {}, sample_episode: {}'.format(
            global_step, global_update, sample_episode))

    writer = SummaryWriter(logdir=log_dir)
    print('TensorBoard run directory: {}'.format(log_dir))

    # Initialize RND observation normalization from random interaction before
    # learning. On resume, a saved ``obs_rms`` avoids rebuilding those statistics.
    print('Start to initialize observation state.....')
    next_obs = []
    initialization_steps = 1 if is_load_model and checkpoint.get('obs_rms') else num_step * pre_obs_norm_step
    for step in range(initialization_steps):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        # fixed: states are now updated and are not zero after the warmup
        warmup_states = []
        for parent_conn in parent_conns:
            s, r, d, rd, lr, rnd_s = unpack_env_result(parent_conn.recv())
            warmup_states.append(s)
            next_obs.append(rnd_s[3, :, :].reshape([1, 84, 84]))
        states = np.stack(warmup_states)

        if not is_load_model and len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End initialization...')

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. Collect an n-step rollout from all workers using the current policy.
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr, rnd_s = unpack_env_result(parent_conn.recv())
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)
                next_obs.append(rnd_s[3, :, :].reshape([1, 84, 84]))

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            next_obs = np.stack(next_obs)
            total_episode += int(np.sum(real_dones))

            # Compute intrinsic reward from the RND observation stream. In
            # ``rnd_only`` noise runs, ``rnd_s`` is the noisy stack while
            # ``states`` remains the clean policy stack.
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs)
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

            if max_episodes > 0 and total_episode >= max_episodes:
                print('Reached MaxEpisodes: {}'.format(max_episodes))
                writer.flush()
                save_checkpoint(agent, save_model_path, save_predictor_path, save_target_path, save_checkpoint_path,
                                global_step, global_update, sample_episode, total_episode, reward_rms, obs_rms,
                                discounted_reward, parent_conns)
                writer.close()
                return

        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(total_reward).transpose().clip(-1, 1)
        total_action = np.stack(total_action).transpose().reshape([-1])
        total_done = np.stack(total_done).transpose()
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        total_ext_values = np.stack(total_ext_values).transpose()
        total_int_values = np.stack(total_int_values).transpose()
        total_logging_policy = np.vstack(total_policy_np)

        # Step 2. Normalize intrinsic reward scale using discounted intrinsic
        # returns, following the original RND training pipeline.
        total_int_reward = np.stack(total_int_reward).transpose()
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        total_int_reward /= np.sqrt(reward_rms.var)
        writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_prob', softmax(total_logging_policy).max(1).mean(), sample_episode)

        # Step 3. Build PPO targets and advantages. Extrinsic rewards are
        # episodic and clipped above; intrinsic rewards are treated as
        # non-episodic by passing zero done flags.
        ext_target, ext_adv = make_train_data(total_reward,
                                              total_done,
                                              total_ext_values,
                                              gamma,
                                              num_step,
                                              num_worker)

        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # Keep the reward-combination coefficients explicit so loss comparisons
        # do not change the PPO objective except through the intrinsic signal.
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. Update observation statistics after reward computation. The
        # predictor update below therefore uses the latest rollout statistics.
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Joint PPO and RND predictor update.
        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy)

        if max_rollouts_per_env > 0 and global_update >= max_rollouts_per_env:
            print('Reached MaxRolloutsPerEnv: {}'.format(max_rollouts_per_env))
            writer.flush()
            save_checkpoint(agent, save_model_path, save_predictor_path, save_target_path, save_checkpoint_path,
                            global_step, global_update, sample_episode, total_episode, reward_rms, obs_rms,
                            discounted_reward, parent_conns)
            writer.close()
            return

        if global_step % (num_worker * num_step * 100) == 0:
            print('Now Global Step: {}, Global Update: {}'.format(global_step, global_update))
            writer.flush()
            save_checkpoint(agent, save_model_path, save_predictor_path, save_target_path, save_checkpoint_path,
                            global_step, global_update, sample_episode, total_episode, reward_rms, obs_rms,
                            discounted_reward, parent_conns)


if __name__ == '__main__':
    main()
