import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import pybullet
import pdb


import argparse
import os

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)
import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback


def get_training_configuration(args):
    # Creates a configuration dictionary that
    # defines the training paramenters, control_mode,
    # network architecture, and reset, action, reward,
    # and observation types.

    if args.algo == "PPO":
        network_architecture = dict(activation_fn=th.nn.Tanh,
                                    net_arch=[
                                        dict(
                                            pi=[256, 128, 128],
                                            vf=[256, 128, 128])
                                            ])
    else:
        network_architecture = dict(activation_fn=th.nn.ReLU,
                                    net_arch=dict(
                                        pi=[128, 128, 128],
                                        qf=[128, 128, 128]))

    if args.log_path is not None:
        local_log_dir = args.log_path
    else:
        if not args.test:
            local_log_dir = (
                os.environ['HOME'] +
                "/rl/tmp/KukaReachEnv/")
        else:
            local_log_dir = "./rl/tmp/KukaReachEnv/"

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 5e8,
        "env_name": "KukaReachEnv-v0",
        "num_workers": 80,
        "env_max_steps": 1200,  # 5 sec. time [sec]/time_step
        "local_log_dir": local_log_dir,
        "model_save_freq": 5e2 if not args.train_single_env else 1e4,
        "policy_kwargs": network_architecture,
        "reward_type": "cost_visak",
        "eval_reward_type": "success",
        "notes": args.notes,
    }
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_single_env', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_path', help="path to the logs directory.")
    parser.add_argument('--notes', help="log extra notes to wandb.")
    parser.add_argument('--algo', default="PPO",
                        help="training algorithm. "
                        "Valid options are [PPO, DDPG, SAC, TD3]. "
                        "Default: PPO")
    args = parser.parse_args()


    config = get_training_configuration(args=args)

    env_name = config["env_name"]
    if args.test:
        num_env = 2
    elif args.train_single_env or args.algo == "DDPG" or args.algo == "TD3":
        num_env = 1
    else:
        num_env = config["num_workers"]
    log_dir = config["local_log_dir"]
    total_timesteps = config["total_timesteps"] if not args.test else 5000
    policy_type = config["policy_type"]
    env_max_steps = config["env_max_steps"] if not args.test else 50
    policy_kwargs = config["policy_kwargs"]
    eval_freq = config["model_save_freq"]
    rew_type = config["reward_type"]
    eval_rew_type = config["eval_reward_type"]

    if args.test:
        run = wandb.init(mode="disabled")
    else:
        run = wandb.init(
            project="sb3_tactile_EE_reach_bullet",
            config=config,
            sync_tensorboard=True,  # Auto-upload sb3's tensorboard metrics.
            monitor_gym=True,  # Auto-upload the videos of agents playing.
            save_code=True,
        )

    if args.train_single_env:
        # pybullet.connect(pybullet.GUI)
        env = gym.make(env_name,
                       maxSteps=env_max_steps,
                       reward_type=rew_type,
                       renders=True,
                       debug=args.debug,
                       )
        #pdb.set_trace()
        check_env(env)
    else:
        env = make_vec_env(env_name,
                           n_envs=num_env,
                           seed=0,
                           vec_env_cls=SubprocVecEnv,
                           env_kwargs={
                               'maxSteps': env_max_steps,
                               'reward_type': rew_type,
                           })

    # Create the model.
    if args.algo == "PPO":
        model_class = PPO
    else:
        if args.algo == "DDPG":
            model_class = DDPG
        elif args.algo == "TD3":
            model_class = TD3
        elif args.algo == "SAC":
            model_class = SAC

    if args.test:
        if args.algo == "PPO":
            model = model_class(policy_type, env, n_steps=4, n_epochs=2,
                                batch_size=8, policy_kwargs=policy_kwargs)
        else:
            model = model_class(
                policy_type,
                env,
                policy_kwargs=policy_kwargs,
            )
    else:
        if args.algo == "PPO":
            n_steps = 101 if not args.train_single_env else 2048 #2048 #int(2048*2/num_env) int(time_limit/env.gym_time_step)
            model = model_class(policy_type,
                                env,
                                n_steps=n_steps,
                                n_epochs=50,
                                # In SB3, this is the mini-batch size.
                                # https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst
                                batch_size=num_env*2,  # factor of n_steps * n_envs
                                # learning_rate=1e-3,
                                # ent_coef=0.01,
                                verbose=1,
                                tensorboard_log=log_dir + "runs/{}".format(run.id),
                                policy_kwargs=policy_kwargs)
        else:
            model = model_class(policy_type,
                                env,
                                verbose=1,
                                tensorboard_log=log_dir + "runs/{}".format(run.id),
                                policy_kwargs=policy_kwargs)

    # Create a separate evaluation environment.

    eval_env = gym.make(env_name,
                        maxSteps=env_max_steps,
                        reward_type=eval_rew_type,
                        renders = True if not args.train_single_env else False,
                        )
    eval_env = DummyVecEnv([lambda: Monitor(eval_env)])

    # Record a video every n=1 evaluation rollouts.
    n = 1
    eval_env = VecVideoRecorder(
                    eval_env,
                    log_dir+"videos/{}".format(run.id),
                    record_video_trigger=lambda x: x % n == 0,
                    video_length=1200)

    eval_dir = log_dir+'eval_logs/{}'.format(run.id)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Use deterministic actions for evaluation.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir+'eval_logs/{}'.format(run.id),
        log_path=eval_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False)

    if not args.test:
        # Log best model to wandb.
        artifact = wandb.Artifact(name='best-model', type='model')
        artifact.add_dir(eval_dir)
        wandb.log_artifact(artifact)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            WandbCallback(
                gradient_save_freq=1e3,
                model_save_path=log_dir+"models/{}".format(run.id),
                verbose=2,
                model_save_freq=config["model_save_freq"],
            ),
            #eval_callback,
        ]
    )
    run.finish()