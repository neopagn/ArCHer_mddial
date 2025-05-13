import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


# def take_action(agent, tokenizer, observation, decode_f=lambda x: x,
#                 noise_std = 0, temperature = 2.0, do_sample=True):
#     raw_action = decode_f(agent.get_action(observation))
#     raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
#     raw_action = [a.split('\n')[0] for a in raw_action]
#     return raw_action


def batch_interact_environment(agent, tokenizer, env, num_trajectories,
        post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x,
        env_idx = None):
    """
    in a batched way, interact with the environments to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    all_trajectories = []
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        trajectories = [[] for _ in range(bsize)]
        batch_obs = env.reset(idx=env_idx)
        batch_done = [False,]*bsize
        steps = 0
        while not all(batch_done):
            steps += 1
            action = agent.get_action(batch_obs)
            # Xác định action nào là chẩn đoán
            is_diag = []
            for a in action:
                a_strip = a.strip()
                # Có thể điều chỉnh điều kiện này cho phù hợp với format action chẩn đoán
                if a_strip.startswith('Based on your symptoms') or a_strip.startswith('Doctor: Based on your symptoms'):
                    is_diag.append(True)
                else:
                    is_diag.append(False)
            # Tạo batch_return: nếu là chẩn đoán thì gọi diagnose, ngược lại gọi step
            batch_return = [None]*bsize
            for i in range(bsize):
                if batch_done[i]:
                    continue
                if is_diag[i]:
                    result = env.env_list[i].diagnose(action[i])
                else:
                    result = env.env_list[i].step(action[i])
                next_obs, r, done = result
                trajectories[i].append({"observation": batch_obs[i],
                                        "next_observation": next_obs,
                                        "reward": r,
                                        "done": done,
                                        "action": action[i]})
                batch_obs[i] = next_obs
                batch_done[i] = done
                if done: 
                    print(f"[Episode {i}] Reward: {r} | Ground truth: {env.env_list[i].curr_disease} | Agent diagnosis: {action[i]}")
            # obs = next_obs
        print(trajectories[0][-1]["next_observation"])
        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory)))
                              for trajectory in trajectories]
    return all_trajectories
