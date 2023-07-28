import gym                  # type: ignore
# import env
import PPO_model
import torch
import time
import os
import copy
from env.fjsp_env import FJSPEnv
from typing import Dict, Any

def get_validate_env(env_paras: Dict[str, Any]):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "./data_dev/{0}{1}/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env: FJSPEnv, model_policy: PPO_model.HGNNScheduler):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    # dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, flag_sample=False, flag_train=False)       # `dones` removed
        state, _, dones = env.step(actions)     # second return: rewards, not used
        done = dones.all()  # type: ignore
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!!!!!!")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
