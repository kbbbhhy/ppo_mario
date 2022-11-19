#the change vision for PPO_MARIO:wq
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil
import collections


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--acc_path",type=str,default="acc")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    #parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_batches', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    batches=opt.num_batches
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
    process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0
    acc=[]
    a=set()
    states_shape=(opt.num_processes,opt.num_local_steps,4,84,84)
    scalar_shape=(opt.num_processes,opt.num_local_steps)
    while True:
        # if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        local_flag_get=False
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        good_actions=np.zeros(scalar_shape)
        good_states=np.zeros(states_shape)
        #states_container=collections.Counter()
        good_prob=np.zeros(scalar_shape)
        good_rewards=np.zeros(scalar_shape)
        good_values=np.zeros(scalar_shape)
        good_episode_dones=np.zeros(scalar_shape)
        for now_step in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            #print(f"The value is {value}")
            #return
            #the_value=value.detach().cpu()
            #print(f"The changed value is {the_value}")
            #return
            the_state=curr_states.squeeze().detach().cpu()
            values.append(value.squeeze())
            the_the_value=value.squeeze().detach().cpu()
            #print(f"state {the_state} and the length is {len(the_state)}")
            #print(f"value {the_the_value} and the length is {len(the_the_value)}")
            #print(f"The changed squeeze value is {the_the_value}")
            #print(f"state shape{the_state.shape}")
            #return
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            the_log_policy=old_log_policy.detach().cpu()
            good_prob[:,now_step]=the_log_policy
            the_action=action.detach().cpu()
            good_actions[:,now_step]=the_action
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            #print(f"The state is {state}")
            #print(f"The reward is {reward}")
            #print(f"The done information now is {done}")
            #print(f"The info information now is {info}")
            #return 
            if info[0]["flag_get"]==True:
                local_flag_get=True
            state = torch.from_numpy(np.concatenate(state, 0))
            #print(f"The state is {state}")
            #print(f"The state length is {len(state)}")
            #return
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state
            the_reward=reward.detach().cpu()
            the_done=done.detach().cpu()
            good_rewards[:,now_step]=the_reward
            good_values[:,now_step]=the_the_value
            good_states[:,now_step]=the_state
            good_episode_dones[:,now_step]=the_done
            #print(f"Reward buffer{good_rewards}")
            #print(f"values buffer{good_values}")
            #print(f"done buffer{good_episode_dones}")
            #return

        #print(f"Reward buffer{good_rewards}")
        #print(f"Values buffer{good_values}")
        #print(f"States buffer{good_states}")
        #print(f"done buffer {good_episode_dones}")
        #print(f"log prob buffer{good_prob}")
        #print(f"actions that have taken{good_actions}")
        #return
        N=opt.num_processes
        T=opt.num_local_steps
        good_episode_dones=(good_episode_dones*-1)+1
        #print(f"Now the done{good_episode_dones}")
        #return
        _, next_value, = model(curr_states)
        next_value = next_value.squeeze()
        the_final_value=[]
        for t in next_value.squeeze().detach().cpu():
            the_final_value.append(t.item())
        #the_final_value=next_value.squeeze().detach().cpu().item()
        final_value=np.ones((N,1))
        final_value[:,0]=the_final_value
        #print(f"value 0 is {good_values[:,0]}")
        #print(f"final value is {the_final_value[:]}")
        #return
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        if local_flag_get==True:
            a.add(curr_episode)
        if curr_episode>100:
            if curr_episode-100 in a:
                a.remove(curr_episode-100)
            acc=len(a)/100
        else:
            acc=0
        gae = 0
        R = []
        gae_step=np.zeros((N,))
        advantages=np.zeros((N,T))
        gamma=opt.gamma
        tau=opt.tau
        #print(f"final value {final_value}")
        #print(f"other value {good_values}")
        #compute the last step
        delta=good_rewards[:,T-1]+gamma*final_value[:,0]*good_episode_dones[:,T-1]-good_values[:,T-1]
        gae_step=delta+gamma*tau*good_episode_dones[:,T-1]*gae_step
        advantages[:,T-1]=gae_step
        #compute other steps
        for t in reversed(range(T-1)):
            delta=good_rewards[:,t]+gamma*good_values[:,t+1]*good_episode_dones[:,t]-good_values[:,t]
            gae_step=delta+gamma*tau*good_episode_dones[:,t]*gae_step
            advantages[:,t]=gae_step
        good_reshape_reward=advantages+good_values
        train_data=[good_states,advantages,good_reshape_reward,good_values,good_actions,good_prob]
        train_data=[torch.tensor(x).to(device='cuda',dtype=torch.float) for x in train_data]
        train_data=[x.reshape((-1,)+x.shape[2:]) for x in train_data]
        #print(f"The flatten train data{train_data}")
        states,advantages,rewards,values,actions,probs=train_data
        epochs=opt.num_epochs
        for _ in range(epochs):
            indice=torch.randperm(N*T)
            for j in range(batches):
                batch_indices=indice[int(j*N*T/batches):int((j+1)*N*T/batches)]
                logits,value=model(states[batch_indices])
                new_policy=F.softmax(logits,dim=1)
                new_m=Categorical(new_policy)
                new_log_policy=new_m.log_prob(actions[batch_indices])
                ratio=torch.exp(new_log_policy-probs[batch_indices])
                actor_loss=-torch.mean(torch.min(ratio*advantages[batch_indices],torch.clamp(ratio,1.0-opt.epsilon,1.0+opt.epsilon)*advantages[batch_indices]))

                critic_loss=F.smooth_l1_loss(rewards[batch_indices],value.squeeze())
                entropy_loss=torch.mean(new_m.entropy())
                total_loss=actor_loss+critic_loss-opt.beta*entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
                optimizer.step()
        print(f"The episode is {curr_episode}")
        with open(opt.acc_path + '/' +'final_'+str(opt.world)+'_'+ str(opt.stage) + '.txt', 'a') as f:
            f.write(str(curr_episode)+' '+str(acc)+'\n')
        print(acc)
        if curr_episode>4000:
            return

if __name__ == "__main__":
    opt = get_args()
    train(opt)

