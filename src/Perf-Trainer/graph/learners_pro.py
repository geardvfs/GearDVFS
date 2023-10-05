import os
import os.path as path
import csv
import json
import time
import datetime

import math
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

# import graph.train_utils as train_utils
# from graph.model import DQN_AB, ReplayMemory, ReplayMemoryTime
# from graph.agents import DQN_AGENT_AB

"""
RL Learners are responsible for 
1. provide train/inference service
2. logging
3. response to server
Transitions dtype:
    <state(np.float32),action(list of int64/long),next_state(np.float32),reward(float)>
"""

def create_log(root, name):
    log_dir = os.path.join(root,name)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    log_file = str(datetime.datetime.now().strftime('%m%d-%H%M'))
    log_file = os.path.join(log_dir,log_file) + ".csv"
    return log_file

def get_ob(log_data):

    def cal_reward(utils, freqs, thermals):
        return r
    return stataes, reward

def inference(agent, model_c, buffer_i, eps):
    # x is current input
    # given states, ouput both belief state and further action result
    model_c.eval()
    AGENT.eps = eps
    with torch.no_grad():
        data = buffer_i[:]
        x = torch.from_numpy(data[-1].state).unsqueeze(0)
        data = [torch.from_numpy(item.state).unsqueeze(0) for item in data]
        context = model_c(data)
        state = torch.cat([x,context],dim=1)
        actions = AGENT.select_action(state)
    return state, actions

def train_context(model, t_buffer, logger):
    # Input time buffer for training context encoder
    epoches, b_size, learning_rate = 100, 100, 1e-2
    g_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    train_loader = torch.utils.data.DataLoader(t_buffer,shuffle=True,batch_size=b_size,drop_last=True)
    for j in range(epoches):
        for i, b in enumerate(train_loader):
            # To test: may include reward as input of context learner
            data = [item.state for item in b]
            recon_loss, kld_loss = model(data)
            loss = recon_loss+kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_scalar("Train/Context", loss, g_step)
            g_step+=1

def dqn_pro_nx(pipe, learn_state, init_params):

    '''AGENT initialization'''
    NAME = "DQN_PRO_NX"
    # Logger 
    ROOT = os.path.join(
        "./db/", NAME, str(datetime.datetime.now().strftime('%m%d-%H%M')))
    log_path, model_savepath = os.path.join(ROOT,"Log"), os.path.join(ROOT,"Model")
    if not os.path.isdir(log_path): os.makedirs(log_path)
    train_logger = SummaryWriter(log_path)

    # Training hyper-parameters
    EPS_START, EPS_END, EPS_DECAY = 0.99, 0.2, 1000
    n_update, n_batch, SYNC_STEP = 20, 100, 30

    # Task Info
    N_S, N_A, N_H, N_BUFFER = 12, [], 12000
    N_W = 10 # window size
    
    contexter = VRNNCell_V0(x_dim=N_S,z_dim=10,h_dim=15)
    AGENT = DQN_AGENT_AB(N_S, N_H, N_A, N_BUFFER, None)
    # AGENT.load_model("./db/DQN_PRO_NX/test/Model")

    # Replay Buffers for Context Learning and Inference (Agent has its own buffer)
    context_buffer = ReplayMemoryTime(capacity=N_BUFFER, w=N_W) # buffer for training context
    inference_buffer = QueueBuffer(N_W) # buffer for inference, size = window_size

    # Reset Training States and Data
    learn_state.value = 1
    prev_state, prev_actions = [None]*2
    record_count, test_count, agent_record_count, n_round, g_step = [0]*5

    # Response to start loop
    pipe.send("ready")

    while True:
        # wait for command
        msg = pipe.recv()
        cmd = msg['cmd']
        print("receive pipe message {} from {}".format(cmd,os.getpid()))

        if cmd == "RECORD_0":
            '''Record new buffers for context learning'''

            # Extract state(require np.float32), rewards(float)
            state, reward = get_ob(msg['data'])

            # Actions Selection (or use a pure random action selection if "first impression")
            if len(inference_buffer) == inference_buffer.capacity:
                # Only Inference when inference buffer is full
                _, actions = inference(AGENT, contexter, state, inference_buffer, eps=0.5)
            else:
                # Random actions if inference buffer is not full
                actions = [np.random.randint(i) for i in N_A]

            # Record in Both Context Buffer and Inference Buffer
            if record_count!=0: 
                context_buffer.push(prev_state, prev_actions, state, reward)
                inference_buffer.push(prev_state, prev_actions, state, reward)
            prev_state, prev_actions = state, actions
            # Write data of a sample slot logFile.write(...)
            record_count+=1

        elif cmd == "CONTEXT":
            '''Train context model'''

            learn_state.value = 0 # disable inference
            context_losses = train_context(contexter, context_buffer, train_logger)
            
            # Save context model

            # Reset initial states/actions to None
            prev_state,prev_action,record_count = None,None,0
            # Reset Buffers
            del context_buffer
            inference_buffer = QueueBuffer(N_W)

            learn_state.value = 1

        elif cmd  == "RECORD":
            '''Record buffers for RL learning'''

            # Extract state(require np.float32), rewards(float)
            state, reward = get_ob(msg['data'])

            # EPS determined by g_step
            AGENT.eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * g_step / EPS_DECAY)

            # Actions Selection (or use a pure random action selection if "first impression")
            if len(inference_buffer) == inference_buffer.capacity:
                # Only Inference when inference buffer is full
                full_state, actions = inference(AGENT, contexter, state, inference_buffer, eps=0.5)
                if agent_record_count != 0:
                    AGENT.mem.push(prev_state_full, prev_actions, full_state, reward)
                prev_state_full = full_state
                agent_record_count += 1

            else:
                # Random actions if inference buffer is not full
                actions = [np.random.randint(i) for i in N_A]

            # Record in Both Agent Buffer and Inference Buffer
            if record_count!=0: 
                inference_buffer.push(prev_state, prev_actions, state, reward)

            prev_state, prev_actions = state, actions
            prev_state_full = full_state
            record_count+=1
            

        elif cmd == "RL":
            '''Train RL Controller'''

            learn_state.value = 0 # disable inference
            losses = AGENT.train(n_round,n_update,n_batch)
            g_step = train_utils.log_scalar_list(train_logger, "Train/Loss", g_step, losses)

            # Reset initial states/actions to None
            prev_state,prev_action,record_count = None,None,0
            prev_state_full, agent_record_count = None, 0

            # Reset Inference Buffer
            inference_buffer = QueueBuffer(N_W)

            # save model
            AGENT.save_model(n_round, model_savepath)
            n_round += 1

            if n_round % SYNC_STEP == 0: AGENT.sync_model()
            learn_state.value = 1


        elif cmd == "Test":
            '''Test Phase: only run inference'''

            # Extract state 
            state,reward = get_ob(msg['data'])
            # Inference
            AGENT.eps = 0

            # Actions Selection (or use a pure random action selection if "first impression")
            if len(inference_buffer) == inference_buffer.capacity:
                # Only Inference when inference buffer is full
                full_state, actions = inference(AGENT, contexter, state, inference_buffer, eps=0.5)
            else:
                # Random actions if inference buffer is not full
                actions = [np.random.randint(i) for i in N_A]

            pipe.send({"action":int(action)})
            if test_count==0:
                test_log_file = create_log(ROOT,"TEST")
                test_log = open(test_log_file,'w',newline='')
                test_writer = csv.DictWriter(test_log, msg['data'].keys())
                test_writer.writeheader()
            test_writer.writerow(msg['data'])
            test_count+=1
            test_log.flush()

        elif cmd == "END_TEST":
            '''End Test and return results'''

            pass

        else:
            print("Invalid Command!")

if __name__== "__main__":
    import train_utils as train_utils
    from model import DQN_AB, ReplayMemory, ReplayMemoryTime, QueueBuffer
    from agents import DQN_AGENT_AB
    from context import VRNNCell_V0
    # Test Code
    NAME = "DQN_PRO_NX"
    # Logger 
    ROOT = os.path.join(
        "./db/", NAME, str(datetime.datetime.now().strftime('%m%d-%H%M')))
    log_path, model_savepath = os.path.join(ROOT,"Log"), os.path.join(ROOT,"Model")
    if not os.path.isdir(log_path): os.makedirs(log_path)
    train_logger = SummaryWriter(log_path)

    # Training hyper-parameters
    EPS_START, EPS_END, EPS_DECAY = 0.99, 0.2, 1000
    n_update, n_batch, SYNC_STEP = 20, 100, 30

    # Task Info
    # (ob, act, cont, hiden, window size)
    N_X, N_A, N_B, N_H, N_BUFFER = 12, [10,5], 10, 15, 12000
    N_S, N_W = N_B + N_X, 10 # window size
    
    contexter = VRNNCell_V0(x_dim=N_X,z_dim=10,h_dim=N_B)
    AGENT = DQN_AGENT_AB(N_S, N_H, N_A, N_BUFFER, None)
    inference_buffer = QueueBuffer(N_W)

    # Inference Test
    for i in range(N_W):
        inference_buffer.push(np.ones(N_X,dtype=np.float32),2,3,4)
    print(len(inference_buffer)==inference_buffer.capacity)
    h, actions = inference(AGENT, contexter, inference_buffer, 1)
    print(h.size(),actions)

    # Context Training Test
    context_buffer = ReplayMemoryTime(capacity=N_BUFFER, w=N_W) # buffer for training context
    for i in range(1000):
        context_buffer.push(np.ones(N_X,dtype=np.float32)*0.5,2,3,4)
    train_context(contexter, context_buffer, train_logger)
    # Record for RL Test


    # RL training test
