from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data

"""
Network backbones for RL Agents
Available Backbones:
1. Backbone for Vanilla Agent
2. Backbone for action branching

Replay buffers for storing state transitions
1. vanilla replay buffer
2. time series replay buffer
"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# RL Controllers
class DQN_v0(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        super(DQN_v0, self).__init__()
        self.fc1 = nn.Linear(in_dim, 25)
        self.fc2 = nn.Linear(25,25)
        self.out = nn.Linear(25, out_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        out = self.out(x2)
        return out

# RL Controller with action branching
class DQN_AB(nn.Module):
    def __init__(self, s_dim=10, h_dim=25, branches=[1,2,3]):
        super(DQN_AB, self).__init__()
        self.s_dim, self.h_dim = s_dim, h_dim
        self.branches = branches
        self.shared = nn.Sequential(nn.Linear(self.s_dim, self.h_dim), nn.ReLU())
        self.shared_state = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU())
        self.domains, self.outputs = [], []
        for i in range(len(branches)):
            layer = nn.Sequential(nn.Linear(self.h_dim, self.h_dim), nn.ReLU())
            self.domains.append(layer)
            layer_out = nn.Sequential(nn.Linear(self.h_dim*2, branches[i]))
            self.outputs.append(layer_out)

    def forward(self, x):
        # return list of tensors, each element is Q-Values of a domain
        f = self.shared(x)
        s = self.shared_state(f)
        outputs = []
        for i in range(len(self.branches)):
            branch = self.domains[i](f)
            branch = torch.cat([branch,s],dim=1)
            outputs.append(self.outputs[i](branch))

        return outputs



class DQN_v1(nn.Module):
    """
    CNN based DQN for openai gym CartPole-v0
    """
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class QueueBuffer(torch.utils.data.Dataset):
    """
    Basic ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def __getitem__(self, idx):        
        return self.memory[idx] 

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(None)
        if len(self.memory) > self.capacity:
            # to avoid index out of range
            self.memory.pop(0)
        transition = Transition(*args)
        self.memory[-1] = transition


class ReplayMemory(torch.utils.data.Dataset):
    """
    Basic ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __getitem__(self, idx):        
        return self.memory[idx] 

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # to avoid index out of range
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


class ReplayMemoryTime(torch.utils.data.Dataset):
    """
    Time series ReplayMemory class. 
    Note: Memory should be filled before load.
    """
    def __init__(self, capacity, w):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.w = w

    def __getitem__(self, idx):        
        return self.memory[idx:idx+self.w] 

    def __len__(self):
        return len(self.memory) - self.w + 1

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # to avoid index out of range
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity


if __name__ == "__main__":

    m,m1,m_time,m_branch = ReplayMemory(10), ReplayMemory(10), ReplayMemoryTime(10,2), ReplayMemory(10)
    for i in range(11):
        # Pytorch only converts numpy and basic types into tensor
        # For other kinds of return type pytorch will return as list
        m.push(np.array([i,1.0],dtype=np.float32),2,3,4)
        m1.push([i,1.0],2,3,4)
        m_time.push(np.array([i,1.0],dtype=np.float32),2,3,4)
        m_branch.push(np.array([i,1.0],dtype=np.float32),np.array([i,2.0],dtype=np.float32),3,4)

    train_loader = torch.utils.data.DataLoader(m, shuffle=False,batch_size=3)
    train_loader1 = torch.utils.data.DataLoader(m1, shuffle=False,batch_size=3,drop_last=True)
    train_loader_time = torch.utils.data.DataLoader(m_time, shuffle=False,batch_size=3,drop_last=True)
    train_loader_branch = torch.utils.data.DataLoader(m_branch, shuffle=False,batch_size=3,drop_last=True)

    print("Push numpy into memory -> tensor")
    for i, b in enumerate(train_loader):
        print(type(b.state),b.state,b.action.size())
        break

    print("Push list into memory -> list of tensors")
    # different features are divided in the list
    for i, b in enumerate(train_loader1):
        print(type(b.state),b.state)
        break

    print("Time series dataset -> list of tensors(tuple)\nEach element contains one time step\n[time, batch, dim]")
    for i, b in enumerate(train_loader_time):
        print(b[0].state)
        break

    print("Multi action dataset -> tensor")
    for i, b in enumerate(train_loader_branch):
        print(type(b.action),b.action,b.action[:,0].size(),b.action[:,0])
        break

    # Test DQN
    model = DQN_v0(2,1)
    with torch.no_grad():
        for i, b in enumerate(train_loader):
            print(b.state.max(0,keepdim=True))
            print(model(b.state).numpy())
            break

    # Test VRM
    print("VRM inference test")
    from context import VRNNCell_V0
    import time
    model = VRNNCell_V0(x_dim=2,z_dim=10,h_dim=15)
    model.eval()
    
    with torch.no_grad():
        for i, b in enumerate(train_loader_time):
            data = [item.state for item in b]
            print(model(data).numpy().shape)
            # shape [batch, h_dim]
            break
    
    # Run time inference test for VRM
    t = time.time()
    with torch.no_grad():
        data = m_branch[:]
        data = [torch.from_numpy(item.state).unsqueeze(0) for item in data]
        print(data[0].size())
        print(model(data).numpy().shape)
    print("inference time:",time.time()-t)
    model.train()
    for i, b in enumerate(train_loader_time):
        data = [item.state for item in b]
        recon_loss, kld_loss = model(data)
        loss = recon_loss+kld_loss
        loss.backward()
        break

    

    model = DQN_AB(10,25,[10,15,12])
    sample_x = torch.rand((3,10))
    result = model(sample_x)
    for domain in result:
        print(domain.size())

    # Test Inference QueueBuffer
    b = QueueBuffer(10)
    for i in range(15):
        b.push(np.array([i,1.0],dtype=np.float32),2,3,4)
    print(b[:])

