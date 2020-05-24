#following video: https://www.youtube.com/watch?v=UlJzzLYgYoE and taking notes
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# two classes
# 1) DeepQ Network, handle forward propogation operation
# 2) Agent that will hold object for DeepQ network and memory func to choose action and learning

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims,fc2_dims,n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims  = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 =  nn.Linear(*self.input_dims, self.fc1_dims)# * unpacks tuple or list
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)
        #DeepQ goal is to approx. an agents estimate of a state for all possible actions
        #therefore we need a network that outputs a number for all actions in action space
        self.optimizer = optim.Adam(self.parameters(), lr=lr)#adam optimizer
        self.loss = nn.MSELoss()#mean-squared error loss function
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')#uses GPu if available, else cpu
        #for PyTorch you must explicity tell the compiler to send the network to the gpu
        self.to(self.device)
        #full definition for network now done

    def forward(self,observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))#take ReLu activation of first connected layer
        x = F.relu(self.fc2(x))#pass forward to second connected layer
        actions = self.fc3(x)#actions are 3rd layer of activated second layer

        return actions

class Agent(object):
    def __init__(self,gamma, epsilon, lr, input_dims, batch_size, n_actions,
                            max_mem_size = 1000000, eps_end = 0.01, eps_dec = 0.996):
        """ gamma = discount factor. How much should our agent discount future rewards
        epsilon = for epsilon greedy. Epsilon greedy is a way of selecting actions where the agent selects randomly most of the time
        then acts greedily taking the best known action some of the time. and the fraction of the time it is random decreases
        lr = learning rate (what does this do ???)
        input_dims = input dimensions
        batch_size =
        n_actions = how many actions in our action_space ??? am i right about this
        max_mem_size =
        eps_end =when to stop epsilon
        eps_dec = how much to decrement epsilon by"""
    #we are decreasing epsilon with each step
    #the batch size and maximum size are what make DeepQ so powerful
    #as the agent plays the game it stores state, action ,reward, and new state transitions as well as done flags
    # in its memory, this allows it to sample from its memory to collect all the experiences its used to update its value for
    #actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0#index in stored mems
        self.Q_eval = DeepQNetwork(lr,n_actions = self.n_actions, input_dims = input_dims,fc1_dims = 256, fc2_dims = 256)
        #Agent has a DeepQ network. We do not include Agent  functionality in the DQN because an Agent is not a DQN, an Agent has a DQN
        self.state_memory = np.zeros((self.mem_size,*input_dims))
        self.new_state_memory = np.zeros((self.mem_size,*input_dims))#we use this to overwrite memories as the Agent acquires them
        self.action_memory = np.zeros((self.mem_size,self.n_actions),
                                            dtype=np.uint8)#must pass in unsigned 8-bit integer
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.uint8)#sequence of done flags

    def store_transition(self,state,action,reward,state_,terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions #one-hot encoded ???
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - terminal
        self.new_state_memory[index] = state_
        self.mem_cntr += 1
    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:#if we are doing a random action
            action = np.random.choice(self.action_space)
        else:#if we are doing a greedy action feed through network and calculate
            actions = self.Q_eval.forward(observation)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr > self.batch_size:#only learn when you have enough memories to fill batch_size
            self.Q_eval.optimizer.zero_grad()#zero out gradient (no exploding gradients)

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                        else self.mem_size
            batch = np.random.choice(max_mem,self.batch_size)

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values  = np.array(self.action_space, dtype=np.int32)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            #want q_target = q_eval for all action except max action

            batch_index = np.arange(self.batch_size, dtype = np.int32)
            q_target[batch_index, action_indices] = reward_batch + \
                        self.gamma*T.max(q_next,dim = 1)[0]*terminal_batch

            self.epsilon = self.epsilon*self.eps_dec if self.epsilon > \
                            self.eps_min else self.eps_min
            loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.Q_eval.state_dict(), 'trained_model.pt')
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.Q_eval.load_state_dict(T.load('trained_model.pt'))
