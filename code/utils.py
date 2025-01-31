import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque



class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, output_size),  # Output Q-values for each action
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            torch.tensor(state, dtype=torch.float32), 
            action, 
            reward, 
            torch.tensor(next_state, dtype=torch.float32), 
            done
        ))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train_q_learning(params, q_net, target_net, optimizer, replay_buffer, num_episodes=700, batch_size=64, gamma=0.999, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, episode_length=10, deployment_cost=0.1):
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay
    target_update_freq = 10
    replay_start_size = 100
    total_rewards = []
    plot_reward_series = []

    for episode in range(num_episodes):
        env = ContinuousEnvironment(params)
        state = torch.tensor([env.current_error, env.current_utility], dtype=torch.float32)
        episode_reward = 0
        reward_series = []
        for t in range(episode_length):  # Limit steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 1)  # Random action
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    #print(q_values)
                    action = torch.argmax(q_values).item()

            action_str = "collect" if action == 0 else "deploy"
            # print(action_str)
            #prev_utility = env.current_utility

            # Take action in the environment
            env.step(action_str) #already updated
            #env.render()
            #next_state = torch.tensor([env.current_utility], dtype=torch.float32)
            next_state = torch.tensor([env.current_error,env.current_utility], dtype=torch.float32)
            if action_str == "collect":
                reward = env.current_utility
            else:
                reward = env.current_utility - deployment_cost
            done = t == (episode_length-1)
            reward_series.append(reward)
            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Update state and accumulate reward
            state = next_state
            episode_reward += reward

            # Train the Q-network if enough samples are in the replay buffer
            if len(replay_buffer) > replay_start_size:
                transitions = replay_buffer.sample(batch_size)
                batch = {
                    "state": torch.stack([k[0] for k in transitions]), #maybe current utility+current_error
                    "action": torch.tensor([k[1] for k in transitions]),
                    "reward": torch.tensor([k[2] for k in transitions]),
                    "next_state": torch.stack([k[3] for k in transitions]),
                    "done": torch.tensor([k[4] for k in transitions], dtype=torch.float32)
                }

                # Compute Q-targets
                with torch.no_grad():
                    # Compute the target Q-value using the current environment's utility
                    #max_next_q_values = torch.tensor(ccf(env.current_utility, params["max_utility"]), dtype=torch.float32)
                    max_next_q_values = target_net(batch["next_state"]).max(1)[0]
                    # TD Target
                    q_targets = batch["reward"] + gamma * max_next_q_values * (1 - batch["done"])

                    # Predicted Q-value for the taken action
                q_targets = q_targets.float()
                q_values = q_net(batch["state"]).gather(1, batch["action"].unsqueeze(1)).squeeze().float()
                
                # Compute loss (MSE or Huber)
                loss_fn = nn.MSELoss()  # You can replace this with nn.HuberLoss() for more stability
                loss = loss_fn(q_values, q_targets) #estimated regret

                #if episode % 10 == 0:
                #    print(loss.item())
                # Update Q-network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Break if terminal state is reached
            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)

        # Update target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
        if len(total_rewards) == 0 or max(total_rewards)< episode_reward:
            env.render()
        if episode == 499:
            env.render(other_filename=f"strategy_noise_test")
        total_rewards.append(episode_reward)
        plot_reward_series.append(reward_series)
        #if episode % 10 == 0:
        #    print(f"Episode {episode}: Total Reward: {episode_reward:.5f}, Epsilon: {epsilon:.2f}")
        

    return total_rewards, plot_reward_series

def get_upperbound(start_utility, episode_length=10, deployment_cost=0.015, max_utility=1.0, ccf_beta=3.0):
    current_utility = start_utility
    utilities = [current_utility]
    for i in range(0,episode_length):
        if current_utility+deployment_cost<1.0:
            current_utility = ccf(current_utility,max_utility, ccf_beta=ccf_beta)-deployment_cost
        elif current_utility+deployment_cost == 1:
            current_utility = 1.0
        utilities.append(current_utility)
    return utilities
    
def get_discounted_utility(discount_factor,utilityseries):
    utility = 0
    # print(discount_factor)
    # print(utilityseries)
    for i in range(0,len(utilityseries)):
        utility = utility + utilityseries[i] * (1+discount_factor)**(-i)
    return utility

def ccf(ml_utility,max_utility, ccf_beta=3.0):
    if ml_utility<0:
        return 0.0
    # elif ml_utility<=0.25:
    #     #return 0.25+0.1*math.sin(500*math.pi*25*ml_utility)
    #     return 0.2+0.5*ml_utility
    else:
        return min(max_utility, 1/(np.exp(-ccf_beta*ml_utility)+1))

def get_error(data_size, c=1.0):
    if data_size<=0:
        return None
    else:
        return c/math.sqrt(data_size)

def get_angle(error, EPS=0.0000005):
    angle = 0
    if error >1:
        return 0
    if error <0:
        return 90
    
    if error <= EPS: #perfect learning
        angle = 90
    elif error >=1: # usually we can assume that 1/sqrt(n) <= 1 
        angle = 0
    else:
        angle = (1-error)*90
        
    return angle

def calculate_intersection(alpha, current_utility):
    alpha_rad = np.radians(alpha)

    #slope of the learning trajectory
    if alpha == 90:
        m = 0 
    else:
        m = -1 / np.tan(alpha_rad)

    #y = m * x + current_utility
    #diagonal: y = x
    #set equal: m * x + current_utility = x
    #=> (m - 1) * x = -current_utility
    if m != 1:
        x_intersection = -current_utility / (m - 1)
        y_intersection = x_intersection  #y = x
        return x_intersection, y_intersection
    else:
        return None  #no intersection
    

class ContinuousEnvironment:
    def __init__(self, params):
        self.current_utility = params["start_utility"]
        self.next_utility = self.current_utility
        self.time_preference = params["time_preference"]
        self.current_datasize = 0
        self.data_collection_size = params["data_collection_size"]
        self.accumulated_utility = []
        self.current_error = 1
        self.max_utility = params["max_utility"]
        self.points = []
        self.next_ml_utility = 0
        self.next_intersections = 0
        self.intersections = []
        self.next_angle = 0
        self.angles = []
        self.ccf_beta = params["ccf_beta"]
        self.filename = params["filename"]
        self.data_variance = params["data_variance"]
        
    def collect_data(self):
        self.current_datasize = self.current_datasize+self.data_collection_size
        return self.current_datasize

    def step(self, action):
        if action == "collect":
            self.accumulated_utility.append(self.current_utility)
            self.current_datasize = self.collect_data()
            self.current_error = get_error(self.current_datasize, c=self.data_variance)
            learning_trajectory = get_angle(self.current_error)
            self.next_angle = learning_trajectory
            self.next_ml_utility,_ = calculate_intersection(learning_trajectory,self.current_utility)
            self.next_utility = ccf(self.next_ml_utility,self.max_utility, ccf_beta=self.ccf_beta)
            self.intersection = self.next_ml_utility
        elif action == "deploy":
            #print(self.current_error)
            self.accumulated_utility.append(self.current_utility)
            self.current_datasize = 0
            self.current_utility = self.next_utility
            self.current_error = 1.0
            self.points.append((self.next_ml_utility, self.current_utility))
            self.intersections.append(self.next_ml_utility)
            self.angles.append(self.next_angle)
            self.next_angle = 0
        else:
            print(f"Action ´´{action}´´ not found")
        #print(f"Current Utility {self.current_utility}")
        #print(f"Next Utility {self.next_utility}")

    def render(self, other_filename=None):
        #diagonal f(x) = x
        fig, ax = plt.subplots(figsize=(10,7))

        x_vals = np.linspace(0, 1, 100)
        y_vals = x_vals
        
        x,y = zip(*self.points)
        ccf_vals = [ccf(xs,1.0, self.ccf_beta) for xs in x_vals]
        ax.plot(x_vals, y_vals, label="Diagonal f(x) = x", linestyle="--")
        ax.scatter(x,y, color='red')
        ax.scatter(self.intersections, self.intersections, color='green')
        #ax.plot(x,y, color='red', linestyle="--")
        ax.plot(x_vals,ccf_vals, color='red', linestyle="--")
        ys = [item for pair in zip(self.intersections,y) for item in pair]
        xs = [item for item in x for _ in range(2)]
        ax.plot(xs,ys, color='blue')
        
        plt.ylabel("ML+Human Utility")
        plt.xlabel("ML Utility")
        plt.title("Deployment Strategy")
        plt.grid()
        #plt.show()
        if other_filename:
            plt.savefig(other_filename, format='png', dpi=300)
        plt.savefig(self.filename, format='png', dpi=300)
        plt.close(fig)
