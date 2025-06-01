import numpy as np
import random
from collections import deque
import math
from neural_network import NeuralNetwork

class Agent:
    def __init__(self, neural_network, is_hunter, epsilon=0.9, min_epsilon=0.05, max_episodes=1000):
        self.nn = neural_network
        self.target_nn = NeuralNetwork(self.nn.layer_sizes, self.nn.dropout_rate)
        self.target_nn.weights = [w.copy() for w in self.nn.weights]
        self.target_nn.biases = [b.copy() for b in self.nn.biases]
        self.target_nn.bn_means = [m.copy() for m in self.nn.bn_means]
        self.target_nn.bn_vars = [v.copy() for v in self.nn.bn_vars]
        self.target_nn.bn_gamma = [g.copy() for g in self.nn.bn_gamma]
        self.target_nn.bn_beta = [b.copy() for b in self.nn.bn_beta]
        self.is_hunter = is_hunter
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_episodes = max_episodes
        self.gamma = 0.99
        self.replay_memory = deque(maxlen=10000)
        self.priorities = deque(maxlen=10000)
        self.last_actions = deque(maxlen=5 if is_hunter else 6)
        self.steps = 0
        self.total_reward = 0  # For diagnostics

    def save_model(self, filename):
        self.nn.save(filename)

    def get_action(self, state, episode):
        # Slower epsilon decay to encourage exploration
        self.epsilon = self.min_epsilon + (0.95 - self.min_epsilon) * math.exp(-episode / 500)
        if random.random() < self.epsilon:
            action = random.randint(0, 5)
        else:
            q_values = self.nn.predict(state)
            action = np.argmax(q_values)
        self.last_actions.append(action)
        return action

    def update_target_network(self):
        self.steps += 1
        if self.steps % 100 == 0:
            self.target_nn.weights = [w.copy() for w in self.nn.weights]
            self.target_nn.biases = [b.copy() for b in self.nn.biases]
            self.target_nn.bn_means = [m.copy() for m in self.nn.bn_means]
            self.target_nn.bn_vars = [v.copy() for v in self.nn.bn_vars]
            self.target_nn.bn_gamma = [g.copy() for g in self.nn.bn_gamma]
            self.target_nn.bn_beta = [b.copy() for b in self.nn.bn_beta]

    def train(self, batch_size=64):
        if len(self.replay_memory) < batch_size:
            return
        priorities = np.array(self.priorities)
        if priorities.sum() == 0:
            probs = np.ones(len(self.replay_memory)) / len(self.replay_memory)
        else:
            probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.replay_memory), batch_size, p=probs)
        batch = [self.replay_memory[i] for i in indices]
        states = []
        targets = []
        for idx, (state, action, reward, next_state, done) in enumerate(batch):
            self.total_reward += reward  # Track cumulative reward
            target = self.nn.predict(state)
            if done:
                target[action] = reward
            else:
                next_actions = np.argmax(self.nn.predict(next_state))
                next_q_value = self.target_nn.predict(next_state)[next_actions]
                target[action] = reward + self.gamma * next_q_value
            states.append(state)
            targets.append(target)
            td_error = abs(target[action] - self.nn.predict(state)[action]) + 1e-5
            self.priorities[idx] = td_error
        self.nn.train_batch(states, targets)
        self.update_target_network()
        # Log diagnostics every 100 steps
        if self.steps % 100 == 0:
            print(f"{'Hunter' if self.is_hunter else 'Prey'} - Steps: {self.steps}, Total Reward: {self.total_reward:.2f}, Epsilon: {self.epsilon:.3f}")