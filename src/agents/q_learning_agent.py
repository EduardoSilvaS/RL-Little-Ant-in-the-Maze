# Em src/agents/q_learning_agent.py

import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions: list, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.999, epsilon_min: float = 0.01):
        self.actions = actions
        self.alpha = alpha      # Taxa de aprendizado
        self.gamma = gamma      # Fator de desconto
        self.epsilon = epsilon    # Taxa de exploração (ações aleatórias)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # O Q-Table armazena o valor de cada ação em cada estado.
        # Usamos um defaultdict para facilitar, ele retorna 0.0 para estados nunca vistos.
        self.q_table = defaultdict(lambda: [0.0] * len(actions))

    def choose_action(self, state: tuple) -> int:
        """
        Decide a próxima ação usando a estratégia epsilon-greedy.
        """
        # Exploração: toma uma ação aleatória
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        # Exploitation: toma a melhor ação conhecida
        else:
            q_values = self.q_table[state]
            return q_values.index(max(q_values))

    def learn(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Atualiza o Q-Table com base na experiência (estado, ação, recompensa).
        Esta é a fórmula do Q-Learning.
        """
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        
        # A fórmula de atualização do valor Q
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        """Reduz a taxa de exploração ao longo do tempo."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay