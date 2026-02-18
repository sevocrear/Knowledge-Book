# Deep Reinforcement Learning: От Основ к Управлению Роботами и Автономными Автомобилями

## Table of Contents

1. [Введение](#введение)
2. [Основы Reinforcement Learning](#основы-reinforcement-learning)
3. [Deep Reinforcement Learning](#deep-reinforcement-learning)
4. [Value-Based Методы](#value-based-методы)
5. [Policy Gradient Методы](#policy-gradient-методы)
6. [Actor-Critic Методы](#actor-critic-методы)
7. [Современные Методы (2020-2026)](#современные-методы-2020-2026)
8. [Deep RL для Робототехники](#deep-rl-для-робототехники)
9. [Deep RL для Автономных Автомобилей](#deep-rl-для-автономных-автомобилей)
10. [Практические Реализации](#практические-реализации)
11. [Сравнение Методов](#сравнение-методов)
12. [Текущее Состояние и Тренды (2024-2026)](#текущее-состояние-и-тренды-2024-2026)
13. [References](#references)

---

## Введение

**Deep Reinforcement Learning (Deep RL)** — это комбинация **Reinforcement Learning (RL)** и **Deep Learning**, которая позволяет агентам обучаться оптимальному поведению в сложных средах, используя нейронные сети для аппроксимации функций ценности (value functions) или политик (policies).

### Ключевая Идея

**Как бы я объяснил это пятилетнему ребенку:**

Представь, что ты учишься играть в видеоигру. Сначала ты не знаешь, какие кнопки нажимать, но каждый раз, когда ты делаешь что-то хорошее (например, собираешь монету), игра говорит тебе "молодец!" и дает очки. Когда ты делаешь что-то плохое (например, падаешь в яму), игра говорит "плохо" и забирает очки. Со временем ты учишься, какие действия приводят к хорошим результатам, и начинаешь играть все лучше и лучше. Deep Reinforcement Learning — это то же самое, но для компьютера: он учится на опыте, пробуя разные действия и получая награды или штрафы, пока не научится играть (или управлять роботом, или водить машину) очень хорошо.

### Исторический Контекст

- **1950-е**: Формализация Reinforcement Learning (Bellman, MDP)
- **1980-1990-е**: Q-Learning, Policy Gradient методы
- **2013**: Deep Q-Network (DQN) — прорыв в Deep RL
- **2015-2017**: Развитие Actor-Critic методов (A3C, DDPG, TD3)
- **2017-2018**: Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC)
- **2019-2021**: Импульс в робототехнике, автономных автомобилях
- **2022-2026**: Transformer-based RL, Foundation Models для RL, Sim-to-Real transfer

### Почему Deep RL для Управления?

**Преимущества:**
- **Адаптивность**: Может адаптироваться к новым ситуациям
- **End-to-End обучение**: Прямо от сенсоров к действиям
- **Обработка сложных состояний**: Изображения, LiDAR, мультимодальные данные
- **Оптимизация долгосрочных целей**: Учитывает будущие последствия действий

**Вызовы:**
- **Sample Efficiency**: Требует много данных для обучения
- **Безопасность**: Критично для роботов и автомобилей
- **Sim-to-Real Gap**: Разница между симуляцией и реальностью
- **Объяснимость**: Сложно понять, почему агент принимает решения

---

## Основы Reinforcement Learning

### Формализация: Markov Decision Process (MDP)

**MDP** определяется кортежем $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **$\mathcal{S}$** — пространство состояний (states)
- **$\mathcal{A}$** — пространство действий (actions)
- **$\mathcal{P}(s'|s,a)$** — функция переходов (transition probability)
- **$\mathcal{R}(s,a,s')$** — функция награды (reward function)
- **$\gamma \in [0,1]$** — коэффициент дисконтирования (discount factor)

### Основные Компоненты

**1. Политика (Policy) $\pi$**

Политика определяет, какое действие выбрать в каждом состоянии:

- **Детерминированная**: $\pi: \mathcal{S} \to \mathcal{A}$
- **Стохастическая**: $\pi(a|s) = \mathbb{P}(A_t = a | S_t = s)$

**2. Функция Ценности Состояния (State Value Function) $V^\pi(s)$**

Ожидаемая сумма наград при следовании политике $\pi$ из состояния $s$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \Big| S_0 = s \right]$$

**3. Функция Ценности Действия (Action Value Function) $Q^\pi(s,a)$**

Ожидаемая сумма наград при выборе действия $a$ в состоянии $s$ и последующем следовании политике $\pi$:

$$Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \Big| S_0 = s, A_0 = a \right]$$

**4. Оптимальная Политика $\pi^*$**

Политика, которая максимизирует ожидаемую сумму наград:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \right]$$

### Bellman Уравнения

**Bellman уравнение для $V^\pi$:**

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} \mathcal{P}(s',r|s,a) \left[ r + \gamma V^\pi(s') \right]$$

**Bellman уравнение для $Q^\pi$:**

$$Q^\pi(s,a) = \sum_{s',r} \mathcal{P}(s',r|s,a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]$$

**Bellman оптимальности для $Q^*$:**

$$Q^*(s,a) = \sum_{s',r} \mathcal{P}(s',r|s,a) \left[ r + \gamma \max_{a'} Q^*(s',a') \right]$$

### Основные Алгоритмы RL

**1. Value Iteration**

Итеративное обновление $Q$-функции:

$$Q_{k+1}(s,a) = \sum_{s',r} \mathcal{P}(s',r|s,a) \left[ r + \gamma \max_{a'} Q_k(s',a') \right]$$

**2. Policy Iteration**

Чередование оценки политики и улучшения:

- **Policy Evaluation**: Вычисление $V^\pi$ для текущей политики
- **Policy Improvement**: Обновление политики: $\pi'(s) = \arg\max_a Q^\pi(s,a)$

**3. Q-Learning (Off-policy)**

Обновление $Q$-функции без следования текущей политике:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

где $\alpha$ — скорость обучения (learning rate).

---

## Deep Reinforcement Learning

### Проблема: Большие Пространства Состояний

В классическом RL таблицы $Q(s,a)$ или $V(s)$ работают только для малых пространств. Для реальных задач (изображения, высокие размерности) нужна **функциональная аппроксимация**.

### Решение: Нейронные Сети

**Deep RL** использует нейронные сети для аппроксимации:
- **$Q(s,a; \theta)$** — $Q$-функция с параметрами $\theta$
- **$V(s; \theta)$** — функция ценности состояния
- **$\pi(a|s; \theta)$** — политика

### Основные Подходы

1. **Value-Based**: Аппроксимация $Q$-функции (DQN, Rainbow DQN)
2. **Policy-Based**: Прямая оптимизация политики (REINFORCE, PPO)
3. **Actor-Critic**: Комбинация обоих подходов (A3C, DDPG, SAC, TD3)

---

## Value-Based Методы

### Deep Q-Network (DQN)

**DQN** (2013, 2015) — первый успешный метод Deep RL, который научил агента играть в Atari игры на уровне человека.

#### Архитектура

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """
    Deep Q-Network для дискретных действий
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
```

#### Ключевые Техники DQN

**1. Experience Replay**

Хранение переходов $(s_t, a_t, r_t, s_{t+1})$ в буфере и случайная выборка для обучения:

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)
```

**2. Target Network**

Отдельная сеть для вычисления целевых $Q$-значений, обновляемая периодически:

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.update_counter = 0
        
        # Основная сеть
        self.q_network = DQN(state_dim, action_dim)
        # Целевая сеть
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Текущие Q-значения
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Целевые Q-значения
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value
        
        # Loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Обновление целевой сети
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
```

#### Математическая Формулировка DQN

**Целевая функция:**

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta) \right)^2 \right]$$

где:
- $\theta$ — параметры основной сети
- $\theta^-$ — параметры целевой сети (обновляются периодически)
- $\mathcal{D}$ — буфер опыта (replay buffer)

**Градиент:**

$$\nabla_\theta L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta) \right) \nabla_\theta Q(s,a; \theta) \right]$$

### Улучшения DQN

**1. Double DQN (2015)**

Решает проблему переоценки $Q$-значений:

$$Y_t^{DoubleDQN} = r + \gamma Q(s', \arg\max_{a'} Q(s',a'; \theta); \theta^-)$$

**2. Dueling DQN (2016)**

Разделение $Q$-функции на ценность состояния и преимущество действия:

$$Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)$$

**3. Prioritized Experience Replay (2016)**

Приоритизация важных переходов для обучения:

$$p_i = |\delta_i| + \epsilon$$

где $\delta_i$ — TD-ошибка для перехода $i$.

**4. Rainbow DQN (2017)**

Комбинация всех улучшений DQN:
- Double DQN
- Dueling DQN
- Prioritized Replay
- Multi-step learning
- Distributional RL
- Noisy Networks

---

## Policy Gradient Методы

### Основная Идея

Вместо аппроксимации $Q$-функции, **Policy Gradient** методы напрямую оптимизируют политику $\pi_\theta(a|s)$.

### Policy Gradient Теорема

**Градиент ожидаемой награды:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t \right]$$

где:
- $\tau = (s_0, a_0, r_1, s_1, a_1, ..., s_T)$ — траектория
- $\hat{A}_t$ — оценка преимущества (advantage) в момент $t$

### REINFORCE (Monte Carlo Policy Gradient)

**Алгоритм:**

1. Собрать траекторию $\tau$ следуя политике $\pi_\theta$
2. Вычислить возврат $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$
3. Обновить параметры:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

**Реализация:**

```python
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()
    
    def finish_episode(self):
        # Вычислить дисконтированные награды
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Нормализация
        
        # Вычислить loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Очистить буферы
        self.saved_log_probs = []
        self.rewards = []
```

### Проблемы Policy Gradient

1. **Высокая дисперсия**: Оценки градиента имеют большую дисперсию
2. **Sample Efficiency**: Требует много траекторий
3. **Медленная сходимость**: Обновления могут быть нестабильными

---

## Actor-Critic Методы

### Основная Идея

**Actor-Critic** комбинирует:
- **Actor** (Policy): Выбирает действия
- **Critic** (Value Function): Оценивает качество действий

### Advantage Actor-Critic (A2C)

**Advantage Function:**

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**Обновление Actor:**

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) A^\pi(s,a)$$

**Обновление Critic:**

$$\phi \leftarrow \phi - \alpha_V \nabla_\phi \left( V_\phi(s) - \hat{V}^\pi(s) \right)^2$$

### Asynchronous Advantage Actor-Critic (A3C)

**A3C** (2016) использует несколько параллельных агентов для сбора опыта:

```python
class A3C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3C, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        shared_out = self.shared(state)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value
```

---

## Современные Методы (2020-2026)

### Proximal Policy Optimization (PPO)

**PPO** (2017) — один из самых популярных методов для непрерывного управления.

#### Основная Идея

Ограничение изменения политики для стабильного обучения:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

где:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ — отношение вероятностей
- $\epsilon$ — параметр обрезки (обычно 0.1-0.2)

**Реализация PPO:**

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Actor-Critic сеть
        self.policy = A3C(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = A3C(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def update(self, states, actions, rewards, dones):
        # Вычислить дисконтированные награды и advantages
        returns = []
        advantages = []
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G * (1 - dones[i])
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Преобразовать в тензоры
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = self.policy_old(states)[0].gather(1, actions.unsqueeze(1)).squeeze(1).detach()
        
        # Множественные обновления
        for _ in range(self.k_epochs):
            # Текущие вероятности
            probs, values = self.policy(states)
            new_log_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Отношение вероятностей
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Advantages
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Общий loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Оптимизация
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Обновить старую политику
        self.policy_old.load_state_dict(self.policy.state_dict())
```

### Soft Actor-Critic (SAC)

**SAC** (2018) — off-policy метод для непрерывных действий с максимальной энтропией.

#### Ключевые Особенности

1. **Максимизация энтропии**: Поощряет исследование
2. **Off-policy**: Использует replay buffer
3. **Непрерывные действия**: Работает с непрерывными пространствами действий

**Целевая функция:**

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

где $\mathcal{H}(\pi(\cdot|s_t))$ — энтропия политики, $\alpha$ — температура.

**Реализация SAC:**

```python
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    """
    Actor (политика) для SAC с непрерывными действиями
    """
    def __init__(self, state_dim, action_dim, action_range, hidden_dim=256):
        super(Actor, self).__init__()
        self.action_range = action_range
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # Ограничение для стабильности
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Reparameterization trick
        u = dist.rsample()
        action = self.tanh(u)
        
        # Вычисление log_prob с учетом tanh
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Масштабирование действия в диапазон
        action = action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
                 (self.action_range[1] + self.action_range[0]) / 2.0
        
        return action, log_prob

class Critic(nn.Module):
    """
    Critic (Q-функция) для SAC
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class SAC:
    def __init__(self, state_dim, action_dim, action_range, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_range = action_range
        
        # Actor (политика)
        self.actor = Actor(state_dim, action_dim, action_range)
        
        # Critic (две Q-сети для уменьшения переоценки)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        
        # Target networks
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
    
    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if deterministic:
            action, _ = self.actor(state_tensor)
        else:
            action, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0]
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # Обновление Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Обновление Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + 
                                   param.data * self.tau)
```

### Twin Delayed DDPG (TD3)

**TD3** (2018) — улучшение DDPG с тремя ключевыми техниками:

1. **Twin Critic**: Две Q-сети для уменьшения переоценки
2. **Delayed Policy Updates**: Обновление политики реже, чем критиков
3. **Target Policy Smoothing**: Добавление шума к целевым действиям

---

## Deep RL для Робототехники

### Особенности Робототехники

**Вызовы:**
- **Непрерывные действия**: Управление суставами, моторами
- **Высокая размерность**: Множество степеней свободы
- **Безопасность**: Критично избегать опасных действий
- **Sample Efficiency**: Реальные роботы медленные и дорогие
- **Sim-to-Real Gap**: Разница между симуляцией и реальностью

### Лучшие Методы для Робототехники

#### 1. Soft Actor-Critic (SAC)

**Почему SAC хорош для роботов:**
- Off-policy обучение (эффективное использование данных)
- Непрерывные действия
- Максимизация энтропии (естественное исследование)
- Стабильное обучение

**Применения:**
- Манипуляция объектами
- Локомоция (ходьба, бег)
- Управление дронами

#### 2. Proximal Policy Optimization (PPO)

**Почему PPO хорош:**
- Простота реализации
- Стабильное обучение
- Хорошо работает в симуляциях

**Применения:**
- Обучение в симуляторах (MuJoCo, PyBullet)
- Sim-to-Real transfer
- Задачи с дискретными и непрерывными действиями

#### 3. Domain Randomization

**Идея:** Обучение в симуляциях с различными параметрами для лучшего переноса в реальность:

```python
def domain_randomization():
    """
    Случайное изменение параметров симуляции
    """
    # Масса объектов
    object_mass = np.random.uniform(0.5, 2.0)
    
    # Трение
    friction = np.random.uniform(0.3, 1.5)
    
    # Визуальные параметры
    lighting = np.random.uniform(0.5, 1.5)
    texture = random.choice(['smooth', 'rough', 'textured'])
    
    return {
        'object_mass': object_mass,
        'friction': friction,
        'lighting': lighting,
        'texture': texture
    }
```

#### 4. Sim-to-Real Transfer Методы

**1. Domain Adaptation:**
- Обучение в симуляции, адаптация к реальности
- Использование adversarial training

**2. System Identification:**
- Идентификация параметров реального робота
- Адаптация симуляции под реальность

**3. Residual Learning:**
- Обучение остаточной политики для компенсации ошибок симуляции

#### 5. Imitation Learning + RL

**Идея:** Начать с демонстраций, затем улучшить через RL:

```python
class ImitationRL:
    def __init__(self, policy, expert_demos):
        self.policy = policy
        self.expert_demos = expert_demos  # Демонстрации эксперта
    
    def pretrain_with_imitation(self, epochs=100):
        """
        Предобучение на демонстрациях
        """
        for epoch in range(epochs):
            for demo in self.expert_demos:
                states, actions = demo
                # Обучение поведенческому клонированию
                predicted_actions = self.policy(states)
                loss = nn.MSELoss()(predicted_actions, actions)
                # ... оптимизация ...
    
    def fine_tune_with_rl(self, env, method='PPO'):
        """
        Доводка через RL
        """
        # Использовать PPO или SAC для улучшения политики
        pass
```

### Пример: Управление Манипулятором

```python
class RobotArmController:
    """
    Управление роботом-манипулятором через SAC
    """
    def __init__(self, state_dim=14, action_dim=7):
        # Состояние: позиция/ориентация end-effector, углы суставов, скорости
        # Действия: целевые углы суставов или скорости
        self.agent = SAC(state_dim, action_dim, action_range=(-1, 1))
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def train(self, env, episodes=1000):
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            while not env.done:
                # Выбрать действие
                action = self.agent.select_action(state)
                
                # Выполнить действие
                next_state, reward, done, info = env.step(action)
                
                # Сохранить в буфер
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Обучение
                if len(self.replay_buffer) > 1000:
                    batch = self.replay_buffer.sample(256)
                    self.agent.update(batch)
                
                state = next_state
                episode_reward += reward
            
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
```

### Современные Достижения в Робототехнике (2022-2026)

**1. Foundation Models для Роботов:**
- **RT-1, RT-2** (Google): Transformer-based роботы
- **PaLM-E**: Мультимодальный LLM для роботов
- **VLA (Vision-Language-Action) модели**: OpenVLA, F1-VLA, Octo
- **См. подробнее**: [Vision-Language-Action (VLA) Models](../vision-language-action-models-vla.md)

**2. Diffusion Policies:**
- Использование diffusion models для генерации действий
- Более плавные и естественные движения
- **Octo**: diffusion-based роботическая политика

**3. Hierarchical RL:**
- Высокоуровневые и низкоуровневые политики
- Долгосрочное планирование

**4. Open-Source Методы:**
- **OpenVLA**: полностью open-source VLA модель
- **Octo**: diffusion-based policy
- **См. подробнее**: [Vision-Based Robot Training Methods](../vision-based-robot-training-methods.md)

---

## Deep RL для Автономных Автомобилей

### Особенности Автономного Вождения

**Вызовы:**
- **Безопасность**: Критично для жизни людей
- **Сложная среда**: Динамические объекты, непредсказуемое поведение
- **Мультимодальные сенсоры**: Камеры, LiDAR, радары
- **Частичная наблюдаемость**: Не все объекты видны
- **Регуляторные требования**: Должны соответствовать стандартам

### Лучшие Методы для Автономных Автомобилей

#### 1. Imitation Learning (Behavioral Cloning)

**Идея:** Обучение на данных опытных водителей:

```python
class BehavioralCloning:
    """
    Обучение на демонстрациях водителей
    """
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(
            # CNN для обработки изображений
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)  # [steering, throttle, brake]
        )
        self.optimizer = optim.Adam(self.policy.parameters())
    
    def train(self, expert_demos):
        """
        Обучение на демонстрациях
        """
        for states, actions in expert_demos:
            predicted_actions = self.policy(states)
            loss = nn.MSELoss()(predicted_actions, actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

**Проблемы:**
- **Distribution Shift**: Ошибки накапливаются
- **Нет восстановления**: Не умеет исправлять ошибки

#### 2. DAgger (Dataset Aggregation)

**Решение проблемы Behavioral Cloning:**

1. Обучить начальную политику на демонстрациях
2. Собрать данные, следуя текущей политике
3. Попросить эксперта аннотировать эти данные
4. Переобучить на объединенном датасете
5. Повторить

#### 3. End-to-End Deep RL

**Использование Deep RL для прямого управления:**

```python
class AutonomousDrivingAgent:
    """
    Агент для автономного вождения с использованием PPO
    """
    def __init__(self):
        # Вход: изображения камеры + LiDAR + состояние автомобиля
        # Выход: [steering_angle, throttle, brake]
        self.agent = PPO(state_dim=512, action_dim=3)
    
    def process_observations(self, camera, lidar, vehicle_state):
        """
        Обработка мультимодальных наблюдений
        """
        # CNN для камеры
        camera_features = self.camera_encoder(camera)
        
        # PointNet для LiDAR
        lidar_features = self.lidar_encoder(lidar)
        
        # Объединение
        state = torch.cat([camera_features, lidar_features, vehicle_state], dim=1)
        return state
    
    def compute_reward(self, state, action, next_state, info):
        """
        Функция награды для автономного вождения
        """
        reward = 0
        
        # Награда за скорость (если в пределах лимита)
        speed = info['speed']
        speed_limit = info['speed_limit']
        if speed <= speed_limit:
            reward += 0.1 * speed / speed_limit
        
        # Штраф за отклонение от центра полосы
        lane_deviation = info['lane_deviation']
        reward -= 10 * lane_deviation**2
        
        # Штраф за близость к другим автомобилям
        min_distance = info['min_distance_to_car']
        if min_distance < 5.0:
            reward -= 100 / (min_distance + 0.1)
        
        # Большой штраф за столкновение
        if info['collision']:
            reward -= 1000
        
        # Награда за достижение цели
        if info['goal_reached']:
            reward += 1000
        
        return reward
```

#### 4. Hierarchical RL для Автономного Вождения

**Двухуровневая архитектура:**

1. **Высокоуровневая политика**: Планирование маршрута, выбор маневра
2. **Низкоуровневая политика**: Управление рулем, газом, тормозом

```python
class HierarchicalDrivingAgent:
    def __init__(self):
        # High-level: выбор маневра (lane_keep, lane_change_left, etc.)
        self.high_level_policy = PPO(state_dim=512, action_dim=5)
        
        # Low-level: управление (steering, throttle, brake)
        self.low_level_policy = SAC(state_dim=512+5, action_dim=3)
    
    def select_action(self, state):
        # Высокоуровневое действие (маневр)
        maneuver = self.high_level_policy.select_action(state)
        
        # Низкоуровневое действие (управление) с учетом маневра
        extended_state = torch.cat([state, maneuver], dim=1)
        control = self.low_level_policy.select_action(extended_state)
        
        return maneuver, control
```

#### 5. Multi-Agent RL для Трафика

**Идея:** Обучение множества агентов одновременно:

```python
class MultiAgentTraffic:
    """
    Множественные агенты для моделирования трафика
    """
    def __init__(self, num_agents=10):
        self.agents = [PPO(state_dim=256, action_dim=3) for _ in range(num_agents)]
        self.shared_replay_buffer = ReplayBuffer()
    
    def train(self, env):
        """
        Обучение множества агентов
        """
        states = env.reset()  # [num_agents, state_dim]
        
        for step in range(max_steps):
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(states[i])
                actions.append(action)
            
            next_states, rewards, dones, info = env.step(actions)
            
            # Сохранить опыт всех агентов
            for i in range(len(self.agents)):
                self.shared_replay_buffer.push(
                    states[i], actions[i], rewards[i], 
                    next_states[i], dones[i]
                )
            
            # Обучение (можно использовать общий или индивидуальный буфер)
            if len(self.shared_replay_buffer) > batch_size:
                batch = self.shared_replay_buffer.sample(batch_size)
                for agent in self.agents:
                    agent.update(batch)
            
            states = next_states
```

### Современные Подходы (2022-2026)

**1. Transformer-based RL:**
- **Decision Transformer**: Последовательное принятие решений
- **Trajectory Transformer**: Генерация траекторий

**2. World Models:**
- Обучение модели мира для планирования
- **Dreamer, DreamerV2, DreamerV3**

**3. Safety-Critical RL:**
- **Constrained RL**: Ограничения на безопасность
- **Safe RL**: Гарантии безопасности

**4. Sim-to-Real для Автомобилей:**
- **CARLA Simulator**: Реалистичная симуляция
- **Domain Adaptation**: Адаптация к реальным условиям

---

## Практические Реализации

### Использование Стандартных Библиотек

**1. Stable-Baselines3:**

```python
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

# Создание окружения
env = make_vec_env('Pendulum-v1', n_envs=4)

# Обучение PPO
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Использование
obs = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

**2. Ray RLlib:**

```python
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC

# Конфигурация
config = {
    "env": "CartPole-v1",
    "framework": "torch",
    "num_workers": 4,
}

# Обучение
algo = PPO(config=config)
for i in range(10):
    result = algo.train()
    print(f"Iteration {i}, reward: {result['episode_reward_mean']}")
```

### Полный Пример: Управление Роботом

```python
import gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# Создание окружения (пример: MuJoCo)
env = gym.make('HalfCheetah-v4')

# Создание агента
agent = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=1000000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    verbose=1
)

# Обучение
agent.learn(total_timesteps=1000000, log_interval=10)

# Сохранение
agent.save("sac_halfcheetah")

# Загрузка и использование
agent = SAC.load("sac_halfcheetah")
obs = env.reset()
for _ in range(1000):
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

---

## Сравнение Методов

### Таблица Сравнения

| Метод | Тип | Действия | Sample Efficiency | Стабильность | Применение |
|-------|-----|---------|-------------------|--------------|------------|
| **DQN** | Value-based | Дискретные | Средняя | Средняя | Игры, простые задачи |
| **PPO** | Policy-based | Дискретные/Непрерывные | Средняя | Высокая | Робототехника, игры |
| **SAC** | Actor-Critic | Непрерывные | Высокая | Высокая | Робототехника, автономные системы |
| **TD3** | Actor-Critic | Непрерывные | Высокая | Высокая | Робототехника |
| **A3C** | Actor-Critic | Дискретные/Непрерывные | Низкая | Средняя | Параллельное обучение |

### Когда Использовать Какой Метод?

**Для Робототехники:**
- **SAC** — лучший выбор для большинства задач
- **PPO** — если нужна простота и стабильность
- **TD3** — для задач с точным управлением

**Для Автономных Автомобилей:**
- **Imitation Learning** — начальная точка
- **PPO/SAC** — для улучшения политики
- **Hierarchical RL** — для сложных сценариев

**Для Игр:**
- **DQN/Rainbow** — для дискретных действий
- **PPO** — для непрерывных действий

---

## Текущее Состояние и Тренды (2024-2026)

### Основные Тренды

**1. Foundation Models для RL:**
- **Gato** (DeepMind): Универсальный агент
- **RT-1, RT-2**: Transformer-based роботы
- **PaLM-E**: Мультимодальные модели

**2. Diffusion Policies:**
- Использование diffusion models для генерации действий
- Более плавные и естественные движения

**3. Large Language Models + RL:**
- Использование LLM для планирования
- **ReAct**: Reasoning + Acting

**4. Offline RL:**
- Обучение на фиксированных датасетах
- **CQL, IQL, TD3+BC**

**5. Multi-Agent RL:**
- Обучение множества агентов
- Применение в трафике, роях роботов

### State-of-the-Art Методы (2024-2026)

**Для Робототехники:**
1. **SAC** — все еще актуален
2. **Diffusion Policies** — новый тренд
3. **RT-2** — foundation model подход

**Для Автономных Автомобилей:**
1. **End-to-End RL** — с улучшенной безопасностью
2. **Hierarchical RL** — для сложных сценариев
3. **World Models** — для планирования

### Будущие Направления

1. **Безопасность**: Гарантии безопасности для критических систем
2. **Эффективность**: Уменьшение количества данных для обучения
3. **Обобщаемость**: Работа в новых, невиданных средах
4. **Объяснимость**: Понимание решений агентов
5. **Мультимодальность**: Интеграция различных типов сенсоров

---

## References

### Основные Работы

1. **Mnih, V., et al.** (2013). "Playing Atari with Deep Reinforcement Learning." *arXiv preprint arXiv:1312.5602*.

2. **Mnih, V., et al.** (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

3. **Schulman, J., et al.** (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.

4. **Haarnoja, T., et al.** (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML*.

5. **Fujimoto, S., et al.** (2018). "Addressing Function Approximation Error in Actor-Critic Methods." *ICML*.

### Робототехника

6. **Levine, S., et al.** (2016). "End-to-end training of deep visuomotor policies." *JMLR*.

7. **Tobin, J., et al.** (2017). "Domain randomization for transferring deep neural networks from simulation to the real world." *IROS*.

8. **Brohan, A., et al.** (2022). "RT-1: Robotics Transformer for Real-World Control at Scale." *arXiv preprint arXiv:2212.06817*.

9. **Brohan, A., et al.** (2023). "RT-2: Vision-Language-Action Models Transfer to Real World." *arXiv preprint arXiv:2307.15818*.

### Автономные Автомобили

10. **Bojarski, M., et al.** (2016). "End to End Learning for Self-Driving Cars." *arXiv preprint arXiv:1604.07316*.

11. **Codevilla, F., et al.** (2018). "End-to-end driving via conditional imitation learning." *ICRA*.

12. **Dosovitskiy, A., et al.** (2017). "CARLA: An Open Urban Driving Simulator." *CoRL*.

### Современные Методы (2022-2026)

13. **Chen, L., et al.** (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling." *NeurIPS*.

14. **Hafner, D., et al.** (2023). "Mastering Diverse Domains through World Models." *arXiv preprint arXiv:2301.04104*.

15. **Chi, C., et al.** (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS*.

### Связанные Темы

- **[Unscented Kalman Filter](./unscented-kalman-filter-and-tracking.md)**: Фильтрация и отслеживание для роботов и автомобилей
- **[Variational Autoencoders](./variational-autoencoders-vaes.md)**: Генеративные модели для представления состояний
- **[Diffusion Models](./diffusion-models.md)**: Diffusion policies для генерации действий

---

## Заключение

**Deep Reinforcement Learning** представляет собой мощный подход к обучению агентов для сложных задач управления, особенно в робототехнике и автономных автомобилях.

**Ключевые Выводы:**

1. **Для робототехники**: **SAC** и **PPO** — лучшие методы для большинства задач
2. **Для автономных автомобилей**: Комбинация **Imitation Learning** и **RL** показывает лучшие результаты
3. **Sim-to-Real**: Критически важно для практического применения
4. **Безопасность**: Должна быть приоритетом при разработке систем управления

**Рекомендации:**

- Начните с **PPO** или **SAC** для большинства задач
- Используйте **симуляции** для начального обучения
- Применяйте **Domain Randomization** для лучшего переноса
- Рассмотрите **Imitation Learning** как отправную точку
- Интегрируйте **безопасность** с самого начала разработки

**Будущее Deep RL** лежит в направлении более эффективных методов, лучшего переноса из симуляции в реальность, и интеграции с foundation models для создания универсальных агентов.
