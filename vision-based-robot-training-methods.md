## Vision-Based Robot Training: Open-Source Methods and Best Practices

### Contents

1. Введение: обучение роботов с визуальным восприятием
2. Основные подходы: Imitation Learning, RL, VLA
3. Лучшие Open-Source методы (2024-2025): OpenVLA, Octo, RT-1/RT-2, AutoRT
4. Обучение для разных типов роботов: гуманоиды, четвероногие, колёсные
5. Датасеты и данные: Open X-Embodiment, RT-1 Dataset
6. Практические примеры: код и использование
7. Sim-to-Real transfer: от симуляции к реальности
8. Сравнение методов и выбор подхода
9. Связанные темы и References
10. Как объяснить это 5‑летнему ребёнку

---

### 1. Введение: обучение роботов с визуальным восприятием

**Проблема:**
- Роботы должны понимать окружающий мир через камеры
- Нужно научить их выполнять задачи в реальном мире
- Традиционное программирование не масштабируется

**Решение:**
- **End-to-end обучение** от изображений к действиям
- Использование **больших датасетов** демонстраций
- **Vision-Language-Action (VLA)** модели для гибкости

**Типы роботов:**
- **Гуманоиды**: человекоподобные роботы (Atlas, Digit, Optimus)
- **Четвероногие**: роботы-собаки (Spot, ANYmal, Unitree)
- **Колёсные**: мобильные роботы (TurtleBot, Fetch)
- **Манипуляторы**: роботы-руки (Franka, UR5, Kuka)

**Ключевые вызовы:**
- Разнообразие роботов и их возможностей
- Sim-to-Real gap (разрыв между симуляцией и реальностью)
- Безопасность обучения
- Масштабирование на множество задач

---

### 2. Основные подходы: Imitation Learning, RL, VLA

#### 2.1. Imitation Learning (Поведенческое клонирование)

**Идея:** Учиться на демонстрациях эксперта

**Преимущества:**
- ✅ Быстрое обучение
- ✅ Безопасно (offline)
- ✅ Не требует наград

**Недостатки:**
- ❌ Требует демонстраций
- ❌ Distribution shift (ошибки накапливаются)
- ❌ Нет понимания языка

**Пример:**
```python
# Behavioral Cloning
class BehavioralCloning(nn.Module):
    def __init__(self):
        self.vision_encoder = ResNet50()
        self.action_decoder = nn.Linear(2048, 7)  # 7-DoF actions
        
    def forward(self, image):
        features = self.vision_encoder(image)
        action = self.action_decoder(features)
        return action

# Обучение
for image, expert_action in demonstrations:
    predicted = model(image)
    loss = F.mse_loss(predicted, expert_action)
    loss.backward()
```

#### 2.2. Reinforcement Learning

**Идея:** Обучение через взаимодействие и награды

**Преимущества:**
- ✅ Оптимизирует награду
- ✅ Может превзойти демонстрации
- ✅ Не требует демонстраций

**Недостатки:**
- ❌ Много данных и времени
- ❌ Рисковано (exploration)
- ❌ Сложно определить награды

**Пример:**
```python
# PPO для робота
from stable_baselines3 import PPO

env = RobotEnv()  # Окружение с роботом
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

#### 2.3. Vision-Language-Action (VLA)

**Идея:** Объединение vision, language и action

**Преимущества:**
- ✅ Понимает инструкции
- ✅ Быстрая адаптация к новым задачам
- ✅ Масштабируется на множество задач

**Недостатки:**
- ❌ Требует больших моделей
- ❌ Нужны демонстрации с инструкциями

**Пример:**
```python
# OpenVLA
from openvla import OpenVLA

model = OpenVLA.from_pretrained("openvla/openvla-7b")
action = model.predict(
    image=camera_image,
    instruction="pick up the red block"
)
```

#### 2.4. Гибридные подходы

**Imitation Learning → RL:**
1. Pre-train на демонстрациях (IL)
2. Fine-tune через RL

**VLA → RL:**
1. Pre-train VLA на демонстрациях
2. Улучшить через RL

---

### 3. Лучшие Open-Source методы (2024-2025)

#### 3.1. OpenVLA

**Описание:**
- Полностью open-source VLA модель
- 7B параметров
- Обучена на 970k демонстраций из Open X-Embodiment

**Преимущества:**
- ✅ Превосходит RT-2-X (55B) на 16.5%
- ✅ Можно fine-tune на consumer GPU (24GB+)
- ✅ Поддержка множества роботов
- ✅ Активное сообщество

**Архитектура:**
- Vision: SigLIP + DinoV2 (fused)
- Language: LLaMA 2 7B
- Action: MLP decoder

**Использование:**
```python
from openvla import OpenVLA

# Загрузка модели
model = OpenVLA.from_pretrained("openvla/openvla-7b")

# Предсказание
action = model.predict(
    image=robot_camera_image,
    instruction="pick up the cup and place it on the table"
)

# Fine-tuning
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=8, lora_alpha=16, ...)
model = get_peft_model(model, lora_config)
# Обучение на новых данных...
```

**GitHub:** [openvla/openvla](https://github.com/openvla/openvla)

**Документация:** [openvla.github.io](https://openvla.github.io/)

#### 3.2. Octo

**Описание:**
- Diffusion-based роботическая политика
- Обучена на 800k демонстраций
- Два размера: Small (27M), Base (93M)

**Преимущества:**
- ✅ Плавные и естественные движения
- ✅ Гибкие task definitions (текст, goal images)
- ✅ Эффективный fine-tuning
- ✅ Меньше параметров, чем VLA

**Архитектура:**
- Vision: ViT encoder
- Language: T5 encoder
- Action: Diffusion decoder (1D UNet)

**Использование:**
```python
from octo.model.octo import OctoModel

# Загрузка модели
model = OctoModel.from_pretrained("octo-models/octo-base")

# Предсказание с текстовой инструкцией
action = model.sample_action(
    image_obs=camera_image,
    task="pick up the red block",
    num_samples=1
)

# Предсказание с goal image
goal_image = load_image("goal_state.jpg")
action = model.sample_action(
    image_obs=camera_image,
    goal_image=goal_image,
    num_samples=1
)
```

**GitHub:** [octo-models/octo](https://github.com/octo-models/octo)

**Документация:** [octo-models.github.io](https://octo-models.github.io/)

#### 3.3. RT-1 / RT-2 (Google)

**Описание:**
- Transformer-based роботические модели
- RT-1: 35M параметров, 130k демонстраций
- RT-2: использует PaLM-E (540B), co-fine-tuning

**Статус:**
- ⚠️ **Не полностью open-source** (веса частично доступны)
- ✅ Архитектура описана в статьях
- ✅ Можно воспроизвести

**RT-1 Архитектура:**
```python
class RT1(nn.Module):
    def __init__(self):
        # Vision encoder
        self.vision_encoder = EfficientNetB3()
        
        # Language encoder
        self.language_encoder = SentencePieceTokenizer()
        
        # Transformer
        self.transformer = TransformerEncoder(...)
        
        # Action head
        self.action_head = MLP(512, 7)  # 7-DoF
        
    def forward(self, images, instruction):
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(instruction)
        
        # Fusion через transformer
        fused = self.transformer(vision_features, language_features)
        
        # Action prediction
        action = self.action_head(fused)
        return action
```

**RT-2:**
- Использует pre-trained PaLM-E
- Co-fine-tuning на роботических данных
- Лучшая генерализация

**Статьи:**
- RT-1: [arXiv:2212.06817](https://arxiv.org/abs/2212.06817)
- RT-2: [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)

#### 3.4. AutoRT

**Описание:**
- Система для масштабного сбора данных
- Использует VLM для понимания сцены
- LLM для генерации задач
- Собрано 77k реальных демонстраций на 20+ роботах

**Преимущества:**
- ✅ Автоматический сбор данных
- ✅ Минимальный человеческий надзор
- ✅ Масштабируемость

**Архитектура:**
```
[VLM] → Scene Understanding → [LLM] → Task Generation → [Robot] → Data Collection
```

**Использование:**
```python
# AutoRT pipeline
from autort import AutoRT

autort = AutoRT(
    vlm_model="gpt-4v",  # Vision-language model
    llm_model="gpt-4",   # Language model для задач
    robot_controller=robot
)

# Автоматический сбор данных
autort.collect_data(
    num_episodes=1000,
    scene_description="kitchen with various objects"
)
```

**GitHub:** [auto-rt/auto-rt](https://github.com/auto-rt/auto-rt)

**Статья:** [auto-rt.github.io](https://auto-rt.github.io/)

#### 3.5. SmolVLA

**Описание:**
- Сверхкомпактная VLA модель от Hugging Face (2025)
- Всего **450M параметров** (в 15 раз меньше OpenVLA!)
- Обучена на community данных из LeRobot платформы
- Производительность сравнима с моделями в 10 раз больше

**Преимущества:**
- ✅ Очень маленький размер (450M)
- ✅ Можно обучать на одном GPU
- ✅ Работает на consumer GPU или даже CPU
- ✅ Asynchronous inference (ускорение на ~30%)
- ✅ Быстрое обучение: ~4 часа на A100 для 20k шагов
- ✅ Полностью open-source

**Архитектура:**
- Vision-Language: SmolVLM-2 (компактная VL модель)
- Action: Flow-Matching Transformer для chunked действий

**Использование:**
```python
from lerobot import SmolVLA

# Загрузка модели
model = SmolVLA.from_pretrained("lerobot/smolvla_base")

# Предсказание
action = model.predict(
    image=robot_camera_image,
    instruction="pick up the cup and place it on the table"
)

# Asynchronous inference для ускорения
action_chunk = model.predict_async(
    image=robot_camera_image,
    instruction="pick up the cup",
    chunk_size=10
)
```

**GitHub:** [lerobot/smolvla](https://github.com/huggingface/lerobot)
**Hugging Face:** [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
**Статья:** [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)

#### 3.6. Другие методы

**Diffusion Policy:**
- Использует diffusion models для генерации действий
- Плавные траектории
- GitHub: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)

**ACT (Action Chunking with Transformers):**
- Chunk-based действия
- Эффективен для манипуляции
- GitHub: [tonyzhaozh/act](https://github.com/tonyzhaozh/act)

**BC-Z (Behavioral Cloning from Zero):**
- Обучение с нуля на демонстрациях
- Простая архитектура

---

### 4. Обучение для разных типов роботов

#### 4.1. Гуманоиды (Humanoids)

**Примеры:** Atlas (Boston Dynamics), Digit (Agility), Optimus (Tesla)

**Особенности:**
- Сложная кинематика (много степеней свободы)
- Балансировка
- Двурукая манипуляция

**Подходы:**

**1. Hierarchical Control:**
```python
# Высокоуровневая политика (куда идти)
high_level_policy = VLA_Model()

# Низкоуровневая политика (как двигаться)
low_level_policy = RL_Policy()  # PPO, SAC

# Интеграция
goal = high_level_policy(image, "pick up the box")
actions = low_level_policy(current_state, goal)
```

**2. End-to-End VLA:**
```python
# Прямое обучение от изображения к действиям
model = OpenVLA.from_pretrained("openvla/openvla-7b")

# Действия: позиции суставов, балансировка
action = model.predict(
    image=stereo_cameras,
    instruction="walk to the door and open it"
)
```

**Датасеты:**
- Humanoid демонстрации (ограниченные)
- Симуляция (MuJoCo, Isaac Sim)

#### 4.2. Четвероногие (Quadrupeds)

**Примеры:** Spot (Boston Dynamics), ANYmal, Unitree Go1/A1

**Особенности:**
- Локомоция (ходьба, бег, прыжки)
- Адаптация к местности
- Стабильность

**Подходы:**

**1. Gait Learning (RL):**
```python
from stable_baselines3 import PPO

env = QuadrupedEnv()  # Симуляция или реальность
model = PPO('CnnPolicy', env)
model.learn(total_timesteps=10_000_000)

# Действия: целевые позиции ног, скорости
```

**2. Vision-Based Navigation:**
```python
# VLA для навигации
model = OpenVLA.from_pretrained("openvla/openvla-7b")

action = model.predict(
    image=front_camera,
    instruction="go to the red door"
)
# Действия: линейная скорость, угловая скорость
```

**3. Terrain Adaptation:**
```python
# Обучение адаптации к разным поверхностям
# Domain randomization в симуляции
for terrain_type in ['flat', 'rough', 'stairs', 'slope']:
    env = QuadrupedEnv(terrain=terrain_type)
    model.learn(env)
```

**Датасеты:**
- Quadruped демонстрации
- Симуляция (Isaac Sim, PyBullet)

#### 4.3. Колёсные роботы (Wheeled Robots)

**Примеры:** TurtleBot, Fetch, Mobile Manipulators

**Особенности:**
- Простая кинематика
- Навигация
- Манипуляция (если есть рука)

**Подходы:**

**1. Navigation:**
```python
# VLA для навигации
model = OpenVLA.from_pretrained("openvla/openvla-7b")

while not at_goal:
    image = robot.get_camera_image()
    action = model.predict(
        image=image,
        instruction="go to the kitchen"
    )
    # action: [linear_velocity, angular_velocity]
    robot.move(action)
```

**2. Mobile Manipulation:**
```python
# Комбинированная навигация + манипуляция
navigation_model = OpenVLA(...)  # Для движения
manipulation_model = OpenVLA(...)  # Для манипуляции

# Сначала навигация
while not near_object:
    nav_action = navigation_model(image, "go to the table")
    robot.move(nav_action)

# Затем манипуляция
manip_action = manipulation_model(image, "pick up the cup")
robot.arm.execute(manip_action)
```

**Датасеты:**
- Open X-Embodiment (много колёсных роботов)
- RT-1 Dataset

#### 4.4. Манипуляторы (Manipulators)

**Примеры:** Franka Panda, UR5, Kuka

**Особенности:**
- Точная манипуляция
- Pick and place
- Сборка

**Подходы:**

**1. VLA (рекомендуется):**
```python
model = OpenVLA.from_pretrained("openvla/openvla-7b")

action = model.predict(
    image=wrist_camera + overhead_camera,
    instruction="pick up the red block and place it in the box"
)
# action: [x, y, z, qx, qy, qz, qw, gripper]
```

**2. ACT (Action Chunking):**
```python
from act import ACT

model = ACT.from_pretrained("act-franka")
# Предсказывает chunk действий (например, 10 шагов)
action_chunk = model.predict(image, instruction)
```

**Датасеты:**
- Open X-Embodiment (много манипуляторов)
- Bridge Dataset
- RT-1 Dataset

---

### 5. Датасеты и данные: Open X-Embodiment, RT-1 Dataset

#### 5.1. Open X-Embodiment

**Описание:**
- Крупнейший open-source датасет роботических демонстраций
- 970k+ демонстраций
- 25+ различных роботов
- Множество задач и сред

**Содержимое:**
- Различные типы роботов (манипуляторы, мобильные, гуманоиды)
- Разнообразные задачи (манипуляция, навигация)
- Различные сенсоры (RGB камеры, depth, proprioception)

**Использование:**
```python
from datasets import load_dataset

# Загрузка датасета
dataset = load_dataset("open-x-embodiment/open-x-embodiment")

# Пример данных
for example in dataset['train']:
    images = example['images']  # Последовательность изображений
    actions = example['actions']  # Последовательность действий
    instruction = example['instruction']  # Текстовая инструкция
    robot_type = example['robot']  # Тип робота
    
    # Обучение модели...
```

**Ссылки:**
- [GitHub](https://github.com/google-deepmind/open_x_embodiment)
- [Hugging Face](https://huggingface.co/datasets/open-x-embodiment/open-x-embodiment)

#### 5.2. RT-1 Dataset

**Описание:**
- 130k демонстраций
- 700+ задач
- Различные объекты и сцены
- Google роботы

**Содержимое:**
- Манипуляционные задачи
- Pick and place
- Сборка
- Открытие/закрытие

**Использование:**
```python
# RT-1 датасет (частично доступен)
# Используется в обучении RT-1 и RT-2
```

#### 5.3. Bridge Dataset

**Описание:**
- 7k демонстраций
- Сложные манипуляционные задачи
- Двурукая манипуляция

**Использование:**
```python
from bridge_dataset import BridgeDataset

dataset = BridgeDataset(data_path="bridge_data/")
```

#### 5.4. Создание собственного датасета

**Сбор данных:**
```python
class DataCollector:
    def __init__(self, robot):
        self.robot = robot
        self.data = []
        
    def collect_demonstration(self, instruction):
        images = []
        actions = []
        
        # Запись демонстрации
        while not task_complete:
            image = self.robot.get_camera_image()
            action = self.robot.get_current_action()  # От оператора
            
            images.append(image)
            actions.append(action)
            
            self.robot.step()
        
        # Сохранение
        self.data.append({
            'images': images,
            'actions': actions,
            'instruction': instruction
        })
        
    def save(self, path):
        # Сохранение в формате для обучения
        pass
```

---

### 6. Практические примеры: код и использование

#### 6.1. Полный пример: обучение манипулятора с OpenVLA

```python
import torch
from openvla import OpenVLA
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

# 1. Загрузка модели
model = OpenVLA.from_pretrained("openvla/openvla-7b")

# 2. Применение LoRA для fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# 3. Датасет
class RobotDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_data(data_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image': item['image'],
            'instruction': item['instruction'],
            'action': item['action']
        }

dataset = RobotDataset("robot_data/")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 4. Обучение
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(10):
    for batch in dataloader:
        images = batch['image']
        instructions = batch['instruction']
        target_actions = batch['action']
        
        # Forward
        predicted_actions = model.predict(images, instructions)
        
        # Loss
        loss = criterion(predicted_actions, target_actions)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Сохранение
model.save_pretrained("fine_tuned_openvla/")
```

#### 6.2. Пример: использование Octo для робота

```python
from octo.model.octo import OctoModel
import numpy as np
import cv2

# Загрузка модели
model = OctoModel.from_pretrained("octo-models/octo-base")

# Инициализация робота
robot = RobotController()

# Цикл управления
while True:
    # Получение изображения
    image = robot.get_camera_image()
    
    # Предсказание действия
    instruction = "pick up the red block"
    action = model.sample_action(
        image_obs=image,
        task=instruction,
        num_samples=1,
        temperature=0.1
    )
    
    # Выполнение действия
    robot.execute_action(action)
    
    # Проверка завершения задачи
    if task_complete:
        break
```

#### 6.3. Пример: Sim-to-Real transfer

```python
# 1. Обучение в симуляции
from isaac_sim import IsaacSim

sim_env = IsaacSim(robot="franka")
model = OpenVLA.from_pretrained("openvla/openvla-7b")

# Domain randomization
for epoch in range(100):
    # Случайные параметры симуляции
    sim_env.set_lighting(random_lighting())
    sim_env.set_textures(random_textures())
    sim_env.set_friction(random_friction())
    
    # Обучение
    train_on_simulation(sim_env, model)

# 2. Адаптация к реальности
real_robot = RealRobot("franka")

# Fine-tuning на реальных данных
real_data = collect_real_data(real_robot, num_episodes=100)
fine_tune_model(model, real_data)
```

---

### 7. Sim-to-Real transfer: от симуляции к реальности

#### 7.1. Проблема Sim-to-Real Gap

**Вызовы:**
- Различия в физике (трение, упругость)
- Визуальные различия (рендеринг vs реальность)
- Сенсорные различия (шум, калибровка)
- Аппаратные различия (динамика, задержки)

#### 7.2. Методы решения

**1. Domain Randomization:**
```python
def domain_randomization():
    # Случайные параметры симуляции
    params = {
        'lighting': random.uniform(0.5, 1.5),
        'textures': random.choice(['smooth', 'rough', 'textured']),
        'friction': random.uniform(0.3, 1.5),
        'object_mass': random.uniform(0.5, 2.0),
        'camera_noise': random.uniform(0.0, 0.1),
    }
    return params
```

**2. Domain Adaptation:**
```python
# Обучение в симуляции, адаптация к реальности
# Использование adversarial training
class DomainAdapter(nn.Module):
    def __init__(self):
        self.feature_extractor = ResNet50()
        self.domain_classifier = nn.Linear(2048, 2)  # sim vs real
        
    def forward(self, image):
        features = self.feature_extractor(image)
        domain_pred = self.domain_classifier(features)
        return features, domain_pred
```

**3. System Identification:**
```python
# Идентификация параметров реального робота
# Адаптация симуляции под реальность
real_robot_params = identify_robot_parameters(real_robot)
sim_env.set_parameters(real_robot_params)
```

**4. Residual Learning:**
```python
# Обучение остаточной политики
# Компенсация ошибок симуляции
base_policy = train_in_simulation()
residual_policy = train_residual(real_robot, base_policy)

final_policy = lambda state: base_policy(state) + residual_policy(state)
```

#### 7.3. Симуляторы

**1. Isaac Sim (NVIDIA):**
- Реалистичная физика
- Хорошая визуализация
- Поддержка многих роботов

**2. MuJoCo:**
- Быстрая физика
- Популярен в RL
- Open-source

**3. PyBullet:**
- Легковесный
- Хорош для прототипирования

**4. Gazebo:**
- Интеграция с ROS
- Реалистичная физика

---

### 8. Сравнение методов и выбор подхода

#### 8.1. Сравнительная таблица

| Метод | Параметры | Данные | Скорость обучения | Генерализация | Open Source | Лучшее применение |
|-------|-----------|--------|-------------------|---------------|-------------|-------------------|
| **SmolVLA** | **450M** | LeRobot | Очень быстро | Хорошая | ✅ | **Эффективность, consumer hardware** |
| **OpenVLA** | 7B | 970k | Быстро (fine-tune) | Отличная | ✅ | Универсальные задачи |
| **Octo** | 27M-93M | 800k | Быстро | Хорошая | ✅ | Плавные движения |
| **RT-1** | 35M | 130k | Средне | Хорошая | ⚠️ | Манипуляция |
| **RT-2** | 540B | 130k | Медленно | Отличная | ❌ | Генерализация |
| **ACT** | ~10M | Зависит | Быстро | Средняя | ✅ | Chunk-based задачи |
| **Diffusion Policy** | ~50M | Зависит | Средне | Хорошая | ✅ | Плавные траектории |

#### 8.2. Выбор метода

**Используйте SmolVLA, если:**
- ✅ Очень ограниченные ресурсы (consumer GPU или CPU)
- ✅ Нужна быстрая модель для прототипирования
- ✅ Важна эффективность и скорость
- ✅ Работа с community данными LeRobot

**Используйте OpenVLA, если:**
- ✅ Нужна универсальная модель
- ✅ Есть GPU 24GB+ для fine-tuning
- ✅ Нужно понимание языка
- ✅ Много разных задач

**Используйте Octo, если:**
- ✅ Нужны плавные движения
- ✅ Ограниченные ресурсы (меньше параметров)
- ✅ Diffusion-based подход предпочтителен

**Используйте RT-1/RT-2, если:**
- ✅ Нужна максимальная производительность
- ✅ Есть доступ к Google моделям
- ✅ Много данных для fine-tuning

**Используйте ACT, если:**
- ✅ Chunk-based действия подходят
- ✅ Нужна простая архитектура
- ✅ Ограниченные ресурсы

#### 8.3. Рекомендации по типу робота

**Гуманоиды:**
- OpenVLA (hierarchical) или RT-2
- Hierarchical control (high-level + low-level)

**Четвероногие:**
- RL для локомоции (PPO, SAC)
- VLA для навигации (OpenVLA)

**Колёсные:**
- OpenVLA или Octo
- Простая интеграция

**Манипуляторы:**
- OpenVLA (рекомендуется)
- ACT для chunk-based
- Diffusion Policy для плавности

---

### 9. Связанные темы и References

#### 9.1. Связанные техники

- **Vision-Language-Action (VLA) Models**: детали в отдельном документе
- **Deep Reinforcement Learning**: RL методы для роботов
- **Imitation Learning**: поведенческое клонирование
- **Diffusion Models**: для генерации действий
- **Transformer Architecture**: основа многих моделей

#### 9.2. Связанные документы

- **[Vision-Language-Action (VLA) Models](./vision-language-action-models-vla.md)**: детальное описание VLA
- **[Deep Reinforcement Learning](./deep-reinforcement-learning.md)**: RL для роботов
- **[Transformers, Attention and Vision Transformers](./transformers-attention-and-vision-transformers-vit.md)**: архитектура Transformer
- **[Low-Rank Adaptation (LoRA)](./low-rank-adaptation-lora.md)**: эффективный fine-tuning

#### 9.3. Ключевые статьи

1. **OpenVLA: An Open-Source Vision-Language-Action Model** (2024)
   - Kim et al.
   - [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
   - [GitHub](https://github.com/openvla/openvla)

2. **Octo: An Open-Source Generalist Robot Policy** (2024)
   - Shafiullah et al.
   - [arXiv:2409.10693](https://arxiv.org/abs/2409.10693)
   - [GitHub](https://github.com/octo-models/octo)

3. **RT-1: Robotics Transformer for Real-World Control at Scale** (2022)
   - Brohan et al., Google
   - [arXiv:2212.06817](https://arxiv.org/abs/2212.06817)

4. **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control** (2023)
   - Brohan et al., Google
   - [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)

5. **AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents** (2024)
   - [auto-rt.github.io](https://auto-rt.github.io/)

6. **Open X-Embodiment: Robotic Learning Datasets and RT-X Models** (2023)
   - [arXiv:2310.08864](https://arxiv.org/abs/2310.08864)

7. **SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics** (2025)
   - Hugging Face
   - [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
   - [GitHub](https://github.com/huggingface/lerobot)
   - [Hugging Face Model](https://huggingface.co/lerobot/smolvla_base)

#### 9.4. Библиотеки и инструменты

- **OpenVLA**: [GitHub](https://github.com/openvla/openvla)
- **Octo**: [GitHub](https://github.com/octo-models/octo)
- **Stable-Baselines3**: RL библиотека
- **Isaac Sim**: симулятор NVIDIA
- **ROS (Robot Operating System)**: интеграция с роботами
- **Hugging Face**: модели и датасеты

#### 9.5. Датысеты

- **Open X-Embodiment**: [GitHub](https://github.com/google-deepmind/open_x_embodiment)
- **RT-1 Dataset**: частично доступен
- **Bridge Dataset**: сложные манипуляции

---

### 10. Как объяснить это 5‑летнему ребёнку

**Представь, что ты хочешь научить робота делать разные вещи, просто показывая ему, что делать, или говоря ему инструкции.**

**Обучение роботов с камерами** — это как научить робота видеть и понимать, что делать:

1. **Показываем примеры**: Мы показываем роботу много раз, как делать разные задачи (например, поднимать кубики, открывать двери). Робот запоминает, что делать в каждой ситуации.

2. **Говорим инструкции**: Мы можем сказать роботу "подними красный кубик", и он понимает, что нужно найти красный кубик на картинке и поднять его.

3. **Робот учится**: Робот смотрит на тысячи примеров и учится, какие действия нужно делать в разных ситуациях.

**Разные типы роботов:**
- **Робот-рука**: учится брать и перемещать предметы
- **Робот-собака**: учится ходить, бегать, следовать командам
- **Робот-человек**: учится делать сложные задачи двумя руками
- **Робот на колёсах**: учится ездить и находить вещи

**Почему это круто:**
- Не нужно программировать каждую задачу отдельно
- Можно просто сказать роботу, что делать
- Робот может научиться новым задачам быстрее

Это как научить робота быть таким же умным помощником, как твой друг, который понимает тебя и может помочь с разными делами!

---
