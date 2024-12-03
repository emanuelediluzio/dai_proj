from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import numpy as np
import logging
import heapq
import os
from datetime import datetime
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Crea la directory dei log se non esiste
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configura il logger principale
log_filename = f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Aggiunge output anche alla console
    ]
)

logger = logging.getLogger(__name__)

ENABLE_DEBUG = False

if ENABLE_DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

ENABLE_ROBOT_LOGS = True
ENABLE_TASK_LOGS = False

def heuristic(a, b):
    """Calcola la distanza Manhattan tra due punti."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, obstacles, grid_size):
    """Algoritmo A* semplificato per una griglia discreta."""
    open_set = []
    heapq.heappush(open_set, (0, start, [start]))
    closed_set = set()

    while open_set:
        cost, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current in closed_set:
            continue

        closed_set.add(current)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                if neighbor not in obstacles and neighbor not in closed_set:
                    new_path = path + [neighbor]
                    heapq.heappush(open_set, (cost + 1, neighbor, new_path))

    logger.warning("Percorso non trovato.")
    return []

class WarehouseEnv(MultiAgentEnv):
    """
    Ambiente personalizzato per la gestione di un magazzino con robot, task e ostacoli.
    """

    metadata = {
        "render_modes": ["human"],
        "is_parallelizable": True
    }

    def __init__(self, num_robots=1, num_tasks=3, num_obstacles=5, grid_size=10):
        super().__init__()

        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.num_obstacles = num_obstacles
        self.grid_size = grid_size
        self.max_speed = 0.5

        self.observation_size = (
            self.num_robots * 2 +
            self.num_tasks * 2 +
            self.num_tasks * 2 +
            self.num_obstacles * 2 +
            self.num_tasks +
            self.num_tasks +
            self.num_tasks +
            self.num_robots +
            self.num_robots
        )

        # Definizione di observation_space come dizionario
        self.observation_space = {
            f"robot_{i}": spaces.Box(low=0.0, high=1.0, shape=(self.observation_size,), dtype=np.float32)
            for i in range(self.num_robots)
        }

        # Definizione di action_space come dizionario
        self.action_space = {
            f"robot_{i}": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for i in range(self.num_robots)
        }

        self.state = {}
        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Resetta l'ambiente con posizioni iniziali sicure."""
        super().reset(seed=seed)

        # Configura lo stato iniziale
        self.state['robots'] = np.array([[5.0, 5.0] for _ in range(self.num_robots)], dtype=np.float32)
        self.state['obstacles'] = np.random.rand(self.num_obstacles, 2) * self.grid_size
        self.state['tasks'] = np.random.rand(self.num_tasks, 2) * self.grid_size
        self.state['dropoff_zones'] = np.random.rand(self.num_tasks, 2) * self.grid_size
        self.state['task_priorities'] = np.random.randint(1, 5, size=self.num_tasks).astype(np.float32)
        self.state['tasks_picked'] = [False] * self.num_tasks
        self.state['delivered_tasks'] = [False] * self.num_tasks
        self.state['assigned_tasks'] = [None] * self.num_robots
        self.state['robot_batteries'] = np.full(self.num_robots, 100.0)
        self.state['carrying_package'] = [False] * self.num_robots

        return self._get_observation(), {}

    def _get_observation(self):
        """Restituisce l'osservazione normalizzata come NumPy array."""
        obs = np.concatenate([
            self.state['robots'].flatten() / self.grid_size,
            self.state['tasks'].flatten() / self.grid_size,
            self.state['dropoff_zones'].flatten() / self.grid_size,
            self.state['obstacles'].flatten() / self.grid_size,
            self.state['task_priorities'] / np.max(self.state['task_priorities']),
            np.array(self.state['tasks_picked'], dtype=np.float32),
            np.array(self.state['delivered_tasks'], dtype=np.float32),
            np.array(self.state['carrying_package'], dtype=np.float32),
            self.state['robot_batteries'] / 100.0
        ])

        if obs.shape[0] != self.observation_size:
            raise ValueError(f"Dimensione dell'osservazione non corrisponde: atteso {self.observation_size}, ottenuto {obs.shape[0]}")

        return obs.astype(np.float32)

    def step(self, action):
        """Esegue un passo nell'ambiente."""
        if len(action) != self.num_robots * 2:
            raise ValueError(f"Lunghezza dell'array di azioni non valida. Atteso: {self.num_robots * 2}, Ottenuto: {len(action)}")

        for i in range(self.num_robots):
            action_i = action[i * 2:(i + 1) * 2]
            # Controlla che l'azione sia nei limiti
            action_i = np.clip(action_i, -1.0, 1.0)  # Limita le azioni tra -1 e 1
            movement = action_i * self.max_speed
            current_pos = self.state['robots'][i].copy()
            new_pos = current_pos + movement

            # Controlla le collisioni
            if not self._check_collision(new_pos):
                self.state['robots'][i] = new_pos

        self._update_tasks()
        self._move_obstacles()
        reward = self._calculate_reward()
        done = all(self.state['delivered_tasks'])
        return self._get_observation(), reward, done, {}

    def _check_collision(self, pos):
        """Controlla se la posizione è valida rispetto agli ostacoli e alla griglia."""
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return True  # Fuori dai limiti
        for obs in self.state['obstacles']:
            if np.linalg.norm(pos - obs) < 0.5:  # Soglia per considerare collisione
                return True
        return False

    def _update_tasks(self):
        """Aggiorna lo stato dei task (raccolta e consegna)."""
        PICKUP_THRESHOLD = 1.5
        DROPOFF_THRESHOLD = 1.5

        for i in range(self.num_robots):
            assigned_task = self.state['assigned_tasks'][i]
            if assigned_task is not None:
                robot_pos = self.state['robots'][i]

                if not self.state['tasks_picked'][assigned_task]:
                    task_pos = self.state['tasks'][assigned_task]
                    distance = np.linalg.norm(robot_pos - task_pos)

                    if distance < PICKUP_THRESHOLD:
                        self.state['tasks_picked'][assigned_task] = True
                        self.assign_tasks()

                else:
                    dropoff_pos = self.state['dropoff_zones'][assigned_task]
                    distance = np.linalg.norm(robot_pos - dropoff_pos)

                    if distance < DROPOFF_THRESHOLD:
                        self.state['delivered_tasks'][assigned_task] = True
                        self.state['assigned_tasks'][i] = None
                        self.state['tasks_picked'][assigned_task] = False
                        self.state['carrying_package'][i] = False
                        self.assign_tasks()

    def _move_obstacles(self):
        """Muove gli ostacoli in modo dinamico."""
        max_speed = 0.1  # Velocità massima per il movimento degli ostacoli
        for i in range(self.num_obstacles):
            direction = np.random.uniform(-1, 1, size=2)
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction = direction / norm  # Normalizza la direzione
            movement = direction * max_speed  # Limita la velocità
            new_pos = self.state['obstacles'][i] + movement

            # Controlla se la nuova posizione è valida
            if not self._check_collision(new_pos):
                self.state['obstacles'][i] = np.clip(new_pos, 0.0, self.grid_size)  # Limita la posizione agli estremi della griglia

    def _calculate_reward(self):
        """Calcola il reward per i robot."""
        reward = 0.0
        for i in range(self.num_robots):
            if self.state['assigned_tasks'][i] is not None:
                task_idx = self.state['assigned_tasks'][i]
                current_pos = self.state['robots'][i]
                if self.state['tasks_picked'][task_idx]:
                    target = self.state['dropoff_zones'][task_idx]
                else:
                    target = self.state['tasks'][task_idx]

                distance = np.linalg.norm(current_pos - target)
                reward -= distance * 0.1  # Penalità per distanza

                if self.state['delivered_tasks'][task_idx]:
                    reward += 100.0  # Bonus per task consegnato
                else:
                    reward -= 10.0  # Penalità per task non completato

            if self._check_collision(self.state['robots'][i]):
                reward -= 50.0  # Penalità per collisioni

        return reward

    def render(self, mode='human'):
        """Visualizza lo stato attuale dell'ambiente."""
        print(f"--- Stato Ambiente ---")
        print(f"Robots: {self.state['robots']}")
        print(f"Tasks: {self.state['tasks']}")
        print(f"Obstacles: {self.state['obstacles']}")
        print(f"Batteries: {self.state['robot_batteries']}")
        print(f"Dropoff Zones: {self.state['dropoff_zones']}")
        print(f"-----------------------")

    def close(self):
        """Chiude l'ambiente."""
        pass

    def assign_tasks(self):
        """Assegna i task ai robot disponibili."""
        for i in range(self.num_robots):
            if self.state['assigned_tasks'][i] is None:  # Se il robot è libero
                available_tasks = [
                    idx for idx, picked in enumerate(self.state['tasks_picked']) 
                    if not picked and not self.state['delivered_tasks'][idx]
                ]

                if available_tasks:
                    # Assegna il task più vicino
                    robot_pos = self.state['robots'][i]
                    assigned_task = min(
                        available_tasks,
                        key=lambda t: np.linalg.norm(robot_pos - self.state['tasks'][t])
                    )
                    self.state['assigned_tasks'][i] = assigned_task
                    self.state['carrying_package'][i] = False  # Indica che il robot non sta trasportando un pacco
                    logger.info(f"Robot {i} assegnato al task {assigned_task}.")

    def update_environment(self):
        """Aggiorna l'ambiente e gestisce la generazione di nuovi task."""
        for i, obstacle in enumerate(self.state["obstacles"]):
            delta = np.random.uniform(-0.1, 0.1, size=2)
            new_position = obstacle + delta
            new_position = np.clip(new_position, 0, self.grid_size)
            self.state["obstacles"][i] = new_position

        if all(self.state["delivered_tasks"]):
            self.done = True
            print("Tutti i task sono stati completati!")

        for i, battery in enumerate(self.state["robot_batteries"]):
            self.state["robot_batteries"][i] = max(0, battery - 0.1)
            if self.state["robot_batteries"][i] == 0:
                self.state["carrying_package"][i] = False
                print(f"Robot {i} ha esaurito la batteria.")

        self.state["task_priorities"] = np.maximum(0, self.state["task_priorities"] - 0.01)

        if np.random.rand() < 0.05:
            new_task = np.random.uniform(0, self.grid_size, size=2)
            self.state["tasks"] = np.vstack([self.state["tasks"], new_task])
            self.state["task_priorities"] = np.append(self.state["task_priorities"], 1.0)
            self.state["tasks_picked"].append(False)
            self.state["delivered_tasks"].append(False)
            print(f"Nuovo task generato: {new_task}")

    def move_robot(self, robot_id, action):
        """Muove il robot in base all'azione fornita."""
        current_position = self.state['robots'][robot_id]
        new_position = current_position + action * self.max_speed

        if (0 <= new_position[0] <= self.grid_size[0] - self.robot_radius) and \
           (0 <= new_position[1] <= self.grid_size[1] - self.robot_radius):
            self.state['robots'][robot_id] = new_position
        else:
            self.collisions[robot_id] = True

    def compute_reward(self):
        """Calcola il reward per i robot."""
        reward = 0.0
        for robot_id, position in enumerate(self.state['robots']):
            if self.carrying_package[robot_id]:
                dropoff_position = self.dropoff_zones[self.assigned_tasks[robot_id]]
                distance = np.linalg.norm(position - dropoff_position)
                reward += max(0, 10 - distance)
            else:
                task_position = self.tasks[self.assigned_tasks[robot_id]]
                distance = np.linalg.norm(position - task_position)
                reward += max(0, 10 - distance)

        reward -= sum(50.0 for collision in self.collisions if collision)

        return reward

    def generate_new_task(self):
        """Genera un nuovo task se il numero massimo non è stato raggiunto."""
        if len(self.state["tasks"]) >= self.max_total_tasks:
            logger.info("Numero massimo di task raggiunto. Nuovi task non verranno generati.")
            return

        new_task = np.random.uniform(0, self.grid_size, size=2)
        new_dropoff = np.random.uniform(0, self.grid_size, size=2)
        new_priority = np.random.randint(1, 5)

        self.state["tasks"] = np.vstack([self.state["tasks"], new_task])
        self.state["dropoff_zones"] = np.vstack([self.state["dropoff_zones"], new_dropoff])
        self.state["task_priorities"] = np.append(self.state["task_priorities"], new_priority)
        self.state["tasks_picked"].append(False)
        self.state["delivered_tasks"].append(False)

        self._update_observation_space()

        logger.info(f"Nuovo task generato: posizione={new_task}, dropoff={new_dropoff}, priorità={new_priority}")



