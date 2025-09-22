# Em src/envs/rl_maze_env.py

from dataclasses import dataclass
from typing import Tuple, List
import random

# Reutilizando o gerador de labirinto do seu arquivo original
from src.engine.maze import MazeGenerator, WALL, FLOOR, EXIT

# Definindo as Ações do Agente
# 0: Andar para Frente, 1: Virar à Esquerda (90 graus), 2: Virar à Direita (90 graus)
ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT = 0, 1, 2
ACTIONS = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT]

# Definindo as Direções (Orientação) do Agente
# 0: Cima (Y+), 1: Direita (X+), 2: Baixo (Y-), 3: Esquerda (X-)
DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = 0, 1, 2, 3

@dataclass
class RLMazeConfig:
    """Configurações para o ambiente do labirinto de RL."""
    cols: int = 31
    rows: int = 21
    rng_seed: int = None

class MazeEnv:
    """
    Uma classe que adapta o seu labirinto para ser um ambiente de RL.
    """
    def __init__(self, cfg: RLMazeConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.rng_seed)
        
        # Gera o labirinto
        gen = MazeGenerator(cfg.cols, cfg.rows, rng_seed=cfg.rng_seed)
        self.grid = gen.generate(start=(1, 1))

        self.floor_cells = [(x, y) for y in range(cfg.rows) for x in range(cfg.cols) if self.grid[y][x] == FLOOR]

        # Define a saída (açúcar)
        ex, ey = self.rng.choice(self.floor_cells)
        self.grid[ey][ex] = EXIT
        self.exit_pos = (ex, ey)
        
        # Atributos do agente
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_dir: int = DIR_UP
        
    def _get_observation(self) -> Tuple[int, int]:
        """
        Retorna o que o agente "vê".
        Estado = (direção_do_agente, tipo_de_bloco_a_frente)
        """
        # Vetores de movimento para cada direção
        dir_vectors = {DIR_UP: (0, 1), DIR_RIGHT: (1, 0), DIR_DOWN: (0, -1), DIR_LEFT: (-1, 0)}
        
        dx, dy = dir_vectors[self.agent_dir]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        tile_in_front = WALL # Assumir parede se estiver fora do grid
        if 0 <= nx < self.cfg.cols and 0 <= ny < self.cfg.rows:
            tile_in_front = self.grid[ny][nx]
            
        return (self.agent_dir, tile_in_front)

    def reset(self) -> Tuple[int, int]:
        """
        Reinicia o ambiente para um novo episódio.
        Posiciona a formiga em um local aleatório e retorna a observação inicial.
        """
        self.agent_pos = self.rng.choice(self.floor_cells)
        self.agent_dir = self.rng.choice([DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT])
        return self._get_observation()

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Executa uma ação do agente no ambiente.
        Retorna: (próxima_observação, recompensa, finalizado)
        """
        done = False
        
        # --- Lógica da Ação ---
        if action == ACTION_TURN_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == ACTION_TURN_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == ACTION_FORWARD:
            dir_vectors = {DIR_UP: (0, 1), DIR_RIGHT: (1, 0), DIR_DOWN: (0, -1), DIR_LEFT: (-1, 0)}
            dx, dy = dir_vectors[self.agent_dir]
            x, y = self.agent_pos
            nx, ny = x + dx, y + dy

            # Verifica o que está à frente
            if 0 <= nx < self.cfg.cols and 0 <= ny < self.cfg.rows:
                tile = self.grid[ny][nx]
                if tile != WALL:
                    self.agent_pos = (nx, ny) # Anda
                # Se for parede, não se move (penalidade será aplicada)
            
        # --- Lógica da Recompensa ---
        current_tile = self.grid[self.agent_pos[1]][self.agent_pos[0]]
        
        if current_tile == EXIT:
            reward = 100.0  # Grande recompensa por encontrar o açúcar!
            done = True
        else:
            # Observa o que está na nova posição para calcular a recompensa
            _, tile_in_front = self._get_observation()
            if action == ACTION_FORWARD and tile_in_front == WALL:
                 reward = -10.0 # Grande penalidade por tentar andar contra a parede
            else:
                 reward = -0.1 # Pequena penalidade por passo, para incentivar a rapidez
            
        return self._get_observation(), reward, done