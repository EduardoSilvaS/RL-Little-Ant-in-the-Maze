from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import random

import arcade
from pyglet.window import key as pygkey

# ### ALTERAÇÃO RL ###
# Mantive o caminho original, mas certifique-se de que ele aponta para sua pasta de assets.
# Dependendo de onde você executa o script, pode precisar de ajuste.
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "images"

# Tile indices
WALL = 1
FLOOR = 0
EXIT = 2


@dataclass
class GridConfig:
    cols: int = 31
    rows: int = 21
    cell_size: int = 32


class MazeGenerator:
    """Depth-first backtracker maze generator on odd-sized grid.

    Grid uses odd dimensions so that walls occupy even indices and cells odd indices.
    """

    def __init__(self, cols: int, rows: int, rng_seed: Optional[int] = None):
        # enforce odd sizes for traditional maze carving
        self.cols = cols if cols % 2 == 1 else cols - 1
        self.rows = rows if rows % 2 == 1 else rows - 1
        self.rng = random.Random(rng_seed)

    def generate(self, start: Tuple[int, int] = (1, 1)) -> List[List[int]]:
        c, r = self.cols, self.rows
        grid = [[WALL for _ in range(c)] for _ in range(r)]

        # Helper to carve a cell to floor
        def carve(x: int, y: int):
            grid[y][x] = FLOOR

        # Directions: (dx, dy), two-step to jump to next cell; between is the wall to remove
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        sx, sy = start
        sx = max(1, min(self.cols - 2, sx))
        sy = max(1, min(self.rows - 2, sy))
        if sx % 2 == 0:
            sx -= 1
        if sy % 2 == 0:
            sy -= 1

        stack: List[Tuple[int, int]] = [(sx, sy)]
        carve(sx, sy)

        while stack:
            x, y = stack[-1]
            self.rng.shuffle(dirs)
            carved = False
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                wx, wy = x + dx // 2, y + dy // 2
                if 1 <= nx < c - 1 and 1 <= ny < r - 1 and grid[ny][nx] == WALL:
                    # carve wall and next cell
                    grid[wy][wx] = FLOOR
                    carve(nx, ny)
                    stack.append((nx, ny))
                    carved = True
                    break
            if not carved:
                stack.pop()
        return grid


class MazeWindow(arcade.Window):
    ### ALTERAÇÃO RL ###
    # O __init__ agora aceita o ambiente (env) e o agente (agent)
    def __init__(self, cfg: GridConfig, start: Tuple[int, int], exit_pos: Optional[Tuple[int, int]], rng_seed: Optional[int],
                 random_exit: bool = True, min_actor_exit_distance: int = 8,
                 env=None, agent=None, train_mode: bool = True):
        super().__init__(width=cfg.cols * cfg.cell_size, height=cfg.rows * cfg.cell_size,
                         title="Ant Learning the Maze", resizable=False)
        arcade.set_background_color(arcade.color.BLACK)

        self.cfg = cfg
        self.rng = random.Random(rng_seed)
        
        ### ALTERAÇÃO RL ###
        # Armazena o ambiente, o agente e o modo de treino
        self.env = env
        self.agent = agent
        self.train_mode = train_mode
        self.total_reward = 0.0

        # Se um ambiente de RL for fornecido, usa o grid dele.
        # Caso contrário, gera um novo labirinto como no código original.
        if self.env:
            self.grid = self.env.grid
            self.exit_pos = self.env.exit_pos
        else:
            self.start = start
            self.exit_pos: Tuple[int, int]
            # Generate maze
            gen = MazeGenerator(cfg.cols, cfg.rows, rng_seed=rng_seed)
            self.grid = gen.generate(start=start)
            # Build list of floor cells
            floor_cells = [(x, y) for y in range(cfg.rows) for x in range(cfg.cols) if self.grid[y][x] == FLOOR]
            # Helper to pick a random border floor cell
            def random_border_floor() -> Tuple[int, int]:
                border = [(x, y) for (x, y) in floor_cells if x in (1, cfg.cols - 2) or y in (1, cfg.rows - 2)]
                pool = border if border else floor_cells
                return self.rng.choice(pool)
            # Decide exit position
            if random_exit or not exit_pos:
                ex, ey = random_border_floor()
            else:
                ex, ey = exit_pos
                ex = max(1, min(cfg.cols - 2, ex))
                ey = max(1, min(cfg.rows - 2, ey))
                if self.grid[ey][ex] == WALL: self.grid[ey][ex] = FLOOR
            # Set exit tile
            self.grid[ey][ex] = EXIT
            self.exit_pos = (ex, ey)

        # Load textures
        self.tex_floor = arcade.load_texture(str(ASSETS / "floor.png"))
        self.tex_wall = arcade.load_texture(str(ASSETS / "wall.png"))
        self.tex_exit = arcade.load_texture(str(ASSETS / "exit.png"))

        # Build sprite lists for tiles
        cs = self.cfg.cell_size
        def scale_for(tex: arcade.Texture) -> float:
            return cs / max(1, tex.width)

        self.floor_sprites = arcade.SpriteList()
        self.wall_sprites = arcade.SpriteList()
        self.exit_sprites = arcade.SpriteList()
        self.actor_sprites = arcade.SpriteList()

        s_floor, s_wall, s_exit = scale_for(self.tex_floor), scale_for(self.tex_wall), scale_for(self.tex_exit)

        for y in range(self.cfg.rows):
            for x in range(self.cfg.cols):
                tile = self.grid[y][x]
                center_x, center_y = x * cs + cs / 2, y * cs + cs / 2

                # Crie um sprite vazio primeiro
                spr = arcade.Sprite()
                # Defina as propriedades comuns
                spr.center_x = center_x
                spr.center_y = center_y

                # Agora, defina a textura e a escala com base no tipo de ladrilho
                if tile == WALL:
                    spr.texture = self.tex_wall
                    spr.scale = s_wall
                    self.wall_sprites.append(spr)
                elif tile == FLOOR:
                    spr.texture = self.tex_floor
                    spr.scale = s_floor
                    self.floor_sprites.append(spr)
                elif tile == EXIT:
                    spr.texture = self.tex_exit
                    spr.scale = s_exit
                    self.exit_sprites.append(spr)

        # Place actor sprite
        if self.env:
            ax, ay = self.env.agent_pos
        else: # Lógica original se não houver ambiente de RL
            floor_cells = [(x, y) for y in range(cfg.rows) for x in range(cfg.cols) if self.grid[y][x] == FLOOR]
            ex, ey = self.exit_pos
            candidates = [(x, y) for (x, y) in floor_cells if abs(x - ex) + abs(y - ey) >= max(0, min_actor_exit_distance)]
            d = min_actor_exit_distance
            while not candidates and d > 0:
                d -= 1
                candidates = [(x, y) for (x, y) in floor_cells if abs(x - ex) + abs(y - ey) >= d]
            if not candidates: candidates = floor_cells
            ax, ay = self.rng.choice(candidates)

        # Place actor sprite
        if self.env:
            ax, ay = self.env.agent_pos
        else: # Lógica original se não houver ambiente de RL
            floor_cells = [(x, y) for y in range(cfg.rows) for x in range(cfg.cols) if self.grid[y][x] == FLOOR]
            ex, ey = self.exit_pos
            candidates = [(x, y) for (x, y) in floor_cells if abs(x - ex) + abs(y - ey) >= max(0, min_actor_exit_distance)]
            d = min_actor_exit_distance
            while not candidates and d > 0:
                d -= 1
                candidates = [(x, y) for (x, y) in floor_cells if abs(x - ex) + abs(y - ey) >= d]
            if not candidates: candidates = floor_cells
            ax, ay = self.rng.choice(candidates)
        
        self.actor_xy: Tuple[int, int] = (ax, ay)
                
        # Crie um sprite vazio para o ator
        actor = arcade.Sprite()
        
        # Carregue e defina a textura separadamente
        actor.texture = arcade.load_texture(str(ASSETS / "actor.png"))
        
        # Defina a posição e a escala
        cs = self.cfg.cell_size
        actor.center_x = ax * cs + cs / 2
        actor.center_y = ay * cs + cs / 2
        actor.scale = cs / max(1, actor.texture.width)
        
        self.actor_sprites.append(actor)
    def on_draw(self):
        self.clear()
        self.floor_sprites.draw()
        self.wall_sprites.draw()
        self.exit_sprites.draw()
        self.actor_sprites.draw()

    def _can_walk(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.cfg.cols and 0 <= gy < self.cfg.rows and self.grid[gy][gx] != WALL

    def _move_actor(self, dx: int, dy: int) -> None:
        if not self.actor_sprites: return
        x, y = self.actor_xy
        nx, ny = x + dx, y + dy
        if self._can_walk(nx, ny):
            self.actor_xy = (nx, ny)
            cs = self.cfg.cell_size
            self.actor_sprites[0].center_x = nx * cs + cs / 2
            self.actor_sprites[0].center_y = ny * cs + cs / 2

    def on_key_press(self, symbol: int, modifiers: int):
        ### ALTERAÇÃO RL ###
        # Desativa o controle do teclado se um agente estiver no comando.
        if self.agent:
            return
            
        # Código original para controle manual
        if symbol in (pygkey.UP, pygkey.W): self._move_actor(0, 1)
        elif symbol in (pygkey.DOWN, pygkey.S): self._move_actor(0, -1)
        elif symbol in (pygkey.LEFT, pygkey.A): self._move_actor(-1, 0)
        elif symbol in (pygkey.RIGHT, pygkey.D): self._move_actor(1, 0)
        
    ### ALTERAÇÃO RL ###
    # Este é o novo método que executa o loop de RL a cada frame.
    def on_update(self, delta_time: float):
        """ Lógica de atualização chamada a cada frame. """
        # Se não houver agente ou ambiente, não faz nada.
        if not self.agent or not self.env:
            return
            
        # 1. Pega o estado atual do ambiente
        state = self.env._get_observation()
        
        # 2. Agente escolhe uma ação
        action = self.agent.choose_action(state)
        
        # 3. Ambiente executa a ação e retorna o resultado
        next_state, reward, done = self.env.step(action)
        self.total_reward += reward
        
        # 4. Agente aprende com a experiência (se estiver em modo de treino)
        if self.train_mode:
            self.agent.learn(state, action, reward, next_state)
            
        # 5. Atualiza a posição do sprite do ator na tela
        actor_sprite = self.actor_sprites[0]
        ax, ay = self.env.agent_pos
        cs = self.cfg.cell_size
        actor_sprite.center_x = ax * cs + cs / 2
        actor_sprite.center_y = ay * cs + cs / 2

        # 6. Se o episódio terminou, reinicia o ambiente
        if done:
            print(f"Episódio finalizado! Recompensa total: {self.total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}")
            self.total_reward = 0.0
            self.env.reset()
            # Reduz o epsilon para que o agente explore menos com o tempo
            if self.train_mode:
                self.agent.decay_epsilon()


def run_from_config(config_path: Path):
    from ..utils.config import load_json

    data = load_json(config_path)
    grid = data.get("grid", {})
    cfg = GridConfig(
        cols=int(grid.get("cols", 31)),
        rows=int(grid.get("rows", 21)),
        cell_size=int(grid.get("cell_size", 32)),
    )
    rng_seed = data.get("rng_seed")
    start = tuple(data.get("start", [1, 1]))
    exit_raw = data.get("exit")
    exit_pos = tuple(exit_raw) if isinstance(exit_raw, list) and len(exit_raw) == 2 else None
    random_exit = bool(data.get("random_exit", True))
    min_actor_exit_distance = int(data.get("min_actor_exit_distance", 8))

    window = MazeWindow(
        cfg,
        start=start,
        exit_pos=exit_pos,
        rng_seed=rng_seed,
        random_exit=random_exit,
        min_actor_exit_distance=min_actor_exit_distance,
    )
    arcade.run()