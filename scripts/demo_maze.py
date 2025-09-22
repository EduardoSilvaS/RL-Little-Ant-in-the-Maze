# Em demo_maze.py (arquivo principal)

from __future__ import annotations
from pathlib import Path
import arcade

# Nossas novas classes
from src.envs.rl_maze_env import MazeEnv, RLMazeConfig, ACTIONS
from src.agents.q_learning_agent import QLearningAgent
# A classe de visualização modificada
from src.engine.maze import MazeWindow, GridConfig

if __name__ == "__main__":
    # 1. Configurar o ambiente
    rl_config = RLMazeConfig(cols=21, rows=15, rng_seed=42)
    env = MazeEnv(rl_config)

    # 2. Criar o agente
    agent = QLearningAgent(actions=list(range(len(ACTIONS))))

    # 3. Configurar a visualização
    grid_cfg = GridConfig(
        cols=rl_config.cols,
        rows=rl_config.rows,
        cell_size=32,
    )

    # 4. Criar a janela, passando o ambiente e o agente
    window = MazeWindow(
        cfg=grid_cfg,
        start=(0,0), # não será usado, pois o env controla
        exit_pos=(0,0), # não será usado
        rng_seed=None, # não será usado
        env=env,
        agent=agent,
        train_mode=True # Defina como False para ver o agente apenas executar a política aprendida
    )
    
    # Reinicia o ambiente uma vez antes de começar
    env.reset()
    
    print("Iniciando o treinamento do agente...")
    arcade.run()