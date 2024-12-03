try:
    import os
    import shutil
    import logging
    import ray
    from ray.rllib.algorithms.ppo import PPO
    from pettingzoo.utils.conversions import aec_to_parallel
    from ray.tune.registry import register_env
    from warehouse_env import WarehouseEnv
    import gym
    import numpy as np
    import torch
except ImportError as e:
    raise ImportError(f"Modulo mancante: {e}. Assicurati di installare tutti i pacchetti richiesti.")

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clear_logs():
    """Pulisce e ricrea le directory dei log."""
    log_dirs = ["./logs/best_model/", "./logs/eval/"]
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    logger.info("Directory dei log pulite e ricreate")

def create_env():
    """Crea e configura l'ambiente WarehouseEnv."""
    try:
        num_robots = 3
        env = WarehouseEnv(num_robots=num_robots, num_tasks=3, num_obstacles=5, grid_size=10)
        env = aec_to_parallel(env)  # Converte l'ambiente in un ambiente parallelo di PettingZoo
        return env, num_robots
    except AttributeError as e:
        logger.error(f"Errore nell'attributo dell'ambiente: {e}")
        raise
    except ImportError as e:
        logger.error(f"Errore di importazione: {e}")
        raise
    except Exception as e:
        logger.error(f"Errore generico nella creazione dell'ambiente: {e}")
        raise

def evaluate_model(trainer, env, num_episodes=5):
    """Valuta il modello su un numero specifico di episodi."""
    logger.info("Inizio della valutazione del modello...")
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = {agent: False for agent in env.agents}
        episode_reward = {agent: 0 for agent in env.agents}
        
        while not all(done.values()):
            actions = {
                agent: trainer.compute_action(obs[agent], policy_id=f"robot_{agent}")
                for agent in env.agents if not done[agent]
            }
            obs, rewards, done, _ = env.step(actions)
            for agent, reward in rewards.items():
                episode_reward[agent] += reward
        
        total_rewards.append(episode_reward)
        logger.info(f"Ricompense episodio {episode + 1}: {episode_reward}")
    
    avg_rewards = {agent: np.mean([ep[agent] for ep in total_rewards]) for agent in env.agents}
    logger.info(f"Ricompense medie: {avg_rewards}")
    return avg_rewards

def check_gpu():
    """Controlla se una GPU è disponibile e restituisce il numero di GPU."""
    if torch.cuda.is_available():
        logger.info(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
        return 1
    else:
        logger.warning("Nessuna GPU rilevata. Il training sarà effettuato su CPU.")
        return 0

def main():
    logger.info("Inizio del training...")
    clear_logs()

    env_train, num_robots = create_env()  # Scomponi correttamente il risultato
    logger.info("Ambiente creato con successo")

    # Aggiungi il test per stampare observation_space e action_space
    print(f"Observation Space: {env_train.observation_space}")
    print(f"Action Space: {env_train.action_space}")

    # Registra l'ambiente con Ray RLlib
    register_env("warehouse_env", lambda config: env_train)

    # Configurazione multi-agente
    config = {
        "env": "warehouse_env",
        "num_workers": 1,
        "num_gpus": check_gpu(),  # Controlla la disponibilità della GPU
        "framework": "torch",
        "multiagent": {
            "policies": {
                f"robot_{i}": (
                    None,
                    env_train.observation_space[f'robot_{i}'],
                    env_train.action_space[f'robot_{i}'],
                    {}
                )
                for i in range(num_robots)  # Usa il numero di robot passato
            },
            "policy_mapping_fn": lambda agent_id: f"robot_{agent_id}",  # Mappa ogni agente alla sua policy
        },
        "disable_env_checking": True
    }

    # Crea il trainer PPO
    trainer = PPO(config=config)

    # Addestra il modello
    total_timesteps = 200_000
    logger.info(f"Inizio training per {total_timesteps} timesteps...")
    for iteration in range(1, 101):  # 100 iterazioni di training
        result = trainer.train()
        logger.info(f"Iterazione {iteration}, risultato: {result}")
        
        if iteration % 10 == 0:  # Salva il modello ogni 10 iterazioni
            checkpoint_path = trainer.save(f"./logs/checkpoints/iteration_{iteration}")
            logger.info(f"Checkpoint salvato in {checkpoint_path}")

    # Salva il modello finale
    model_path = "./logs/ppo_warehouse_model"
    trainer.save(model_path)
    logger.info(f"Modello salvato in {model_path}")

    # Valuta il modello
    evaluate_model(trainer, env_train, num_episodes=5)

    # Chiudi l'ambiente
    env_train.close()
    logger.info("Ambiente chiuso")

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    try:
        main()
    except Exception as e:
        logger.error(f"Errore critico: {e}")
    finally:
        ray.shutdown()
        logger.info("Sessione Ray terminata.")

