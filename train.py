from stable_baselines3 import PPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from examples.base2 import make_env

env = make_env()
env = DummyVecEnv([lambda: env]) #don't really need this but useful if we have multiple envs, which we will eventually

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("saves/meta-rl")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()