#1.Import Dependencies
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#2.Load Environment
env = gym.make("git commit -m "first commit"-v1", render_mode="human")
#env=gym.make('CartPole-v1')


# 3.Understanding the Environment
# we are going to test our environment 5 times
episodes =5
for episode in range(1,episodes+1):
    state = env.reset() #By using env.reset(), we are going reset the environment and obtain initail set of observation (1.Agent 2.Reward 3.Action 4.Environment)
    done= False
    score=0
    while not done:
        env.render()
        action=env.action_space.sample()
        n_state,reward,done,temp,info =env.step(action) # env.step() = Apply an action to an environment
        score+=reward
    print('Episode={} Score={}'.format(episode,score))
env.close() # close down the render frame


# 3.Understanding the Environment

# print("The initial state of obervation is")
# state=env.reset()
# print(state)

# print("The action spaces are")
# print(env.action_space)
# print(env.action_space.sample()) # random value of Action

# print("The observation spaces are")
# print(env.observation_space)
# print(env.observation_space.sample()) # random value of observation


# 4.Train an RL Module

log_path = os.path.join('Training','Logs')

env = gym.make("CartPole-v1", render_mode="human")
# pendulum_env = Monitor(pendulum_env, log_dir, allow_early_resets=True)
env=DummyVecEnv([lambda:env])
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=10000)

# 5.Save and Reload Model

PPO_Path =os.path.join('Training','Saved Models','PPO_Model_cartpole')
model.save(PPO_Path)


del model
model=PPO.load(PPO_Path,env=env)
model.learn(total_timesteps=1000)

# 6.Evalution

evaluate_policy(model,env,n_eval_episodes=10)


# 7.Test Model
episodes =5
for episode in range(1,episodes+1):
    obs = env.reset() #By using env.reset(), we are going reset the environment and obtain initail set of observation (1.Agent 2.Reward 3.Action 4.Environment)
    done= False
    score=0
    while not done:
        #env.render()
        action,_=model.predict(obs)
        obs,reward,done,info =env.step(action) # env.step() = Apply an action to an environment
        score+=reward
    print('Episode={} Score={}'.format(episode,score))
env.close() # close down the render frame


#7 . Adding a callback to the training Stage

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

stop_callback=StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
save_path=os.path.join('Training','Saved Models','New Models')

eval_callback=EvalCallback(eval_env=env,callback_on_new_best=stop_callback,eval_freq=10000,best_model_save_path=save_path,verbose=1)

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000,callback=eval_callback)

