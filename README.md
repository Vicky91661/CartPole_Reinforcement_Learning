# 1.Insatlling all the required libraries and Import Dependencies
    >>pip install gym
    >>pip install pygame
    >>pip install stable-baselines3

# 2.Load Environment
    First reset the environment
    By using env.reset(), we are going reset the environment and obtain initail set of observation (1.Agent 2.Reward 3.Action 4.Environment)
    ## Actio Space
    The action spaces are=env.action_space => there are 2 action space 0 and 1
    0 = cart will go to left
    1 = cart will go to right

    ## Observation space 
    The observation space is a `ndarray` with shape `(4,)` 
    0 = Cart Position
    1 = Cart velocity
    2 = Pole Angle
    3 = Pole Angular Velocity


# 3.Understanding the Environment
# 4.Train an RL Module
# 5.Save and Reload Model
# 6.Evalution
# 7.Test Model
# 8.Viewing Logs in Tensorboard
# 9.Adding a callback to the training Stage
# 10.Changing Policies
# 11.Using an Alternate Algorithm
