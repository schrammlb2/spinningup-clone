import torch
import gym
import os



def test(actor):
  with torch.no_grad():
    env = Env(env_name)
    state, done, total_reward = env.reset(), False, 0
    # while not done:
    for i in range(1000):
      a=actor(state)
      if type(a) != torch.Tensor:
        a = a.mean
      action = torch.clamp(a, min=-1, max=1)  # Use purely exploitative policy at test time
      state, reward, done = env.step(action)
      total_reward += reward
      if done: 
        break
    return total_reward


def test_worst_case(actor, critic_1, critic_2):
  env = Env(env_name)
  state, done, total_reward = env.reset(), False, 0
  epsilon = .03
  for i in range(1000):
    a=actor(state)
    if type(a) != torch.Tensor:
      a = a.mean
      # Use purely exploitative policy at test time
    g1 = action_gradient(critic_1, state, action, epsilon=.1)
    g2 = action_gradient(critic_2, state, action, epsilon=.1)
    bad_action = action-(g1+g2)/2*epsilon
    action = torch.clamp(bad_action, min=-1, max=1)
    state, reward, done = env.step(action)
    total_reward += reward
    if done: 
      break
  return total_reward


def test_action_noise(actor):
  for i in range(4):
    sigma = .03
    with torch.no_grad():
      env = Env(env_name)
      state, done, total_reward = env.reset(), False, 0
      # while not done:
      for i in range(1000):
        s = np.random.normal(0, sigma, action_dim)
        a=actor(state) + s
        if type(a) != torch.Tensor:
          a = a.mean
        action = torch.clamp(a, min=-1, max=1)  # Use purely exploitative policy at test time
        state, reward, done = env.step(action)
        total_reward += reward
        if done: 
          break
      return total_reward

def test_transfer(actor, path):
  from randomize_xml import randomize_xml
  if path is not None:
    filenames = [os.getcwd() + '/'+ path + xml for xml in os.listdir(path) if xml.endswith('.xml') ]
    # for xml in os.listdir(path):
    fn = sorted(filenames, key = lambda x: len(x))
    randomize_xml(fn[0], count = 4, scale=scale)
    # for filename in filenames:
    reward_list = []
    for filename in fn[1:]:
      with torch.no_grad():
        env = Env(env_name, xml_file=filename)
        # env.modify(filename)
        state, done, total_reward = env.reset(), False, 0
        for i in range(1000):
          a=actor(state)
          if type(a) != torch.Tensor:
            a = a.mean
          action = torch.clamp(a, min=-1, max=1)  # Use purely exploitative policy at test time
          state, reward, done = env.step(action)
          total_reward += reward
          if done: 
            break
        reward_list.append(total_reward)

    return mean(reward_list)
  else:
    return 0
