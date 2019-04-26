
import atari_cgp as cgp
import gym
import pickle
import sys


POP_SIZE = cgp.MU + cgp.LAMBDA
RNG_SEED = 1
steps_limit = 300
NULL_ACTION = 0



def game():

  gen_scores = []
  pop = cgp.create_population(POP_SIZE)
  env = gym.make("LunarLander-v2")
  env.seed(RNG_SEED)
  env.reset()
  
  R = [0]*POP_SIZE
  
  for gen in range(cgp.N_GEN):
  
    for p in range(len(pop)):
      
      R[p] = run_episode(env, pop[p])
  
    # print("Generation {0}: ".format(gen), R)
    gen_scores.append(sum(R)/len(R))

    max_score = max(R)
    print("Generation {0}: ".format(gen), max_score)
    print("")
    pb = cgp.MUT_PB

    if max_score < -200.0:
      pb = cgp.MUT_PB * 3
    elif max_score < -100:
      pb = cgp.MUT_PB * 2
    elif max_score < 0:
      pb = cgp.MUT_PB * 1.5
    elif max_score < 50:
      pb = cgp.MUT_PB * 1.2
    
    pop = cgp.evolve(pop, R ,pb, cgp.MU, cgp.LAMBDA)

  sort_pop = zip(R, pop)
  import pdb; pdb.set_trace()
  pop = [x for _,x in sorted(zip(R, pop), key=lambda x: x[0])]

  with open('best.pkl', 'wb') as handle:
    pickle.dump(pop[-1], handle, protocol=pickle.HIGHEST_PROTOCOL)

  run_episode(env, pop[-1], True)

def run_episode(env, genome ,render = False):
  env.seed(RNG_SEED)
  obs = env.reset()
  done = False
  episode_reward = 0
  step = 0

  while not done:
    if render:
      env.render()

    if step > steps_limit:
      action = NULL_ACTION
    else:
      action = genome.eval(*obs)
      # print(action)

    obs, reward, done, info = env.step(action)
    step += 1
    episode_reward += reward

  return episode_reward

if __name__ == "__main__":
  if sys.argv[1] == 'l':
    game()

  elif sys.argv[1] == 'p':
    import tkinter as tk
    data = []
    prev_obs = None

    env = gym.make("LunarLander-v2")
    env.seed(RNG_SEED)
    prev_obs = env.reset()
    env.render()
    treward = 0

    def key(event):
      global treward, prev_obs
      """shows key or tk code for the key"""
      action = 0

      if event.keysym == 'Escape':
        root.destroy()
      if event.char == event.keysym:
        # normal number and letter characters
        print( 'Normal Key %r' % event.char )
        if event.char == 's':
          action = 0
        if event.char == 'd':
          action = 1
        elif event.char == 'w':
          action = 2
        elif event.char == 'a':
          action = 3

      data.append([prev_obs, action])
      obs, reward, done, info = env.step(action)
      treward += reward
      prev_obs = obs

      env.render()
      
      if done:
        root.destroy()
        with open('human.pkl', 'wb') as handle:
          pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(treward)

    root = tk.Tk()
    print( "Press a key (Escape key to exit):" )
    root.bind_all('<Key>', key)
    root.mainloop()

  else:
    with open('best.pkl', 'rb') as handle:
      genome = pickle.load(handle)
      env = gym.make("LunarLander-v2")
      run_episode(env, genome, True)
