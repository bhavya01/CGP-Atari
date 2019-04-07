
import atari_cgp as cgp

import gym


POP_SIZE = cgp.MU + cgp.LAMBDA
RNG_SEED = 8
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
  
    print("Generation {0}: ".format(gen), R)
    gen_scores.append(sum(R)/len(R))

    max_score = max(R)
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

  run_episode(env, pop[0], True)
  print(gen_scores)

def run_episode(env, genome ,render = False):

  obs = env.reset()
  done = False
  episode_reward = 0
  step = 0

  while not done:
    if render:
      env.render()

    if step > steps_limit:
      move = NULL_ACTION
    else:
      action = genome.eval(*obs)
      print(action)

    obs, reward, done, info = env.step(action)
    step += 1
    episode_reward += reward

  return episode_reward

if __name__ == "__main__":
  game()
