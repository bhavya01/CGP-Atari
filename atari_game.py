
import atari_cgp as cgp
import gym
import pickle
import sys
import time
import pygame
import matplotlib
import argparse
from gym import logger
try:
		matplotlib.use('TkAgg')
		import matplotlib.pyplot as plt
except ImportError as e:
		logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
		plt = None

from collections import deque
from pygame.locals import VIDEORESIZE


POP_SIZE = cgp.MU + cgp.LAMBDA
RNG_SEED = 1
NULL_ACTION = 0

def learn(env):

	gen_scores = []
	pop = cgp.create_population(POP_SIZE)

	env.seed(RNG_SEED)
	env.reset()
	
	R = [0]*POP_SIZE
	
	for gen in range(cgp.N_GEN):
	
		for p in range(len(pop)):
			
			R[p] = run_episode(env, pop[p], True)
	
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
	episode_reward = 0

	if render:
		env.render()

	env_done = False

	while not env_done:
		prev_obs = obs

		action = genome.eval(*(obs - prev_obs))

		obs, reward, env_done, info = env.step(action)
				
		if render:
			env.render()

		episode_reward += reward

	return episode_reward  

def display_arr(screen, arr, video_size, transpose):
		arr_min, arr_max = arr.min(), arr.max()
		arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
		pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
		pyg_img = pygame.transform.scale(pyg_img, video_size)
		screen.blit(pyg_img, (0,0))

def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None, render=False):
		"""Allows one to play the game using keyboard.
		To simply play the game use:
				play(gym.make("Pong-v4"))
		Above code works also if env is wrapped, so it's particularly useful in
		verifying that the frame-level preprocessing does not render the game
		unplayable.
		If you wish to plot real time statistics as you play, you can use
		gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
		for last 5 second of gameplay.
				def callback(obs_t, obs_tp1, action, rew, done, info):
						return [rew,]
				plotter = PlayPlot(callback, 30 * 5, ["reward"])
				env = gym.make("Pong-v4")
				play(env, callback=plotter.callback)
		Arguments
		---------
		env: gym.Env
				Environment to use for playing.
		transpose: bool
				If True the output of observation is transposed.
				Defaults to true.
		fps: int
				Maximum number of steps of the environment to execute every second.
				Defaults to 30.
		zoom: float
				Make screen edge this many times bigger
		callback: lambda or None
				Callback if a callback is provided it will be executed after
				every step. It takes the following input:
						obs_t: observation before performing action
						obs_tp1: observation after performing action
						action: action that was executed
						rew: reward that was received
						done: whether the environment is done or not
						info: debug info
		keys_to_action: dict: tuple(int) -> int or None
				Mapping from keys pressed to action performed.
				For example if pressed 'w' and space at the same time is supposed
				to trigger action number 2 then key_to_action dict would look like this:
						{
								# ...
								sorted(ord('w'), ord(' ')) -> 2
								# ...
						}
				If None, default key_to_action mapping for that env is used, if provided.
		"""
		env.reset()
		rendered=env.render( mode='rgb_array')

		if keys_to_action is None:
				if hasattr(env, 'get_keys_to_action'):
						keys_to_action = env.get_keys_to_action()
				elif hasattr(env.unwrapped, 'get_keys_to_action'):
						keys_to_action = env.unwrapped.get_keys_to_action()
				else:
						assert False, env.spec.id + " does not have explicit key to action mapping, " + \
													"please specify one manually"
		relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))
		
		video_size=[rendered.shape[1],rendered.shape[0]]
		if zoom is not None:
				video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

		pressed_keys = []
		running = True
		env_done = True

		screen = pygame.display.set_mode(video_size)
		clock = pygame.time.Clock()


		while running:
				if env_done:
						env_done = False
						obs = env.reset()
				else:
						action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
						prev_obs = obs
						obs, rew, env_done, info = env.step(action)
						if callback is not None:
								callback(prev_obs, obs, action, rew, env_done, info)
				if obs is not None:
						rendered=env.render( mode='rgb_array')
						display_arr(screen, rendered, transpose=transpose, video_size=video_size)

				# process pygame events
				for event in pygame.event.get():
						# test events, set key states
						if event.type == pygame.KEYDOWN:
								if event.key in relevant_keys:
										pressed_keys.append(event.key)
								elif event.key == 27:
										running = False
						elif event.type == pygame.KEYUP:
								if event.key in relevant_keys:
										pressed_keys.remove(event.key)
						elif event.type == pygame.QUIT:
								running = False
						elif event.type == VIDEORESIZE:
								video_size = event.size
								screen = pygame.display.set_mode(video_size)
								print(video_size)

				pygame.display.flip()
				clock.tick(fps)
		pygame.quit()

class PlayPlot(object):
		def __init__(self, callback, horizon_timesteps, plot_names):
				self.data_callback = callback
				self.horizon_timesteps = horizon_timesteps
				self.plot_names = plot_names

				assert plt is not None, "matplotlib backend failed, plotting will not work"

				num_plots = len(self.plot_names)
				self.fig, self.ax = plt.subplots(num_plots)
				if num_plots == 1:
						self.ax = [self.ax]
				for axis, name in zip(self.ax, plot_names):
						axis.set_title(name)
				self.t = 0
				self.cur_plot = [None for _ in range(num_plots)]
				self.data     = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

		def callback(self, obs_t, obs_tp1, action, rew, done, info):
				points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
				for point, data_series in zip(points, self.data):
						data_series.append(point)
				self.t += 1

				xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

				for i, plot in enumerate(self.cur_plot):
						if plot is not None:
								plot.remove()
						self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
						self.ax[i].set_xlim(xmin, xmax)
				plt.pause(0.000001)

def main():
		parser = argparse.ArgumentParser()
		parser.add_argument('--env', type=str, default='PongNoFrameskip-v0', help='Define Environment')
		parser.add_argument('--mode', type=str, default='play', help='Define Mode (play, learn, best)')
		args = parser.parse_args()
		env = gym.make(args.env)

		if args.mode == 'l' or args.mode == 'learn':
			learn(env)

		elif args.mode == 'p' or args.mode == 'play':
			play(env, zoom=4, fps=30)

		else:
			with open('best.pkl', 'rb') as handle:
				genome = pickle.load(handle)
				env = gym.make("Pong-v0")
				reward = run_episode(env, genome, True)
				print("episode reward is : " + str(reward))


if __name__ == '__main__':
		main()