"""
Central module of evosim package.

Documentation auto-generated with pdoc. Use: pdoc evosim/evosim.py > docs/docs.txt
"""

import numpy as np
import pandas as pd

class Population(object):
	"""
	Describes a collection of interacting agents.
	"""
	def __init__(self, agents, mutation_rate=0.1):
		"""
		Args:
			agents (list): list of DataFrames of action probabilities for each agent
			mutation_rate (float): default mutation rate
		"""
		self.agents = pd.Panel.from_dict({agent_num: agent.T for agent_num, agent in enumerate(agents)})
		assert (self.agents.sum()==1).all().all()

		self.mutation_rate = mutation_rate
		self.count = self.agents.shape[0]

	def birth(self, new_agents):
		"""
		Args:
			new_agents (DataFrame or list): action probabilities for single or multiple new agents
		"""
		if type(new_agents) is pd.DataFrame:
			self.agents[self.count] = new_agents
			self.count += 1
		else:
			for new_agent in new_agents:
				birth(self, new_agent)

	def die(self, dying_agents):
		"""
		Args:
			dying_agents (int or list): agent name(s) of agents to kill
		"""
		self.agents.drop(dying_agents, axis='items', inplace=True)

	def update_action_probs(self, updates, agent=None, choice=None):
		"""
		Args:
			updates (Series, DataFrame, or Panel): Series if agent and choice specified, DataFrame if agent OR choice
												   specified, Panel if neither specified
			agent (int): agent number to update 
			choice (int): choice number to update
		"""
		if agent is None and choice is None:
			# all agents across all choices
			pass

		elif not agent is None and choice is None:
			# single agent across all choices
			pass

		elif agent is None and not choice is None:
			# single choice across all agents
			pass

		else:
			# single agent and single choice
			pass

		if type(updates) is tuple:
			# single update
			self.action_probs.iat[update[0]] = update[1]
			assert sum(self.action_probs.values()) == 1

		if type(updates) is list:
			# list of updates
			for update in updates:
				update_action_probs(self, update)
			assert sum(self.action_probs.values()) == 1

		else:
			# full new Series of probabilities
			self.action_probs.update(updates)
			assert sum(self.action_probs.values()) == 1

	def draw_actions(self, agents=None, actions=None, samples=1):
		"""
		Args:
			agents (int or list): agent name(s) of agents to take action. Default all agents in population
			actions (int or list): choice name(s) of choices to draw. Default all choices
			samples (int): number of samples to draw. Default 1 sample
		Returns:
			actions (int, Series, or DataFrame): int if single agent and action, Series if single agent OR action,
												 DataFrame if all actions and agents
		"""
		pass
		# return np.random.choice(self.action_probs.index, p=self.action_probs.values)

class SimplePopulation(Population):
	"""
	Population subclass with simple reproduction, mutation, and natural selection schemes.
	"""
	def mutate(self, mutation_rate=None):
		"""
		Args:
			mutation_rate (Series): override to default mutation rate
		"""
		# self.update_action_probs()
		pass

	def reproduce(self, utilities):
		"""
		Args:
			utilities (Series): utilities of each agent in population
		"""
		# self.birth()
		pass

	def select(self, utilities):
		"""
		Args:
			utilities (Series): utilities of each agent in population
		"""
		# self.die()
		pass

class Environment(object):
	"""
	Describes the selcting pressures of the surroundings.
	"""
	def __init__(self, populations, capacities):
		"""
		Args:
			populations (list) : list of populations
			capacities (list): list of carrying capacaties of each population
		"""
		assert len(populations) == len(capacities)
		self.populations = {pop_number: populations[pop_number] for pop_number in range(len(populations))}
		self.capacities = capacities

class SimpleEnvironment(Environment):
	"""
	Environment with a sinlge population.
	"""
	def __init__(self, population, capacity):
		self.populations = population
		self.capacities = capacity

	def get_utilities(self, actions):
		return None

class Controller(object):
	"""
	Coordinates flow over time.
	"""
	def __init__(self, population, evironment, num_time_steps):
		self.num_time_steps = num_time_steps
		self.population = population
		self.environment = environment

	def run(self):
		"""
		Steps through entire time span.
		"""
		for time_step in range(self.num_time_steps):
			pass 
			# flow during single time step
			actions = self.population.draw_actions()

			utilities = self.environment.get_utilities(actions)

			self.population.reproduce(utilities)

			self.population.select(utilities)

# instantiate everything
agents = [pd.DataFrame([[0.4, 0.6], [0.6, 0.4]]), pd.DataFrame([[0.4, 0.6], [0.5, 0.5]]), pd.DataFrame([[0.7, 0.3], [0.6, 0.4]])]
population = SimplePopulation(agents)
environment = SimpleEnvironment(population, 10)
controller = Controller(population, environment, 100)

controller.run()

