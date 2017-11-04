"""
Central module of evosim package.
"""

# DESIGNS: have agent class? update sum==1 checks, with fun def for tol?

import numpy as np
import pandas as pd

class Population(object):
	"""
	Describes a collection of interacting agents.
	"""
	def __init__(self, agents, default_mutation_rate=0.1):
		"""
		Args:
			agents (list): list of DataFrames of action probabilities for each agent
			default_mutation_rate (float): default mutation rate
		"""
		self.agents = pd.Panel.from_dict({agent_num: agent.T for agent_num, agent in enumerate(agents)})
		assert (self.agents.sum()==1).all().all(), 'probabilities do not sum to 1.'
		self.default_mutation_rate = default_mutation_rate
		
		self.count = self.agents.shape[0]
		self.prob_tolerance = 0.0001

		np.random.seed()

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
				self.birth(self, new_agent)

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
			assert (abs(updates.sum() - 1) < self.prob_tolerance).all().all(), 'probabilities do not sum to 1.'
			self.agents = updates

		elif not agent is None and choice is None:
			# single agent across all choices
			assert (updates.sum()==1).all(), 'probabilities do not sum to 1.'
			self.agents[agent] = updates

		elif agent is None and not choice is None:
			# single choice across all agents
			updates_T = updates.T
			assert (updates.sum()==1).all(), 'probabilities do not sum to 1.'
			assert(updates_T.shape == reversed(self.agents.shape[:2])), 'must provide full update for each agent'
			self.agents.ix[:, :, choice] = updates_T

		else:
			# single agent and single choice
			assert (updates.sum()==1), 'probabilities do not sum to 1.'
			self.agents.ix[agent, :, choice] = updates

	def draw_actions(self, agent=None, choice=None, samples=1):
		"""
		Args:
			agents (int): agent name of agent to take action. Default all agents in population
			choice (int): choice name of choice to draw. Default all choices
			samples (int): number of samples to draw. Default 1 sample
		Returns:
			actions (int, Series, or DataFrame): int if single agent and action, Series if single agent OR action,
												 DataFrame if all actions and agents
		"""
		# TODO: multiple samples 
		num_agents, _, num_choices = self.agents.shape

		if agent is None and choice is None:
			# all agents across all choices
			actions = pd.DataFrame([[np.random.choice(self.agents[agent].index, p=self.agents[agent][choice].values, replace=True) for choice in range(num_choices)] for agent in range(num_agents)])

		elif not agent is None and choice is None:
			# single agent across all choices
			actions = pd.Series([np.random.choice(self.agents[agent].index, p=self.agents[agent][choice].values, replace=True) for choice in range(num_choices)])

		elif agent is None and not choice is None:
			# single choice across all agents
			actions = pd.Series([np.random.choice(self.agents[agent].index, p=self.agents[agent]
				[choice].values, replace=True) for agent in range(num_agents)])
			
		else:
			# single agent and single choice
			actions = np.random.choice(self.agents[agent].index, p=self.agents[agent][choice].values, replace=True)

		return actions

class SimplePopulation(Population):
	"""
	Population subclass with simple reproduction, mutation, and natural selection schemes.
	"""
	def mutate(self, mutation_rate=None):
		# TODO: don't add to 0s, don't add over 1
		"""
		Adds independent N(0, mutation_rate) perturbation to each probability with reflective boundaries on [0, 1], then
		renormalizes.

		Args:
			mutation_rate (Series or DataFrame): override to default mutation rate. Series if different rate for each 
												 agent. DataFrame if different mutation rate for each agent for each 
												 choice
		"""
		num_agents, num_actions, num_choices = self.population.agents.shape

		if type(mutation_rate) is pd.Series:
			# one rate per agent
			mutation_rate_array = pd.Panel([pd.DataFrame(mutation_rate[agent], index=np.arange(num_actions), 
								      columns=np.arange(num_choices)) for agent in num_agents])
		
		elif type(mutation_rate) is pd.DataFrame:
			# one rate per agent per choice
			mutation_rate_array = pd.Panel([pd.DataFrame([[pd.Series(mutation_rate.iloc[choice, agent], index=arange(
								      num_actions))] for choice in range(num_choices)]) for agent in range(num_agents)])
		
		else:
			# scalar
			mutation_rate_array = self.default_mutation_rate if mutation_rate is None else mutation_rate
			
		
		perturbed = abs(self.agents.add(pd.Panel(np.random.normal(scale=mutation_rate_array, 
																  size=self.population.agents.shape))))
		
		perturbed = perturbed.apply(lambda x: x / sum(x), axis=1)
		self.update_action_probs(perturbed)

	def reproduce(self, utilities):
		"""
		Args:
			utilities (Series): utilities of each agent in population
		"""
		mutate()
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
		assert len(populations) == len(capacities), 'must provide exactly one capacity per population'
		self.populations = {pop_number: populations[pop_number] for pop_number in range(len(populations))}
		self.capacities = capacities

class SimpleEnvironment(Environment):
	"""
	Environment with a sinlge population.
	"""
	def __init__(self, population, capacity):
		self.populations = population
		self.capacities = capacity

class MapEnvironment(SimpleEnvironment):
	"""
	Environment with a sinlge population and two choices: {North, South} and {East, West}.
	"""
	def __init__(self, population, capacity):
		self.populations = population
		self.capacities = capacity

	def get_utilities(self, actions):
		# utility is inverse of agents in given position
		utilities = pd.Series([1. / sum([actions[agent].equals(actions[i]) for i in actions]) for agent in actions])
		return utilities

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
			# flow during single time step
			actions = self.population.draw_actions()

			utilities = self.environment.get_utilities(actions)

			self.population.select(utilities)

			self.population.reproduce(utilities)

# instantiate everything
# agents = [pd.DataFrame([[0.4, 0.6], [0.6, 0.4]]), pd.DataFrame([[0.4, 0.6], [0.5, 0.5]]), pd.DataFrame([[0.7, 0.3], [0.6, 0.4]])]
# population = SimplePopulation(agents)
# environment = MapEnvironment(population, 10)
# controller = Controller(population, environment, 100)

# controller.run()

