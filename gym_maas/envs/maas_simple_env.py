import gym
from gym import error, spaces, logger, utils
from gym.utils import seeding
import math
import numpy as np

# We assume only the rideshare operator is the actor and users' only preference is price
class MaasSimpleEnv(gym.Env):
    """
    Description: Simple mobility as a service environment in which users choice options based on price and
                 only the rideshare operator has influence over the system. User demand is fixed per timestep.

    Observation:
        Type: Box(|V|), where V is number of vertices
        Observation             Min    Max
        Number of vehicles       0      n

        Here, n is the total number of vehicles. This specification holds for every vertex

    Action:
        Type: Dict Space of Tuples, each tuple entry represents an edge
        price:  (..., Box(1){Min: 0, Max: np.inf} ,...)
        rebalance: (..., Discrete(n_{ij}), ...)

        The price actions represent setting the price multiplier
        The relabalcing actions represent the number of cars to send along each edge without carrying a customer
        The numbers here represent the valid number of cars to send
    """

    metadata = {'render.modes': ['human']}

    MAX_MULTIPLIER = 5
    PRICE_MULT = 'price-mult'
    REBALANCE = 'rebalance'
    COST_FACTOR = 1

    def __init__(self):


        self.num_vertices = 4
        self.total_cars = 20
        self.num_edges = 8

        # For now, we create a hard-coded transportation graph of 4 nodes
        self.transportation_graph = {
            0: [TransportEdge(0, 1, 6, 4, 10), TransportEdge(0, 2, 2, 1.5, 12)],
            1: [TransportEdge(1, 0, 0, 0.1, 0), TransportEdge(1, 3, 0, 1, 0)],
            2: [TransportEdge(2, 0, 0, 1, 0), TransportEdge(2, 3, 0, 0.1, 0)],
            3: [TransportEdge(3, 1, 2, 1.5, 12), TransportEdge(3, 2, 6, 4, 10)]
        }

        # Define the observation space
        observation_min = np.zeros(self.num_vertices)
        observation_max = np.full(shape=self.num_vertices, fill_value=self.total_cars)
        self.observation_space = spaces.Box(observation_min, observation_max, dtype=np.int32)

        # We make our action space a VxV matrix for simplicity. This can be translated
        # to an adjacency list style model later on for space conservation.
        self.price_actions = self._form_matrix_action_space(max_value = self.MAX_MULTIPLIER)

        # Initialize price action space
        # price_multipliers = dict()
        # for i in range(0, self.num_vertices):
        #     price_tuples = tuple([spaces.Box(0, np.inf, shape=(1,), dtype='float32') for _ in range(len(self.transportation_graph[i]))])
        #     price_multipliers[str(i)] = spaces.Tuple(price_tuples)
        # self.price_actions = spaces.Dict(price_multipliers)

        # Initialize Seed
        self.seed()

        # The state is a list of the number of available cars at each node
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        action_space = self.get_action_space()
        assert action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # We simulate the action on this particular state. The transition depends on the randomness
        # introduced by consumer actions
        state = self.state
        rideshare_profits = 0

        # Initialize New State
        new_state = np.zeros(len(state))

        for source in range(0, self.num_vertices):
            out_edges = self.transportation_graph[source]
            rideshare_counts = []
            profits = []
            for edge in out_edges:

                # Get the price multiplier specified by this action
                price_multiplier = self._get_action_value(action, self.PRICE_MULT, edge)
                
                # Keep track of the price of a ride on this edge for tiebreaking, we subtract 1 because
                # profits are determined by the multiplier price above the cost of transportation
                profits.append(edge.rideshare_cost * (price_multiplier - self.COST_FACTOR))

                prob_of_rideshare = self.get_rideshare_probability(edge.rideshare_cost * price_multiplier, edge.transit_cost)
                num_rideshare = self._simulate_rideshare(edge.demand, prob_of_rideshare)
                rideshare_counts.append((num_rideshare, edge.dest))

            # The number of people who want to take rideshare may be greater than the number of available
            # cars. The default policy for rideshare operators is to use cars on the links which give maximum
            # profit.
            sorted_price_indices = np.argsort(profits)
            num_cars = state[source]
            for i in sorted_price_indices:
                if num_cars <= 0:
                    break

                # Determine the number of cars to send
                count, dest = rideshare_counts[i]
                cars_to_send = min(count, num_cars)

                # Decrement number of cars available at this node
                num_cars -= cars_to_send

                # Add the profit of these trips
                rideshare_profits += cars_to_send * profits[i]

                # Update Car Counts
                new_state[dest] += cars_to_send

            # After rides have been decided, rebalancing is factored in. By default, if more rebalancing than
            # available cars is asked, then the links with lowest cost are chosen
            rebalance_counts = [(self._get_action_value(action, self.REBALANCE, edge), edge.dest) for edge in out_edges]
            rebalance_costs = [edge.rideshare_cost for edge in out_edges]
            rebalance_indices = np.flip(np.argsort(rebalance_costs))
            for i in rebalance_indices:
                if num_cars <= 0:
                    break
                count, dest = rebalance_counts[i]
                cars_to_send = min(count, num_cars)

                num_cars -= cars_to_send

                rideshare_profits -= cars_to_send * rebalance_costs[i]

                new_state[dest] += cars_to_send
            
            # The remaining cars stay put
            new_state[source] += num_cars 

        # The simulation is never done, this is an infinite horizon MDP and there is no 'failure' case
        return np.array(new_state), rideshare_profits, False, {}

    def reset(self):
        # Generate an initial distribution of cars
        self.state = self.np_random.rand(self.num_vertices)

        # We need to ensure the sum is equal to the total number of cars
        # This method may not generate a uniform distribution, but suffices for our purposes
        self.state = self.state / np.sum(self.state)
        self.state = self.state * self.total_cars
        self.state = self.state.astype(int)
        diff = self.total_cars - np.sum(self.state)
        self.state[0] += diff # We add the remaining cars to the first vertex
        
        return np.array(self.state)

    def render(self, mode='human', close=False):
        return None

    # The actions are dependent on the states because rebalancing depends on availability of cars
    def get_action_space(self):
        min_rebalance = np.zeros(shape=(self.num_vertices, self.num_vertices))
        max_rebalance = np.zeros(shape=(self.num_vertices, self.num_vertices))
        for i in range(0, self.num_vertices):
            edges = self.transportation_graph[i]
            for j in range(0, self.num_vertices):
                edge = self._get_edge(i, j)
                if edge:
                    # Our rebalancing action is dependent on the number of cars at this vertex
                    num_cars = self.state[i]
                    max_rebalance[i, j] = num_cars
        rebalance_action_space = spaces.Box(min_rebalance, max_rebalance, dtype='int32')

        # Adjacency list style of action space
        # rebalancing = dict()
        # for i in range(0, self.num_vertices):
        #     num_cars = self.state[i]
        #     rebalancing_tuples = tuple([spaces.Box(0, num_cars, shape=(1,), dtype=np.int32) for _ in range(len(self.transportation_graph[i]))])
        #     rebalancing[str(i)] = spaces.Tuple(rebalancing_tuples)

        return spaces.Dict({self.PRICE_MULT: self.price_actions, self.REBALANCE: rebalance_action_space})


    # This represents the probability of a user choosing rideshare over public transit
    # using the given prices. Operates under a Gumbel choice model.
    def get_rideshare_probability(self, rideshare_cost, transit_cost):
        exp = np.exp(self._choice_cost_function(rideshare_cost, transit_cost))
        return float(exp) / float(1 + exp)

    # This cost function has been arbitrarily created for experimentation
    def _choice_cost_function(self, rideshare_cost, transit_cost):
        return 1 - rideshare_cost + transit_cost

    # Returns the number of users who want to take rideshare. The remaining take public transit.
    # This function generates a sample from the consumer discrete choice model.
    def _simulate_rideshare(self, demand, rideshare_prob):
        consumer_decisions = self.np_random.rand(demand)
        rideshare_count = 0
        for decision in consumer_decisions:
            if decision < rideshare_prob:
                rideshare_count += 1
        return rideshare_count

    def _get_action_value(self, action, action_type, edge):
        action_matrix = action[action_type]
        return action_matrix[edge.source, edge.dest]

    def _get_edge(self, source, dest):
        if not source in self.transportation_graph:
            return None
        edges = self.transportation_graph[source]
        for edge in edges:
            if edge.dest == dest:
                return edge
        return False

    def _form_matrix_action_space(self, max_value, data_type='float32'):
        min_values = np.zeros(shape=(self.num_vertices, self.num_vertices))
        max_values = np.zeros(shape=(self.num_vertices, self.num_vertices))
        for i in range(0, self.num_vertices):
            edges = self.transportation_graph[i]
            for j in range(0, self.num_vertices):
                edge = self._get_edge(i, j)
                if edge:
                    max_values[i, j] = max_value
        return spaces.Box(min_values, max_values, dtype=data_type)


class TransportEdge:

    def __init__(self, source, dest, transit_cost, rideshare_cost, demand):
        self.source = source
        self.dest = dest
        self.transit_cost = transit_cost
        self.rideshare_cost = rideshare_cost
        self.demand = demand

class TransportVertex:

    def __init__(self, label):
        self.label = label
