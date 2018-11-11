import gym
from gym import error, spaces, logger, utils
from gym.utils import seeding
import math
import numpy as np
import os
import json

SOURCE = "source"
DEST = "dest"
TRANSIT_COST = "transit_cost"
RIDESHARE_COST = "rideshare_cost"
DEMAND = "demand"


def parse_edge(vertex, edge_json_obj):
    return TransportEdge(source=vertex, dest=edge_json_obj[DEST], transit_cost=edge_json_obj[TRANSIT_COST],
                         rideshare_cost=edge_json_obj[RIDESHARE_COST], demand=edge_json_obj[DEMAND])

def import_transportation_graph(file_name):
    json_string = None
    with open(file_name, 'r') as config_file:
        json_string = config_file.read()

    assert json_string, "JSON configuration could not be parsed. Given: {}".format(json_string)

    json_config = json.loads(json_string)

    graph = {}
    vertex_list = []
    for vertex in json_config.keys():
        graph[vertex] = [parse_edge(vertex, edge_obj) for edge_obj in json_config[vertex]]
        vertex_list.append(vertex)

    return graph, vertex_list
        


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
        Type: Dict Space of Box Matrices. Element (i,j) of the box represents the price or rebalance
              factor on the edge from i to j. If no such edge exists, the bounds for this element will be
              min 0, max 0.
        price-mult:  Box of price multipliers
        rebalance: Box of rebalancing quantities 

        The price actions represent setting the price multiplier
        The relabalcing actions represent the number of cars to send along each edge without carrying a customer
        The numbers here represent the valid number of cars to send
    """

    metadata = {'render.modes': ['human']}

    # Maximum price multiplier. Making this value small decreases action space.
    MAX_MULTIPLIER = 5

    # Proportion of a rideshare ride which is seen as cost. A value of 1 means all rideshare profits
    # come from multipliers (i.e. the baseline price is equal to the cost).
    COST_FACTOR = 1

    # Total number of available rideshare cars
    TOTAL_CARS = 20

    # Constant in the logistic discrete choice probability. Ideally, this constant should be fit
    # from data. For now, we hard-code its value.
    PROBABILITY_CONSTANT = 1

    # Action space dictionary keys
    PRICE_MULT = 'price-mult'
    REBALANCE = 'rebalance'

    def __init__(self):

        # For now, we create a hard-coded transportation graph of 4 nodes. This is represented
        # as a dictionary to easily enable re-naming vertices. 
        # self.transportation_graph = {
        #     0: [TransportEdge(0, 1, 6, 4, 10), TransportEdge(0, 2, 2, 1.5, 12)],
        #     1: [TransportEdge(1, 0, 0, 0.1, 0), TransportEdge(1, 3, 0, 1, 0)],
        #     2: [TransportEdge(2, 0, 0, 1, 0), TransportEdge(2, 3, 0, 0.1, 0)],
        #     3: [TransportEdge(3, 1, 2, 1.5, 12), TransportEdge(3, 2, 6, 4, 10)]
        # }
        self.transportation_graph, self.vertex_list = import_transportation_graph('transportation_graph.json')

        # Compute the number of vertices based on the graph
        self.num_vertices = len(self.transportation_graph.keys())

        print(self.transportation_graph)

        # Compute the number of edges based on the graph
        self.num_edges = 0
        for value in self.transportation_graph.values():
            self.num_edges += len(value)

        print(self.num_vertices)
        print(self.num_edges)


        # Define the observation space
        observation_min = np.zeros(self.num_vertices)
        observation_max = np.full(shape=self.num_vertices, fill_value=self.TOTAL_CARS)
        self.observation_space = spaces.Box(observation_min, observation_max, dtype=np.int32)

        # We make our action space a VxV matrix for simplicity. This can be translated
        # to an adjacency list style model later on for space conservation.
        self.price_actions = self._form_price_action_space(max_value = self.MAX_MULTIPLIER)

        # Initialize price action space in an adjacendy list manner
        # price_multipliers = dict()
        # for i in range(0, self.num_vertices):
        #     price_tuples = tuple([spaces.Box(0, np.inf, shape=(1,), dtype='float32') for _ in range(len(self.transportation_graph[i]))])
        #     price_multipliers[str(i)] = spaces.Tuple(price_tuples)
        # self.price_actions = spaces.Dict(price_multipliers)

        # Initialize Seed
        self.seed()

        # The state is a list of the number of available cars at each node. Initialized in reset() function.
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

        for source_index in range(0, self.num_vertices):
            source = self.vertex_list[source_index]
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

                # Update the number of cars at the destination
                new_state[dest] += cars_to_send

                # Add the profit of these trips
                rideshare_profits += cars_to_send * profits[i]

            # After rides have been decided, rebalancing is factored in. By default, if more rebalancing than
            # available cars is asked, then the links with lowest cost are chosen
            rebalance_counts = [(self._get_action_value(action, self.REBALANCE, edge), edge.dest) for edge in out_edges]
            rebalance_costs = [edge.rideshare_cost * self.COST_FACTOR for edge in out_edges]
            rebalance_indices = np.flip(np.argsort(rebalance_costs))
            for i in rebalance_indices:
                if num_cars <= 0:
                    break

                # Find the rebalancing counts and destinations determined by this action
                count, dest = rebalance_counts[i]

                # Determine the maximum number of cars we can send
                cars_to_send = min(count, num_cars)

                # Reduce the number of cars at this vertex
                num_cars -= cars_to_send

                # Increase the number of cars at the destination
                new_state[dest] += cars_to_send

                # Reduce the profits by the cost it takes to send this car without a customer
                rideshare_profits -= cars_to_send * rebalance_costs[i]

            # The remaining cars stay put at this vertex
            new_state[source] += num_cars


        # Validate the new state. We leave this in for now for testing purposes. This part should
        # be removed later on.
        car_count = 0
        for count in new_state:
            assert count >= 0, "Number of cars should not be negative."
            car_count += count
        assert car_count == self.TOTAL_CARS, "Number of cars should never decrease."

        # The simulation is never done. This is an infinite horizon MDP and there is no 'failure' case.
        # Trianing algorithm should instead determine a stopping threshold.
        return np.array(new_state), rideshare_profits, False, {}

    def reset(self):
        # Generate an initial distribution of cars
        self.state = self.np_random.rand(self.num_vertices)

        # We need to ensure the sum is equal to the total number of cars
        # This method may not generate a uniform distribution, but suffices for our purposes
        self.state = self.state / np.sum(self.state)
        self.state = self.state * self.TOTAL_CARS
        self.state = self.state.astype(int)

        # We add the remaining cars to a random vertex
        rand_index = self.np_random.randint(low=0, high=self.num_vertices)
        diff = self.TOTAL_CARS - np.sum(self.state)
        self.state[rand_index] += diff
        
        return np.array(self.state)

    def render(self, mode='human', close=False):
        return None

    # The actions are dependent on the states because rebalancing depends on availability of cars
    def get_action_space(self):
        min_rebalance = np.zeros(shape=(self.num_vertices, self.num_vertices))
        max_rebalance = np.zeros(shape=(self.num_vertices, self.num_vertices))
        for i in range(0, self.num_vertices):
            vertex = self.vertex_list[i]
            edges = self.transportation_graph[vertex]
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
        return self.PROBABILITY_CONSTANT - rideshare_cost + transit_cost

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

    def _form_price_action_space(self, max_value, data_type='float32'):
        min_values = np.zeros(shape=(self.num_vertices, self.num_vertices))
        max_values = np.zeros(shape=(self.num_vertices, self.num_vertices))
        for i in range(0, self.num_vertices):
            vertex = self.vertex_list[i]
            edges = self.transportation_graph[vertex]
            for j in range(0, self.num_vertices):
                edge = self._get_edge(i, j)
                if edge and edge.demand > 0:
                    max_values[i, j] = max_value
        return spaces.Box(min_values, max_values, dtype=data_type)


class TransportEdge:

    def __init__(self, source, dest, transit_cost, rideshare_cost, demand):
        self.source = source
        self.dest = dest
        self.transit_cost = transit_cost
        self.rideshare_cost = rideshare_cost
        self.demand = demand


