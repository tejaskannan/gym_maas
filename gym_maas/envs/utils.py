import json

SOURCE = 'source'
DEST = 'dest'
TRANSIT_COST = 'transit_cost'
RIDESHARE_COST = 'rideshare_cost'
DEMAND = 'demand'


def parse_edge(vertex, edge_json_obj, vertex_map):
    return TransportEdge(source=vertex_map[vertex], dest=vertex_map[edge_json_obj[DEST]], transit_cost=edge_json_obj[TRANSIT_COST],
                         rideshare_cost=edge_json_obj[RIDESHARE_COST], demand=edge_json_obj[DEMAND])

def import_parameters(file_name):
    json_string = None
    with open(file_name, 'r') as config_file:
        json_string = config_file.read()

    assert json_string, 'JSON configuration could not be parsed. Given: {}'.format(json_string)

    json_config = json.loads(json_string)

    config = json_config['params']
    graph_config = json_config['graph']

    # We enforce an ordering here consistent with label ordering for debugging purposes
    vertex_index = 0
    vertex_map = {} # Maps Vertex name to Vertex number
    key_list = list(graph_config.keys())
    key_list.sort()
    for vertex in key_list:
        vertex_map[vertex] = vertex_index
        vertex_index += 1

    # Vertices are encoded as numbers
    graph = [0] * len(vertex_map)
    for vertex in graph_config.keys():
        vertex_index = vertex_map[vertex]
        graph[vertex_index] = [parse_edge(vertex, edge_obj, vertex_map) for edge_obj in graph_config[vertex]]

    return graph, vertex_map, config

class TransportEdge:

    def __init__(self, source, dest, transit_cost, rideshare_cost, demand):
        self.source = source
        self.dest = dest
        self.transit_cost = transit_cost
        self.rideshare_cost = rideshare_cost
        self.demand = demand