import numpy as np
import networkx as nx
import random
import pandas as pd
import csv
import matplotlib.pyplot as plt

ID1, ID2 = 319044434, 314779166
# influencers = [555]

NoseBook_path = 'NoseBook_friendships.csv'
cost_path = 'costs.csv'


def influencers_submission(ID1, ID2, lst):
    with open(f'{ID1}_{ID2}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(lst)


def create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the NoseBook social network
    :param edges_path: A csv file that contains information obout "friendships" in the network
    :return: The NoseBook social network
    """
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net


def independent_cascade(G, initial_active):
    active = set(initial_active)
    new_active = set(initial_active)
    
    while new_active:
        current_active = set(new_active)
        new_active = set()
        for node in current_active:
            neighbors = set(G.neighbors(node)) - active
            for neighbor in neighbors:
                if random.random() < 0.5:  # Probability of influencing
                    new_active.add(neighbor)
                    active.add(neighbor)
    return active


def total_influencer_cost(influencers, costs):
    return sum(costs.get(influencer, 0) for influencer in influencers)

def is_within_budget(influencers, budget):
    return total_influencer_cost(influencers, costs) <= budget


class Node:
    def __init__(self, id, num_neighbors, expected_num_bought_neighbors=[0]*6):
        self.id = id
        self.cost = None
        self.num_neighbors = num_neighbors
        self.expected_num_bought_neighbors = expected_num_bought_neighbors
    
    def get_buying_prob(self, t):
        return self.expected_num_bought_neighbors[t] / self.num_neighbors
    
    def add_neighbor_to_buyers(self, t):
        self.expected_num_bought_neighbors[t] += 1

    def set_cost(self, cost):
        self.cost = cost



def choose_influencers(NoseBook_network, ordered_nodes):
    """
    Choose the influencers you want to work with
    :param NoseBook_network: The NoseBook social network
    :return: A list of the influencers you chose
    """
    # the total cost of the influencers should not exceed 1000
    money_left = 1000
    influencers = []
    neigh_to_remove = set()
    i = 0
    while money_left > 0 and i < len(ordered_nodes):
        influencer = ordered_nodes[i]
        i += 1
        # while influencer in influencers:
        #     influencer = random.choice(list(NoseBook_network.nodes))
        while get_influencers_cost(cost_path, influencers + [influencer]) > 1000:
            influencer = ordered_nodes[i]
            i += 1
            print("Expensive influencer: ", influencer)
        influencers.append(influencer)

        curr_neighbors = {influencer}
        nodes_created = dict()
        # Simulating 6 rounds of buying
        for t in range(6):
            neighbors = set()
            curr_neighbors = curr_neighbors.difference(neigh_to_remove)  # Remove the neighbors that were already bought
            neigh_to_remove = neigh_to_remove.union(curr_neighbors)  # Add the neighbors that were already bought

            # Update the neighbors of the current neighbors
            for neigh in curr_neighbors:
                curr_node_neighs = set(NoseBook_network.neighbors(neigh))
                neighbors = neighbors.union(curr_node_neighs)  # Add the neighbors of the current neighbors
                # Create a node for each neighbor
                if neigh not in nodes_created.keys():
                    node = Node(neigh, len(curr_node_neighs))
                    nodes_created[neigh] = node
                else:
                    node = nodes_created[neigh]
                    node.add_neighbor_to_buyers(t)

                
            print("Updated neighbors: ", neighbors)
            curr_neighbors = neighbors  # Add bernoulli according to the probability of buying
            # Also, think about maybe we want duplicates.


        ordered_nodes = [node for node in ordered_nodes if node not in neigh_to_remove]
        money_left -= get_influencers_cost(cost_path, influencers)
        print("Money left: ", money_left)

    return influencers


def buy_products(net: nx.Graph, purchased: set) -> set:
    """
    Gets the network at the beginning of stage t and simulates a purchase round
    :param net: The network at stage t
    :param purchased: All the users who recieved or bought the product up to and including stage t-1
    :return: All the users who recieved or bought the product up to and including stage t
    """
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)

    return new_purchases.union(purchased)


def product_exposure_score(net: nx.Graph, purchased_set: set) -> int:
    """
    Returns the number of users who have seen the product
    :param purchased_set: A set of users who bought the product
    :param net: The NoseBook social network
    :return:  The sum for all users who have seen the product.
    """
    exposure = 0
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))

        if user in purchased_set:
            exposure += 1
        elif len(neighborhood.intersection(purchased_set)) != 0:
            b = len(neighborhood.intersection(purchased_set))
            rand = random.uniform(0, 1)
            if rand < 1 / (1 + 10 * np.exp(-b/2)):
                exposure += 1
    return exposure


def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum([costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in influencers])


if __name__ == '__main__':

    print("STARTING")

    NoseBook_network = create_graph(NoseBook_path)
    # # Draw the network
    # pos = nx.random_layout(NoseBook_network)  # positions for all nodes
    # nx.draw(NoseBook_network, pos, with_labels=False, node_size=10, width=0.01, alpha=0.5, node_color='teal')
    # plt.show()
    
    costs = pd.read_csv(cost_path)
    costs = dict(zip(costs['user'], costs['cost']))


    #sort the nodes by degree centrality and influencers cost of this node
    degree_centrality = nx.degree_centrality(NoseBook_network)
    # Normalize and compute cost-effectiveness
    cost_effectiveness = {node:degree_centrality[node] /0.1* costs[node]  if node in costs and costs[node] != 0 else 0 for node in NoseBook_network.nodes}

    # Sort nodes based on cost-effectiveness
    sorted_nodes = sorted(cost_effectiveness.keys(), key=lambda x: cost_effectiveness[x], reverse=True)

    #sorted_nodes = sorted(NoseBook_network.nodes, key = lambda n: degree_centrality[n]/costs[costs['user'] == n]['cost'].iloc[0], reverse=True)


    # # calculate the degree centrality of each node in the network
    # degree_centrality = [key for key, _ in sorted(nx.degree_centrality(NoseBook_network).items(), key=lambda item: item[1], reverse=True)]
    # # TODO: divide by the cost.
    print("sorted_nodes: ", sorted_nodes)
    print(type(sorted_nodes))
    # .sort_values(ascending=False).keys()
    # calculate the closeness centrality of each node in the network
    # closeness_centrality = nx.closeness_centrality(NoseBook_network).sort_values(ascending=False).keys()
    # calculate the betweenness centrality of each node in the network  
    # betweenness_centrality = nx.betweenness_centrality(NoseBook_network).sort_values(ascending=False).keys()
    # calculate the eigenvector centrality of each node in the network
    
    # Select influencers within budget
    budget = 1000
    influencers = []
    current_budget = budget
    for node in sorted_nodes:
        if costs[node] <= current_budget:
            influencers.append(node)
            current_budget -= costs[node]
        if current_budget <= 0:
            break    

    #influencers = choose_influencers(NoseBook_network, list(sorted_nodes))
    print("Influencers: ", influencers)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    print("Influencers cost: ", influencers_cost)
    if influencers_cost > 1000:
        print("*************** Influencers are too expensive! ***************")
        exit()

    purchased = set(influencers)

    for i in range(6):
        purchased = buy_products(NoseBook_network, purchased)
        print("finished round", i + 1)

    score = product_exposure_score(NoseBook_network, purchased)

    print("*************** Your final score is " + str(score) + " ***************")
