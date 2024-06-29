import numpy as np
import networkx as nx
import random
import pandas as pd
import csv
import matplotlib.pyplot as plt
import itertools


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




def get_influencers_by_score(sorted_nodes, costs):
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
    return influencers

import networkx as nx

# Assuming G is your NetworkX graph, and node_a and node_b are the two nodes you're comparing
def calculate_neighborhood_overlap(G, node_a, node_b):
    neighbors_a = set(G.neighbors(node_a))
    neighbors_b = set(G.neighbors(node_b))
    
    # Intersection of neighbors
    common_neighbors = neighbors_a.intersection(neighbors_b)
    
    # Union of neighbors
    all_neighbors = neighbors_a.union(neighbors_b)
    
    # Avoid division by zero
    if len(all_neighbors) == 0:
        return 0
    
    # Neighborhood overlap
    overlap = len(common_neighbors) / len(all_neighbors)
    return overlap


def visualize_influence_by_cost(nodes, degree_centrality, costs):
    # Visualize influence by cost
    degree = [degree_centrality[node] for node in nodes]
    cost = [costs[node] for node in nodes]
    plt.scatter(cost, degree)
    plt.xlabel('Cost')
    plt.ylabel('Degree')
    plt.title('Degree by Cost')
    plt.show()


def get_expectation_of_final_score(NoseBook_network, purchased):
    epochs = 1000
    scores = np.zeros(epochs)
    initial_purchased = purchased
    for j in range(epochs):
        for i in range(6):
            purchased = buy_products(NoseBook_network, purchased)

        score = product_exposure_score(NoseBook_network, purchased)
        purchased = initial_purchased
        scores[j] = score
    print("*************** Expected final score is " + str(np.mean(scores)) + " ***************")


def most_neighbors_per_cost(degree_centrality, costs):
    costs_set = set(costs.values())
    # making list on nodes with the same cost
    dict_of_costs = {cost: [] for cost in costs_set}
    for cost in costs_set:
        dict_of_costs[cost] = [node for node in degree_centrality.keys() if costs[node] == cost]
    # max_per_cost = max(dict_of_costs.values(), key=lambda x: degree_centrality[x])
    # print("Cost: ", cost, " Node with most neighbors: ", max_per_cost, " Degree centrality: ", degree_centrality[dict_of_costs[cost]])
    top_5_per_cost = {cost: sorted(dict_of_costs[cost], key=lambda x: degree_centrality[x], reverse=True)[:5] for cost in costs_set}
    top_5_per_cost[100] += [sorted(dict_of_costs[100], key=lambda x: degree_centrality[x], reverse=True)[5]]
    for cost in costs_set:
        print("Cost: ", cost, " 5 top nodes: ")
        for node in top_5_per_cost[cost]:
            print(node, " Degree centrality: ", degree_centrality[node])
    return top_5_per_cost


def scores_calculations(graph, top_5_per_cost):
    top_nodes = [node for nodes in top_5_per_cost.values() for node in nodes]

    # Create a mapping of nodes to indices
    node_to_index = {node: i for i, node in enumerate(top_nodes)}

    # Calculate and store neighborhood overlap for every pair of nodes
    num_nodes = len(top_nodes)
    overlap_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for node_a in top_nodes:
        for node_b in top_nodes:
            overlap = calculate_neighborhood_overlap(graph, node_a, node_b)
            index_a = node_to_index[node_a]
            index_b = node_to_index[node_b]
            # Store the overlap in both symmetric positions in the matrix
            overlap_matrix[index_a][index_b] = overlap
            overlap_matrix[index_b][index_a] = overlap
            print("Overlap between nodes", node_a, "and", node_b, "is", overlap)
    # overlap_matrix now contains the neighborhood overlap values for all node pairs

    # Normalize and compute cost-effectiveness
    degree_scores = {node: costs[node] ** (degree_centrality[node] * 10) if node in costs and costs[node] != 0 else 0 for node in NoseBook_network.nodes}
    # closeness_scores = {node: betweenness_centrality[node] * 10 + 100 * costs[node]  if node in costs and costs[node] != 0 else 0 for node in NoseBook_network.nodes}
    # neighborhood_overlap_scores = {node: (sum(overlap_matrix[node_to_index[node]]) / num_nodes) * 10 + 100 * costs[node] if node in costs and costs[node] != 0 else 0 for node in NoseBook_network.nodes}
    # Sort nodes based on cost-effectiveness
    sorted_by_degree = sorted(degree_scores.keys(), key=lambda x: degree_scores[x], reverse=True)
    
    # sorted_by_closeness = sorted(closeness_scores.keys(), key=lambda x: closeness_scores[x], reverse=True)
    # sorted_by_neighborhood_overlap = sorted(neighborhood_overlap_scores.keys(), key=lambda x: neighborhood_overlap_scores[x], reverse=False)
    return sorted_by_degree


if __name__ == '__main__':

    print("STARTING")

    NoseBook_network = create_graph(NoseBook_path)
    # # Draw the network
    # pos = nx.random_layout(NoseBook_network)  # positions for all nodes
    # nx.draw(NoseBook_network, pos, with_labels=False, node_size=10, width=0.01, alpha=0.5, node_color='teal')
    # plt.show()
    
    costs = pd.read_csv(cost_path)
    costs = dict(zip(costs['user'], costs['cost']))

    degree_centrality = nx.degree_centrality(NoseBook_network)
    # closeness_centrality = nx.closeness_centrality(NoseBook_network)  # Runs for too long
    # betweenness_centrality = nx.betweenness_centrality(NoseBook_network)

    # visualize_influence_by_cost(NoseBook_network.nodes, degree_centrality, costs)
    # top_5_per_cost = most_neighbors_per_cost(degree_centrality, costs)
    # scores_calculations(NoseBook_network, top_5_per_cost)


    # influencers = get_influencers_by_score(sorted_by_degree, costs)
    """ Current best score - influencers = [1608,3266,3260,3448]-1430,1417 """
    influencers = [1608,3266,3260,3448]
    #influencers = choose_influencers(NoseBook_network, list(sorted_nodes))
    print("Influencers: ", influencers)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    print("Influencers cost: ", influencers_cost)
    if influencers_cost > 1000:
        print("*************** Influencers are too expensive! ***************")
        exit()

    purchased = set(influencers)

    get_expectation_of_final_score(NoseBook_network, purchased)

    # for i in range(6):
    #     purchased = buy_products(NoseBook_network, purchased)
    #     print("finished round", i + 1)

    # score = product_exposure_score(NoseBook_network, purchased)

    # print("*************** Your final score is " + str(score) + " ***************")
