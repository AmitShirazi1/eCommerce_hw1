# *************** Expected final score is 1368.55 ***************
# *************** Expected final score is 1277.85 ***************
# *************** Expected final score is 1335.15 ***************
# *************** Expected final score is 1336.9 ***************
# *************** Expected final score is 1399.2 ***************
# *************** Expected final score is 1307.45 ***************
# *************** Expected final score is 1268.0 ***************
# *************** Expected final score is 1434.1 ***************
# *************** Expected final score is 1419.65 ***************
# *************** Expected final score is 1344.15 ***************
# *************** Expected final score is 1304.1 ***************
# *************** Expected final score is 1279.8 ***************
# *************** Expected final score is 1366.3 ***************
# *************** Expected final score is 1269.5 ***************
# *************** Expected final score is 1453.7 ***************
# *************** Expected final score is 1448.25 ***************
# *************** Expected final score is 1325.45 ***************
# *************** Expected final score is 1348.1 ***************
# *************** Expected final score is 1407.15 ***************
# *************** Expected final score is 1314.1 ***************
# *************** Expected final score is 1273.35 ***************
# *************** Expected final score is 1358.05 ***************
# *************** Expected final score is 1364.4 ***************
# *************** Expected final score is 1365.9 ***************
# *************** Expected final score is 1344.0 ***************
# *************** Expected final score is 1343.65 ***************
# *************** Expected final score is 1510.05 ***************
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


###DELETE LATER
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


def visualize_influence_by_cost(nodes, centrality_measure, costs):
    # Visualize influence by cost
    degree = [centrality_measure[node] for node in nodes]
    cost = [costs[node] for node in nodes]
    plt.scatter(cost, degree)
    plt.xlabel('Cost')
    plt.ylabel('score')
    plt.title('score by Cost')
    plt.show()


def get_expectation_of_final_score(NoseBook_network, purchased):
    epochs = 20
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
    top_5_per_cost = {cost: sorted(dict_of_costs[cost], key=lambda x: degree_centrality[x], reverse=True)[:10] for cost in costs_set}

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

def normalize_and_combine_centrality_scores(*centrality_dicts):
    # Extract nodes
    nodes = list(centrality_dicts[0].keys())
    
    # Combine all centrality scores into a single array
    combined_scores = np.array([
        [centrality_dict[node] for centrality_dict in centrality_dicts]
        for node in nodes
    ])
    
    # Normalize scores manually
    min_vals = combined_scores.min(axis=0)
    max_vals = combined_scores.max(axis=0)
    normalized_scores = (combined_scores - min_vals) / (max_vals - min_vals)
    
    # Sum normalized scores to get a single combined score for each node
    combined_scores = normalized_scores.sum(axis=1)
    
    # Create a dictionary of combined scores
    combined_centrality = dict(zip(nodes, combined_scores))
    
    return combined_centrality




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
    #eigen_centrality = nx.eigenvector_centrality(NoseBook_network)

    """CURRENTLY WORKS THE BEST"""
    #visualize_influence_by_cost(NoseBook_network.nodes, degree_centrality, costs)#-currently works the best
    #top_5_per_cost = most_neighbors_per_cost(degree_centrality, costs) - currently works best

   # combined_centrality = normalize_and_combine_centrality_scores(degree_centrality, closeness_centrality, betweenness_centrality, eigen_centrality)

    
    # comb_scores = {node: 100*degree_centrality[node]+costs[node] if node in costs and costs[node] != 0 else 0 for node in NoseBook_network.nodes}
    # # Sort nodes by normalized degree centrality
    # sorted_nodes = sorted(comb_scores, key=lambda x: comb_scores[x], reverse=True)
    # print("Sorted nodes: ", sorted_nodes)
    # #influencers= get_influencers_by_score(sorted_nodes, costs)
    # #influencers = choose_influencers(NoseBook_network, list(sorted_nodes))
    # influencers=[2035]
    # print("Influencers: ", influencers)
    # influencers_cost = get_influencers_cost(cost_path, influencers)
    # print("Influencers cost: ", influencers_cost)
    # if influencers_cost > 1000:
    #     print("*************** Influencers are too expensive! ***************")
    #     exit()
    # purchased = set(influencers)    
    # get_expectation_of_final_score(NoseBook_network, purchased)


  #  print("Combined centrality: ", combined_centrality)

    """trying to find the best score - each top 10 nodes per cost"""
#    # visualize_influence_by_cost(NoseBook_network.nodes, betweenness_centrality, costs)
#     top_5_per_cost_deg = most_neighbors_per_cost(degree_centrality, costs) 
#     top_5_per_cost_bet = most_neighbors_per_cost(betweenness_centrality, costs) 
#     top_5_per_cost_clos = most_neighbors_per_cost(closeness_centrality, costs)
#     top_5_per_cost_eig = most_neighbors_per_cost(eigen_centrality, costs)
#     #intersection of the four
#     top_5_per_cost = {cost: list(set(top_5_per_cost_deg[cost]) & set(top_5_per_cost_bet[cost]) & set(top_5_per_cost_clos[cost]) & set(top_5_per_cost_eig[cost])) for cost in costs.values()}
#     print("Top 5 per cost: ", top_5_per_cost)
#     scores_calculations(NoseBook_network, top_5_per_cost)

    # top_5_per_cost_combined = most_neighbors_per_cost(combined_centrality, costs)
    # print("Top 5 per cost combined: ", top_5_per_cost_combined)

    # influencers = get_influencers_by_score(sorted_by_degree, costs)
    """ Current best score - influencers = [1608,3266,3260,3448]-1460 """
    influencers = [ 2035]

    '''maximization by trying the best nodes with 100 score that did not make through the intersection of each centrality'''
    nodes_cost_100_set1 = [
        318, 2295, 617, 3002, 1455, 3448, 1453, 814, 2480, 3391
    ]
    nodes_cost_100_set2 = [
        3892, 2516, 3798, 3654, 200, 3448, 2191, 3293, 1897, 1887
    ]
    nodes_cost_100_set3 = [
        3798, 3654, 1897, 2771, 3659, 1719, 2282, 3047, 1110
    ]
    nodes_cost_100_set4 = [
        318, 617, 2295, 3002, 2480, 814, 1453, 259, 3391, 525
    ]
    node_cost_100_set_5 =[318,2295, 617, 3002,1455,3448,1453,814, 2480,3391 ]

    # Combine all sets into a single list
    all_nodes = (
        nodes_cost_100_set1 + nodes_cost_100_set2 +
        nodes_cost_100_set3 + nodes_cost_100_set4)
    
    #influencers = choose_influencers(NoseBook_network, list(sorted_nodes))
    print("Influencers: ", influencers)
    influencers_cost = get_influencers_cost(cost_path, influencers)
    print("Influencers cost: ", influencers_cost)
    if influencers_cost > 1000:
        print("*************** Influencers are too expensive! ***************")
        exit()
    purchased = set(influencers)    
    get_expectation_of_final_score(NoseBook_network, purchased)

    # Remove duplicates by converting the list to a set and back to a list
    unique_nodes = list(set(all_nodes))
    print("Unique nodes: ", unique_nodes)

  
   
    exp_scores = []

    # Calculate expectation scores for each nod
    for node1 in unique_nodes:
        influencers.append(node1)
        for node in unique_nodes:
            influencers.append(node)
            purchased = set(influencers)
            print("Influencers: ", influencers)
            influencers_cost = get_influencers_cost(cost_path, influencers)
            print("Influencers cost: ", influencers_cost)
            exp_score = get_expectation_of_final_score(NoseBook_network, purchased)
            influencers.remove(node)
            
            # Append the score only if it is not None
            if exp_score is not None:
                exp_scores.append((node, exp_score))
        influencers.remove(node1)
        # Find the maximum expectation score and corresponding node
        if exp_scores:
            max_node, max_score = max(exp_scores, key=lambda x: x[1])
            # Print the result
            print("*************** Your final score is " + str(max_score) + ", node " + str(max_node) + " ***************")
        else:
            print("No valid scores calculated.")



'''the original main'''

    # for i in range(6):
    #     purchased = buy_products(NoseBook_network, purchased)
    #     print("finished round", i + 1)

    # score = product_exposure_score(NoseBook_network, purchased)

    # print("*************** Your final score is " + str(score) + " ***************")

   

