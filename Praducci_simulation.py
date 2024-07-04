import numpy as np
import networkx as nx
import random
import pandas as pd
import csv

ID1, ID2 = 319044434, 314779166

NoseBook_path = 'NoseBook_friendships.csv'
cost_path = 'costs.csv'

""" ---------------------------------- Given Functions --------------------------------------------- """


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


""" ---------------------------------- Old Unused Functions --------------------------------------------- """

class Node:
    def __init__(self, id, num_neighbors, expected_num_bought_neighbors=[0]*6):
        self.id = id
        self.cost = None
        self.num_neighbors = num_neighbors
        self.expected_num_bought_neighbors = expected_num_bought_neighbors
    
    def get_buying_prob(self, t):
        """ Returns the probability of buying at round t """
        return self.expected_num_bought_neighbors[t] / self.num_neighbors
    
    def add_neighbor_to_buyers(self, t):
        """ Adds a neighbor to the set of neighbors that bought at round t """
        self.expected_num_bought_neighbors[t] += 1

    def set_cost(self, cost):
        self.cost = cost


def choose_influencers(NoseBook_network, ordered_nodes):
    """
    Choose the influencers we want to start with
    :param NoseBook_network: The NoseBook social network
    :param ordered_nodes: A list of the nodes ordered by a centrality measure of some sort
    :return: A list of the influencers we chose
    """
    # the total cost of the influencers should not exceed 1000
    money_left = 1000
    influencers = []
    neigh_to_remove = set()
    i = 0
    while money_left > 0 and i < len(ordered_nodes):
        influencer = ordered_nodes[i]
        i += 1
        # If the influencer is too expensive, skip to the next one
        while get_influencers_cost(cost_path, influencers + [influencer]) > 1000:
            influencer = ordered_nodes[i]
            i += 1
        influencers.append(influencer)

        curr_neighbors = {influencer}
        nodes_created = dict()
        # Simulating 6 rounds of buying
        for t in range(6):
            neighbors = set()
            curr_neighbors = curr_neighbors.difference(neigh_to_remove)  # Remove the neighbors that already bought in a previous round from the neighbors of the current round
            neigh_to_remove = neigh_to_remove.union(curr_neighbors)  # Add the neighbors that already bought in a previous round to the set of neighbors to remove

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

            curr_neighbors = neighbors  # Add bernoulli according to the probability of buying
                                        # Also, cancel duplicate detection and refer to probabilities


        ordered_nodes = [node for node in ordered_nodes if node not in neigh_to_remove]
        money_left -= get_influencers_cost(cost_path, influencers)  # Update the money left after the addition of the influencer
    return influencers


def scores_calculations(NoseBook_network, degree_centrality, costs):
    """ Calculate the scores for the top k nodes per cost """
    # Normalize and compute cost-effectiveness
    degree_scores = {node: costs[node] ** (degree_centrality[node] * 10) if node in costs and costs[node] != 0 else 0 for node in NoseBook_network.nodes}

    sorted_by_degree = sorted(degree_scores.keys(), key=lambda x: degree_scores[x], reverse=True)
    
    return sorted_by_degree


def get_influencers_by_score(sorted_nodes, costs):
    """ Get the influencers based on the score and the cost.
        This function is instead of the choose_influencers function. """
    budget = 1000
    influencers = []
    current_budget = budget
    for node in sorted_nodes:
        if costs[node] <= current_budget:  # If the cost of the influencer is less than the budget, add them
            influencers.append(node)
            current_budget -= costs[node]  # Update the budget
        if current_budget <= 0:
            break    
    return influencers


def build_neighborhood_overlap_matrix(graph, top_k_per_cost):
    """ Build the neighborhood overlap matrix, only for the top k nodes (per cost). """
    top_nodes = [node for nodes in top_k_per_cost.values() for node in nodes]

    # Create a mapping of nodes to indices
    node_to_index = {node: i for i, node in enumerate(top_nodes)}

    # Calculate and store neighborhood overlap for every pair of nodes
    num_nodes = len(top_nodes)
    overlap_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for node_a in top_nodes:
        for node_b in top_nodes:
            overlap = neighborhood_overlap(graph, node_a, node_b)
            index_a = node_to_index[node_a]
            index_b = node_to_index[node_b]
            # Store the overlap in both symmetric positions in the matrix
            overlap_matrix[index_a][index_b] = overlap
            overlap_matrix[index_b][index_a] = overlap
            print("Overlap between nodes", node_a, "and", node_b, "is", overlap)
    # overlap_matrix now contains the neighborhood overlap values for all node pairs
    return overlap_matrix, node_to_index


""" ---------------------------------- Current used Functions --------------------------------------------- """


def calculate_centrality_measures(NoseBook_network):
    costs = pd.read_csv(cost_path)
    costs = dict(zip(costs['user'], costs['cost']))
    degree_centrality = nx.degree_centrality(NoseBook_network)
    closeness_centrality = nx.closeness_centrality(NoseBook_network)
    betweenness_centrality = nx.betweenness_centrality(NoseBook_network)
    eigen_centrality = nx.eigenvector_centrality(NoseBook_network)
    return costs, degree_centrality, closeness_centrality, betweenness_centrality, eigen_centrality


def neighborhood_overlap(graph, node1, node2):
    if node1 == node2:
        return 1
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    
    intersection = len(neighbors1 & neighbors2)
    union = len(neighbors1 | neighbors2)
    
    if union == 0:
        return 0
    else:
        return intersection / union
    

def create_neighborhood_overlap_matrix(graph, nodes):
    size = len(nodes)
    overlap_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            overlap_matrix[i][j] = neighborhood_overlap(graph, nodes[i], nodes[j])
    
    return overlap_matrix


def get_expectation_of_final_score(NoseBook_network, purchased):
    """ Get the expectation of the final score """
    epochs = 1000  # Number of epochs to run, change this value
    scores = np.zeros(epochs)
    initial_purchased = purchased  # Save the initial purchased set
    for j in range(epochs):
        # Run the simulation for 6 rounds
        for i in range(6):
            purchased = buy_products(NoseBook_network, purchased)  # Find the new nodes that bought the product

        score = product_exposure_score(NoseBook_network, purchased)  # Calculate the score
        purchased = initial_purchased
        scores[j] = score
    print("Expected final score is " + str(np.mean(scores)))  # Print the approximation to the expectation of the final score


def centrality_score_per_cost(centrality_measure, costs, k=30):
    """ Get the k nodes with the best value w.r.t a centrality measure per cost """
    costs_set = set(costs.values())
    # A dictionary that will contain all nodes with the same cost
    dict_of_costs = {cost: [] for cost in costs_set}
    for cost in costs_set:
        # Making lists of nodes with the same cost
        dict_of_costs[cost] = [node for node in centrality_measure.keys() if costs[node] == cost]
    # The 5 nodes with the best value w.r.t a centrality measure per cost
    top_k_per_cost = {cost: sorted(dict_of_costs[cost], key=lambda x: centrality_measure[x], reverse=True)[:k] for cost in costs_set}

    # Print the chosen k nodes for each cost
    for cost in costs_set:
        print("Cost: ", cost, str(k)+"  top nodes: ")
        for node in top_k_per_cost[cost]:
            print(node, " Centrality measure: ", centrality_measure[node])
    return top_k_per_cost


""" ---------------------------------- Experimenting Functions --------------------------------------------- """


def present_top_nodes_for_every_measure(costs, degree_centrality, closeness_centrality, betweenness_centrality, eigen_centrality, k=30):
    """ Present the top 30 nodes for each centrality measure per cost. """
    top_k_per_cost_deg = centrality_score_per_cost(degree_centrality, costs, k)
    print("top", k, "degree centrality: ", top_k_per_cost_deg)
    top_k_per_cost_bet = centrality_score_per_cost(betweenness_centrality, costs, k)
    print("top", k, "betweenness centrality: ", top_k_per_cost_bet)
    top_k_per_cost_clos = centrality_score_per_cost(closeness_centrality, costs, k)
    print("top", k, "closeness centrality: ", top_k_per_cost_clos)
    top_k_per_cost_eig = centrality_score_per_cost(eigen_centrality, costs, k)
    print("top", k, "eigen centrality: ", top_k_per_cost_eig)
    return top_k_per_cost_deg, top_k_per_cost_bet, top_k_per_cost_clos, top_k_per_cost_eig


def nodes_with_same_cost_across_measures(top_k_per_cost_deg, top_k_per_cost_bet, top_k_per_cost_clos, top_k_per_cost_eig, chosen_cost):
    # Make unique nodes
    all_nodes = (
            top_k_per_cost_deg[chosen_cost] + top_k_per_cost_bet[chosen_cost] +
            top_k_per_cost_clos[chosen_cost] + top_k_per_cost_eig[chosen_cost])
    return list(set(all_nodes))


def overlap_matrix_of_top_nodes(NoseBook_network, unique_nodes):
    """ Build the neighborhood overlap matrix for the top k nodes per cost. """
    print("Unique nodes: ", sorted(unique_nodes))
    print("len unique nodes", len(unique_nodes))
    overlap_matrix = create_neighborhood_overlap_matrix(NoseBook_network, unique_nodes)
    # Make the matrix dataframe
    overlap_matrix_df = pd.DataFrame(overlap_matrix, columns=unique_nodes, index=unique_nodes)
    #export data frame to csv
    overlap_matrix_df.to_csv('overlap_matrix.csv')
    #open with excel
    overlap_matrix_df.to_excel('overlap_matrix.xlsx')


def complete_influencers_list_120_200_100(curr_influencers, top_across_measures_from_cost_100, top30_across_measures_from_costs_120_200):
    """ Complete the influencers list with the best nodes that cost 120, 200 and 100. """
    top_across_measures_from_cost_100 = set(top_across_measures_from_cost_100) - set(curr_influencers)
    print(top_across_measures_from_cost_100)
    influencers =curr_influencers
    # Maximize the score by trying the best nodes with 100 cost that did not make through the intersection of each centrality
    max_score = 0
   
    for node2 in top30_across_measures_from_costs_120_200:
        influencers.append(node2)
        for node1 in top_across_measures_from_cost_100:
            influencers.append(node1)
            purchased = set(influencers)
            print("Influencers: ", influencers)
            influencers_cost = get_influencers_cost(cost_path, influencers)
            # Check if the cost is higher then 1000
            if influencers_cost > 1000:
                print("Influencers are too expensive!")
                return
            print("Influencers cost: ", influencers_cost)
            current_score = get_expectation_of_final_score(NoseBook_network, purchased)
            if current_score>max_score:
                max_score=current_score
                max_node1=node1
                max_node2=node2
            influencers.remove(node1)
        influencers.remove(node2)    
        print(" Max score is  " + str(max_score) + ", 200 or 120 node " + str(max_node2) +", 100 node ", max_node1) 
        print("Influencers: ", influencers + [max_node1]  + [max_node2]) 


def complete_influencers_list_100(curr_influencers, top_across_measures_from_cost_100):
    """ Complete the influencers list with the best node that costs 100. """
    top_across_measures_from_cost_100 = set(top_across_measures_from_cost_100) - set(curr_influencers)
    print(top_across_measures_from_cost_100)
    influencers =curr_influencers
    #maximize the score by trying the best nodes with 100 cost that did not make through the intersection of each centrality
    max_score=0
   
    for node1 in top_across_measures_from_cost_100:
        influencers.append(node1)
        purchased = set(influencers)
        print("Influencers: ", influencers)
        influencers_cost = get_influencers_cost(cost_path, influencers)
        #check if the cost  then 1000
        if influencers_cost > 1000:
            print("Influencers are too expensive!")
            return
        print("Influencers cost: ", influencers_cost)
        current_score = get_expectation_of_final_score(NoseBook_network, purchased)
        if current_score>max_score:
            max_score=current_score
            max_node1=node1
        influencers.remove(node1)
        
    print(" Max score is  " + str(max_score) +", 100 node ", max_node1) 
    print("Influencers: ", influencers + [max_node1] )



if __name__ == '__main__':

    print("STARTING")

    NoseBook_network = create_graph(NoseBook_path)
    
    costs, degree_centrality, closeness_centrality, betweenness_centrality, eigen_centrality = calculate_centrality_measures(NoseBook_network)
    
    # For every cost, get the top 20 nodes with the best value w.r.t a centrality measure.
    # The printed output shows that the nodes that cost 100 give the best score to all the measures with respect to the cost.
    # Meaning, we concluded that the cost of nodes that are more expensive is not worth the investement.
    top_20_per_cost_deg, top_20_per_cost_bet, top_20_per_cost_clos, top_20_per_cost_eig = present_top_nodes_for_every_measure(costs, degree_centrality, closeness_centrality, betweenness_centrality, eigen_centrality, 20)
    
    # Out of the top nodes of the various measures (for the cost of 100),
    # check the overlapping of the neighborhoods of the nodes.
    top20_across_measures_from_cost_100 = nodes_with_same_cost_across_measures(top_20_per_cost_deg, top_20_per_cost_bet, top_20_per_cost_clos, top_20_per_cost_eig, 100)
    overlap_matrix_of_top_nodes(NoseBook_network,top20_across_measures_from_cost_100)

    top_30_per_cost_deg, top_30_per_cost_bet, top_30_per_cost_clos, top_30_per_cost_eig = present_top_nodes_for_every_measure(costs, degree_centrality, closeness_centrality, betweenness_centrality, eigen_centrality, 30)
    # After checking the Excel sheet from the previous function,
    # we found 7 nodes with overlap 0 between them: [3448, 2516, 2771, 3293, 259, 3461, 3851].
    # We know we need to get to the cost 1000 in total, so we need to complete the list with a node that costs 100, 120 or 200.
    # So, we want to test each of the top nodes (across all measures) that cost 100, 120 and 200 and see which fits best.
    top30_across_measures_from_cost_120 = nodes_with_same_cost_across_measures(top_30_per_cost_deg, top_30_per_cost_bet, top_30_per_cost_clos, top_30_per_cost_eig, 120)
    top30_across_measures_from_cost_200 = nodes_with_same_cost_across_measures(top_30_per_cost_deg, top_30_per_cost_bet, top_30_per_cost_clos, top_30_per_cost_eig, 200)
    top30_across_measures_from_costs_120_200 = list(set(top30_across_measures_from_cost_120 + top30_across_measures_from_cost_200))

    top30_across_measures_from_cost_100 = nodes_with_same_cost_across_measures(top_30_per_cost_deg, top_30_per_cost_bet, top_30_per_cost_clos, top_30_per_cost_eig, 100)

    # This is where we iterate over all "good" nodes of each cost value, and test which combination gives the best score.
    # During the calculations, we pproximated the expectation of every combination of nodes.
    # This is done by removing nodes that donate the least to the list's expected score, and adding a different node every time.
    # (We put example nodes in the lists for demonstration)
    complete_influencers_list_120_200_100([3448, 318, 3851, 24, 777, 3654, 2771], top30_across_measures_from_cost_100, top30_across_measures_from_costs_120_200)
    complete_influencers_list_100([3448, 318, 3851, 3370, 24, 777, 3654, 2771], top30_across_measures_from_cost_100)
    """ Recent results we got from this calculation:

        Reducing the node 3448 lowers the score in 200 points.
        Reducing the node 3370 lowers the score in 300 points.
        Reducing the node 449 lowers the score in 110 points.

        influencers = [3448, 3659, 318, 3851, 3370, 24, 777, 3654, 449], expected score: 1885.646
        influencers = [3448, 318, 3851, 3370, 24, 777, 3654, 449, 2771], expected score: 1911
        influencers = [3448, 2516, 3659, 318, 3851, 3370, 24, 777, 3654], expected score: 1867.832
        influencers = [3448, 318, 3851, 3370, 24, 777, 3654, 2771, 132], expected score: 1953
    """

    influencers = [3448, 318, 3851, 3370, 24, 777, 3654, 2771, 132]
    
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
