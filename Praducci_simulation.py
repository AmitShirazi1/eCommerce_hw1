import numpy as np
import networkx as nx
import random
import pandas as pd
import csv

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


def choose_influencers(NoseBook_network, ordered_nodes):
    """
    Choose the influencers you want to work with
    :param NoseBook_network: The NoseBook social network
    :return: A list of the influencers you chose
    """
    # the total cost of the influencers should not exceed 1000
    money_left = 1000
    influencers = []
    while money_left > 0:
        influencer = ordered_nodes.pop(0)
        # while influencer in influencers:
        #     influencer = random.choice(list(NoseBook_network.nodes))
        while get_influencers_cost(cost_path, influencers + [influencer]) > 1000:
            influencer = ordered_nodes.pop(0)
            print(" expensive influencer: ", influencer)
        influencers.append(influencer)
        neighbors = [influencer]
        for _ in range(6):
            for node in neighbors:
                neighbors += list(NoseBook_network.neighbors(node))
                print("")
        ordered_nodes = [node for node in ordered_nodes if node not in neighbors]
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
    # calculate the degree centrality of each node in the network
    degree_centrality = [key for key, _ in sorted(nx.degree_centrality(NoseBook_network).items(), key=lambda item: item[1], reverse=True)]
    print("Degree centrality: ", degree_centrality)
    print(type(degree_centrality))
    # .sort_values(ascending=False).keys()
    # calculate the closeness centrality of each node in the network
    # closeness_centrality = nx.closeness_centrality(NoseBook_network).sort_values(ascending=False).keys()
    # calculate the betweenness centrality of each node in the network  
    # betweenness_centrality = nx.betweenness_centrality(NoseBook_network).sort_values(ascending=False).keys()
    # calculate the eigenvector centrality of each node in the network
    





    influencers = choose_influencers(NoseBook_network, list(degree_centrality))
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
