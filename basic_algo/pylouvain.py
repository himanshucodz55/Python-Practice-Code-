#!/usr/bin/env python3

'''
    Implements the Louvain method.
    Input: a weighted undirected graph
    Ouput: a (partition, modularity) pair where modularity is maximum
'''
class PyLouvain:

    '''
        Builds a g
        raph from _path.
        _path: a path to a file containing "node_from node_to" edges (one per line)
    '''
    @classmethod
    def from_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()    #reads until EOF using readline() and returns a list containing the lines
        f.close()
        nodes = {}     #dict
        edges = []     #list
        for line in lines:
            n = line.split()      #retn a list
            if not n:
                break
            nodes[n[0]] = 1        #assigning that node is present by making dict
            nodes[n[1]] = 1        #   ---
            w = 1                  #weight
            if len(n) == 3:
                w = int(n[2])
            edges.append(((n[0], n[1]), w))     #ex. [(('0', '9'), 1), (('0', '14'), 1)]
            #print(edges)

        # rebuild graph with successive identifiers (gives consecutive indexs)
        nodes_, edges_ = in_order(nodes, edges)     #def on line.295`
        print("%d nodes, %d edges" % (len(nodes_), len(edges_)))
        return cls(nodes_, edges_)    #retn usable class obj...its a factory method

    '''
        Builds a graph from _path.
        _path: a path to a file following the Graph Modeling Language(gml) specification
    '''
    @classmethod
    def from_gml_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        current_edge = (-1, -1, 1)    #tuple
        in_edge = 0
        for line in lines:
            words = line.split()      #retn list
            if not words:
                break
            if words[0] == 'id':
                nodes[int(words[1])] = 1
            elif words[0] == 'source':
                in_edge = 1        #to tell that info about edge is being inputed
                current_edge = (int(words[1]), current_edge[1], current_edge[2])
            elif words[0] == 'target' and in_edge:
                current_edge = (current_edge[0], int(words[1]), current_edge[2])
            elif words[0] == 'value' and in_edge:
                current_edge = (current_edge[0], current_edge[1], int(words[1]))
            elif words[0] == ']' and in_edge:
                #edges.append(((current_edge[0], current_edge[1]), 1))
                edges.append(((current_edge[0], current_edge[1]), current_edge[2]))  ########Changes made
                current_edge = (-1, -1, 1)
                in_edge = 0
        nodes, edges = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return cls(nodes, edges)

    '''
        Initializes the method.
        _nodes: a list of ints
        _edges: a list of ((int, int), weight) pairs
    '''
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        # precompute m (sum of the weights of all links in network)
        #            k_i (sum of the weights of the links incident to node i)
        self.m = 0
        self.k_i = [0 for n in nodes]
        self.edges_of_node = {}
        self.w = [0 for n in nodes]
        for e in edges:
            self.m += e[1]
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1] # there's no self-loop initially
            # save edges by node
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)

            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)    #to avoid duplicates due to self loops
        # access community of a node in O(1) time
        #print(self.m)
        #print(self.k_i)
        #print(edges)
        #print(self.edges_of_node)
        self.communities = [n for n in nodes]
        self.actual_partition = []


    '''
        Applies the Louvain method.
    '''
    @property            #retn property object ---getter --setter
    def apply_method(self):
        network = (self.nodes, self.edges)     #tuple
        #print(type(network))
        best_partition = [[node] for node in network[0]]
        #print(best_partition)
        best_q = -1
        i = 1
        while 1:
            print("pass #%d" % i)
            i += 1
            partition = self.first_phase(network)     # Def line 172
            q = self.compute_modularity(partition)     #just below
            partition = [c for c in partition if c]
            #print("partition :: ",partition)
            print("%s (%.8f)" % (partition, q))
            #print(self.communities)
            print("No. of communities :: ",len(partition))

            # clustering initial nodes with partition
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                        #print("part :: ",part)
                    actual.append(part)
                    #print("actual :: ",actual)
                self.actual_partition = actual           #making the actual partition,,,,,using previous and current partitions
                print("actual_partition :: ",self.actual_partition)
            else:
                self.actual_partition = partition
            if q == best_q:
                break
            network = self.second_phase(network, partition)   #def on line-264
            best_partition = partition
            best_q = q

        return (self.actual_partition, best_q)

    '''
        Computes the modularity of the current network.
        _partition: a list of lists of nodes
    '''
    def compute_modularity(self, partition):
        q = 0
        m2 = self.m * 2
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q

    '''
        Computes the modularity gain of having node in community _c.
        _node: an int
        _c: an int
        _k_i_in: the sum of the weights of the links from _node to nodes in _c
    '''
    def compute_modularity_gain(self, node, c, k_i_in):
        return 2 * k_i_in - self.s_tot[c] * self.k_i[node] / self.m

    '''tgg
        Performs the first phase of the method.
        _network: a (nodes, edges) pair
    '''
    def first_phase(self, network):
        # make initial partition
        best_partition = self.make_initial_partition(network)    #def on line-237
        while 1:            #if there is at least one node being moved, the algorithm will jump back to step  for another iteration
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]
                # neigh_chosen = []
                # neigh_list = self.get_neighbors(node)  ##
                # for neigh in neigh_list:
                #     if self.communities[node] != self.communities[neigh]:
                #         neigh_chosen.append(neigh)  ##new


                # default best community is its own
                best_community = node_community
                best_gain = 0
                # remove _node from its community
                best_partition[node_community].remove(node)      #****************
                best_shared_links = 0
                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_links += e[1]
                self.s_in[node_community] -= 2 * (best_shared_links + self.w[node])    #i guess w is for weight of self loops
                self.s_tot[node_community] -= self.k_i[node]
                #equal = self.communities[node]
                self.communities[node] = -1
                # print("******************\n",best_shared_links)
                # print(self.s_in[node_community])
                # print(self.s_tot[node_community])
                # print(self.communities)
                # print("*************************")
                communities = {} # only consider neighbors of different communities
                for neighbor in self.get_neighbors(node):       #def on line 225
                    # if equal == self.communities[neighbor]:
                    #     #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    #     continue
                    #else :
                    #    print("||")
                    community = self.communities[neighbor]   #it is the community of neighbour
                    if community in communities:             ###if neigh with same community is already parsed then dont do it again
                        continue
                    communities[community] = 1
                    shared_links = 0    #sum of weight of links shared by node with neighbour community
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue
                        if e[0][0] == node and self.communities[e[0][1]] == community or e[0][1] == node and self.communities[e[0][0]] == community:
                            shared_links += e[1]
                    # compute modularity gain obtained by moving _node to the community of _neighbor
                    gain = self.compute_modularity_gain(node, community, shared_links)    #def on line 164
                    if gain > best_gain:
                        best_community = community     ##changing community to neighbour community
                        best_gain = gain
                        best_shared_links = shared_links
                # insert _node into the community maximizing the modularity gain
                best_partition[best_community].append(node)
                self.communities[node] = best_community
                self.s_in[best_community] += 2 * (best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                # print(node_community,"==",best_community)
                if node_community != best_community:
                    improvement = 1
            if not improvement:
                break
        return best_partition

    '''
        Yields the nodes adjacent to _node.
        _node: an int
    '''
    def get_neighbors(self, node):
        for e in self.edges_of_node[node]:
            if e[0][0] == e[0][1]: # a node is not neighbor with itself
                continue
            if e[0][0] == node:
                yield e[0][1]
            if e[0][1] == node:
                yield e[0][0]

    # def get_neighbors(self, node):       ##############
    #     neigh=[]
    #     for e in self.edges_of_node[node]:
    #         if e[0][0] == e[0][1]: # a node is not neighbor with itself
    #             continue
    #         if e[0][0] == node:
    #             neigh.append(e[0][1])
    #         if e[0][1] == node:
    #             neigh.append(e[0][0])
    #     return neigh

    '''
        Builds the initial partition from _network.
        _network: a (nodes, edges) pair
    '''
    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        self.s_in = [0 for node in network[0]]
        self.s_tot = [self.k_i[node] for node in network[0]]
        for e in network[1]:
            if e[0][0] == e[0][1]: # only self-loops
                self.s_in[e[0][0]] += e[1]
                self.s_in[e[0][1]] += e[1] #????
        return partition

    '''
        Performs the second phase of the method.
        _network: a (nodes, edges) pair
        _partition: a list of lists of nodes
    '''
    def second_phase(self, network, partition):
        nodes_ = [i for i in range(len(partition))]
        # relabelling communities
        communities_ = []
        d = {}
        i = 0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                i += 1
        # print(self.communities)
        # print(communities_)
        self.communities = communities_
        # building relabelled edges
        edges_ = {}
        #print(network[1])
        for e in network[1]:
            ci = self.communities[e[0][0]]      #renumbering edges and vertices
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] += e[1]
            except KeyError:
                edges_[(ci, cj)] = e[1]    #keyerror means value of key is not present in dict ...so if not present declare it e[1]
        edges_ = [(k, v) for k, v in edges_.items()]
        #print(edges_)
        # recomputing k_i vector and storing edges by node

        self.k_i = [0 for n in nodes_]
        self.edges_of_node = {}
        self.w = [0 for n in nodes_]
        for e in edges_:
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1]
            if e[0][0] == e[0][1]:
                self.w[e[0][0]] += e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # resetting communities
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)

'''
    Rebuilds a graph with successive nodes' ids.
    _nodes: a dict of int
    _edges: a list of ((int, int), weight) pairs
'''
def in_order(nodes, edges):
        # rebuild graph with successive identifiers
        nodes = list(nodes.keys())
        #print(nodes)
        nodes.sort()
        i = 0
        nodes_ = []
        d = {}
        for n in nodes:
            nodes_.append(i)
            d[n] = i
            i += 1
        edges_ = []
        for e in edges:
            edges_.append(((d[e[0][0]], d[e[0][1]]), e[1]))

        return (nodes_, edges_)
