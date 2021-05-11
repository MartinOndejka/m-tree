import math


DEFAULT_NODE_SIZE = 5

def euclidean_distance(a, b):
    return math.sqrt(sum([(a[i] - b[i]) ** 2 for i in range(len(a))]))

class MTree:

    def __init__(self, node_size=DEFAULT_NODE_SIZE, df=euclidean_distance):
        self.root = []
        self.node_size = node_size
        self.df = df

    def add(self, obj):
        self._insert(self.root, obj)
    
    def add_bulk(self, l):
        for i in l:
            self.add(i)

    def range_query(self, query, radius):
        pass

    def knn_query(self, query, k):
        pass

    def _insert(self, node, obj):
        pass

    def _find_leaf(self, node, obj):
        if isinstance(node, LeafNode):
            return node
        
        

    def _split(self, node, obj):
        pass


class AbstractNode:
    def __init__(self, entries=None):
        self.entries = entries if entries else []

class InternalNode(AbstractNode):
    pass

class LeafNode(AbstractNode):
    pass


class RoutingObject:
    def __init__(self, value, radius, p_dist, subtree=None):
        self.value = value
        self.radius = radius
        self.p_dist = p_dist
        self.subtree = subtree if subtree else []


class GroundObject:
    def __init__(self, value, uid, p_dist):
        self.value = value
        self.uid = uid
        self.p_dist = p_dist
