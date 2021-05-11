import math
from uuid import uuid4


DEFAULT_NODE_SIZE = 5

def euclidean_distance(a, b):
    return math.sqrt(
        sum([
            (a[i] - b[i]) ** 2
            for i in range(len(a))
        ]))

class MTree:

    def __init__(self, node_size=DEFAULT_NODE_SIZE, df=euclidean_distance):
        self.root = LeafNode()
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

    def _insert(self, root, obj):
        leaf = self._find_leaf(root, obj)

        if len(leaf.entries) < self.node_size:
            leaf.entries.append(
                GroundObject(
                    obj,
                    uuid4(),
                    self.df(leaf.parent.value, obj) if leaf.parent else 0
                )
            )
        else:
            self._split(leaf, obj)

    def _find_leaf(self, node, obj):
        if isinstance(node, LeafNode):
            return node
        
        # (obj , dist)
        distances = map(
            lambda i: (i, self.df(i.value, obj)),
            node.entries,
        )

        inside = filter(lambda o: o[1] <= o[0].radius, distances)

        if inside:
            best_entry, _ = min(inside, key=lambda o: o[1])
        else:
            best_entry, dist = min(inside, key=lambda o: o[1] - o[0].radius)
            best_entry.radius = dist
        
        return self._find_leaf(best_entry.subtree, obj)


    def _split(self, node, obj):
        print("SPLIT")


class AbstractNode:
    def __init__(self, entries=None, parent=None):
        self.entries = entries if entries else []
        self.parent = parent

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
