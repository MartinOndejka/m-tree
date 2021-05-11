import math
from uuid import uuid4
import random


DEFAULT_NODE_SIZE = 5

def euclidean_distance(a, b):
    return math.sqrt(
        sum([
            (a[i] - b[i]) ** 2
            for i in range(len(a))
        ]))

def promote_random(all_objects):
    o1 = random.choice(all_objects)
    all_objects.remove(o1)

    o2 = random.choice(all_objects)
    all_objects.remove(o2)

    return o1, o2

def partition_generalized_hyperplane(df):
    def closure(all, o1, o2):
        n1 = []
        n2 = []

        for i in all:
            o1_dist = df(i.value, o1.value)
            o2_dist = df(i.value, o2.value)

            if o1_dist < o2_dist:
                n1.append(i)
            else:
                n2.append(i)
        
        if not n1:
            n2 = n1[:int(len(n1)/2)]
            n1 = n1[int(len(n1)/2):]
        
        if not n2:
            n1 = n2[:int(len(n2)/2)]
            n2 = n2[int(len(n2)/2):]

        return n1, n2
    
    return closure

class MTree:

    def __init__(
        self,
        node_size=DEFAULT_NODE_SIZE,
        df=euclidean_distance,
        promotion_method=promote_random,
        partition_method=None
    ):
        self.root = Node()
        self.node_size = node_size
        self.df = df
        self.promote = promotion_method
        self.partition = partition_method if partition_method else partition_generalized_hyperplane(df)

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
                EntryObject(
                    value=obj,
                    uid=uuid4(),
                    p_dist=self.df(leaf.parent.value, obj) if leaf.parent_object else 0
                )
            )
        else:
            self._split(leaf, obj)

    def is_leaf(self, node):
        return all([i.subtree is None for i in node.entries])

    def _find_leaf(self, node, obj):
        if self.is_leaf(node):
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
        parent_object = None

        if node is not self.root:
            parent_object = node.parent_object
            parent_node = node.parent_node
        
        all_objects = node.entries + [EntryObject(value=obj, p_dist=self.df(parent_object.value, obj) if parent_object else 0)]

        o1, o2 = self.promote(all_objects)
        n1, n2 = self.partition(all_objects, o1, o2)

        node.entries = n1
        node.parent_object = o1
        o1.subtree = node

        new_node = Node(
            entries=n2,
            parent_object=o2,
        )
        o2.subtree = new_node

        if node is self.root:
            self.root = Node(
                entries=[
                    EntryObject(value=o1.value),
                    EntryObject(value=o2.value),
                ],
            )
            self._update_node(self.root)
            node.parent_node = self.root
            new_node.parent_node = self.root
        
        else:
            parent_node.entries.remove(parent_object)
            parent_node.entries.append(o1)

            new_node.parent_node = parent_node

            if parent_node.entries < self.node_size:
                parent_node.entries.append(o2)
            else:
                self._split(parent_node, o2.value)
    
    def _update_node(self, node):
        if node.parent_object is None:
            for i in node.entries:
                i.p_dist = 0
        
        else:
            node.parent_object.radius = 0

            for i in node.entries:
                i.p_dist = self.df(i.value, node.parent_object.value)
                
                if i.p_dist > node.parent_object.radius:
                    node.parent_object.radius = i.p_dist



class Node:
    def __init__(self, entries=None, parent_object=None, parent_node=None):
        self.entries = entries if entries else []
        self.parent_object = parent_object
        self.parent_node = parent_node


class EntryObject:
    def __init__(self, value, p_dist=0, radius=None, uid=None, subtree=None):
        self.value = value
        self.radius = radius
        self.p_dist = p_dist
        self.uid = uid
        self.subtree = subtree
