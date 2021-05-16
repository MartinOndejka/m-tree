import abc
from heapq import heappush, heappop
import math
import random
from operator import itemgetter


def euclidean_distance(a, b):
    return math.sqrt(
        sum(
            map(lambda x, y: (x - y) ** 2, *[a, b])
        )    
    )


def m_lb_dist_promote(entries, old):
    if old is None or any(e.p_dist is None for e in entries):
        return random_promote(entries)
    
    new_entry = max(entries, key=lambda e: e.p_dist)
    return old.obj, new_entry.obj


def random_promote(entries):
    o1, o2 = random.sample(entries, 2)
    return o1.obj, o2.obj


def balanced_distribution(entries, o1, o2, d):
    distances = list(map(
        lambda entry: {
            "o1": d(o1, entry.obj),
            "o2": d(o2, entry.obj),
            "obj": entry,
        },
        entries
    ))

    o1_set, o2_set = set(), set()

    for i in range(len(distances)):
        if i % 2 == 0:
            distances.sort(key=itemgetter("o1"), reverse=True)
            distances[-1]["obj"].p_dist = distances[-1]["o1"]
            o1_set.add(distances[-1]["obj"])
        else:
            distances.sort(key=itemgetter("o2"), reverse=True)
            distances[-1]["obj"].p_dist = distances[-1]["o2"]
            o2_set.add(distances[-1]["obj"])

        distances.pop()

    return o1_set, o2_set


class MTree(object):
    def __init__(self,
                 d=euclidean_distance,
                 node_size=4,
                 promote=m_lb_dist_promote,
                 partition=balanced_distribution,
                 dimensions=2):
        self.d = self.inc_counter(d)
        self.node_size = node_size
        self.promote = promote
        self.partition = partition
        self.size = 0
        self.root = LeafNode(self)
        self.dimensions = dimensions
        self.sequential_data = set()
        self.dcall_counter = 0

    def seed_db(self, dimension, count, lower, upper):
        inp = [
            tuple(random.sample(range(lower, upper), dimension))
            for i in range(count)
        ]
        self.add_bulk(inp)

    def inc_counter(self, d):
        def closure(*args, **kwargs):
            self.dcall_counter += 1
            return d(*args, **kwargs)

        return closure
    
    def set_d_function(self, d):
        self.d = self.inc_counter(d)

    def add(self, obj):
        self.sequential_data.add(obj)
        self._insert(obj)
        self.size += 1

    def add_bulk(self, objects):
        for obj in objects:
            self.add(obj)
    
    def range_search(self, query_obj, r):
        self.dcall_counter = 0
        s = RangeSearch(self)
        s.search(self.root, query_obj, r)
        return s.result
    
    def sequential_range_search(self, query_obj, r):
        self.dcall_counter = 0
        distances = map(
            lambda obj: {
                "obj": obj,
                "d": self.d(obj, query_obj),
            },
            self.sequential_data,
        )

        res = filter(lambda obj: obj["d"] <= r, distances)
        return set(map(itemgetter("obj"), res))

    def sequential_knn(self, query_obj, k=1):
        self.dcall_counter = 0
        distances = list(map(
            lambda obj: {
                "obj": obj,
                "d": self.d(obj, query_obj),
            },
            self.sequential_data,
        ))

        distances.sort(key=itemgetter("d"))

        return set(map(itemgetter("obj"), distances[:k]))

    def knn_search(self, query_obj, k=1):
        self.dcall_counter = 0
        s = knnSearch(self)
        s.search(self.root, query_obj, k)
        return s.result

    def _insert(self, obj):
        leaf = self._find_leaf(self.root, obj)
        entry = Entry(obj,
                      p_dist=leaf.d(obj, leaf.parent_entry.obj) if leaf.parent_entry else None)

        if not leaf.is_full():
            leaf.store(entry)
        else:
            self._split(leaf, entry)

    def _find_leaf(self, node, obj):
        if isinstance(node, LeafNode):
            return node

        distances = list(map(lambda e: (e, self.d(e.obj, obj)), node.entries))
        valid_distances = list(filter(lambda e: e[1] <= e[0].radius, distances))

        if valid_distances:
            best = min(valid_distances, key=lambda e: e[1])
        else:
            best = min(distances, key=lambda e: e[1] - e[0].radius)
            best[0].radius = best[1]

        return self._find_leaf(best[0].subtree, obj)

    def _split(self, node, entry):
        # Union node entries with new entry
        entries = node.entries.copy()
        entries.add(entry)

        # let op be the parent entry of node
        op_entry = node.parent_entry

        # Allocating new node
        new_node = None
        if isinstance(node, InternalNode):
            new_node = InternalNode(mtree=self)
        elif isinstance(node, LeafNode):
            new_node = LeafNode(mtree=self)

        # promoting o1, o2
        o1, o2 = self.promote(entries, node.parent_entry)
        # partition all entries - also computes the new distances
        entries1, entries2 = self.partition(entries, o1, o2, self.d)

        # Create routing entries for objects o1, o2
        o1_entry = Entry(o1, None, node, None)
        o2_entry = Entry(o2, None, new_node, None)

        # Store partitioned entries into nodes and update parent entry radius
        node.update_node(entries1, o1_entry)
        new_node.update_node(entries2, o2_entry)

        # Store promoted routing entries into parent node or a new root node
        # If current node is root, create new root
        if node.is_root():
            new_root = InternalNode(node.mtree)
            self.root = new_root

            node.parent_node = new_root
            new_node.parent_node = new_root

            new_root.store(o1_entry)
            new_root.store(o2_entry)
        # Current node is not root, replace op_entry with o1_entry
        else:
            np = node.parent_node
            np.remove(op_entry)
            np.store(o1_entry)

            # Update distances for promoted entries
            if not np.is_root():
                o1_entry.p_dist = self.d(o1_entry.obj, np.parent_entry.obj)
                o2_entry.p_dist = self.d(o2_entry.obj, np.parent_entry.obj)

            # Store o2_entry in np, if np is full, call split on np
            if np.is_full():
                self._split(np, o2_entry)
            else:
                np.store(o2_entry)
                new_node.parent_node = np


class RangeSearch:
    def __init__(self, mtree):
        self.result = set()
        self.mtree = mtree

    @property
    def d(self):
        return self.mtree.d
    
    def search(self, node, query, r):
        if node == self.mtree.root:
            for i in node.entries:
                self.search(i.subtree, query, r)
            return

        o_p = node.parent_entry

        d_op_q = self.d(o_p.obj, query) if o_p else 0

        if isinstance(node, InternalNode):
            for o_i in node.entries:
                if abs(d_op_q - o_i.p_dist) <= r + o_i.radius:
                    d_oi_q = self.d(o_i.obj, query)

                    if d_oi_q <= r + o_i.radius:
                        self.search(o_i.subtree, query, r)
        
        else:
            for o_i in node.entries:
                if abs(d_op_q - o_i.p_dist) <= r:
                    d_oi_q = self.d(o_i.obj, query)

                    if d_oi_q <= r:
                        self.result.add(o_i.obj)


class knnSearch:
    def __init__(self, mtree):
        self.result = None
        self.mtree = mtree

    def search(self, node, query, k):
        if k <= 0:
            self.result = []
            return

        pr = []
        heappush(pr, PrObject(node, 0, 0))

        # at the end will contain the results
        nn = NN(k)

        while pr:
            prEntry = heappop(pr)
            if prEntry.dmin > nn.rq:
                break
            prEntry.tree.search(query, pr, nn, prEntry.d_query)

        self.result = nn.result_list()


class NN(object):
    def __init__(self, size):
        self.neighbours = [(None, float("inf"))] * size
        self.rq = float("inf")

    def __len__(self):
        return len(self.neighbours)

    def update(self, obj, dmax):
        if obj is None:
            self.rq = min(self.rq, dmax)
            return
        self.neighbours.append((obj, dmax))
        for i in range(len(self)-1, 0, -1):
            if self.neighbours[i][1] < self.neighbours[i-1][1]:
                self.neighbours[i-1], self.neighbours[i] = self.neighbours[i], self.neighbours[i-1]
            else:
                break
        self.neighbours.pop()

    def result_list(self):
        result = map(lambda entry: entry[0], self.neighbours)
        return result
            

class PrObject(object):
    def __init__(self, tree, dmin, d_query):
        self.tree = tree
        self.dmin = dmin
        self.d_query = d_query

    def __lt__(self, other):
        return self.dmin < other.dmin

    
class Entry(object):
    def __init__(self, obj, radius=None, subtree=None, p_dist=None):
        self.obj = obj
        self.radius = radius
        self.subtree = subtree
        self.p_dist = p_dist


class AbstractNode(abc.ABC):
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set()

    @property
    def d(self):
        return self.mtree.d

    def is_full(self):
        return len(self.entries) == self.mtree.node_size

    def is_root(self):
        return self is self.mtree.root

    def store(self, entry):
        self.entries.add(entry)

    def remove(self, entry):
        self.entries.remove(entry)

    @abc.abstractmethod
    def update_node(self, entries, parent_entry):
        pass

    @abc.abstractmethod
    def search(self, query_obj, pr, nn, d_parent_query):
        pass
        

class LeafNode(AbstractNode):
    def __init__(self, mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

        super().__init__(mtree, parent_node, parent_entry, entries)

    def update_node(self, entries, parent_entry):
        self.entries = entries
        self.parent_entry = parent_entry
        self.parent_entry.radius = max(map(lambda e: e.p_dist, self.entries))

    def could_contain_results(self, rq, distance_to_parent, d_parent_query):
        if self.is_root():
            return True
        return abs(d_parent_query - distance_to_parent) <= rq
        
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(nn.rq, entry.p_dist, d_parent_query):
                distance_entry_to_q = self.d(entry.obj, query_obj)
                if distance_entry_to_q <= nn.rq:
                    nn.update(entry.obj, distance_entry_to_q)


class InternalNode(AbstractNode):
    def __init__(self,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

        super().__init__(mtree, parent_node, parent_entry, entries)

    def update_node(self, entries, parent_entry):
        self.entries = entries
        self.parent_entry = parent_entry
        self.parent_entry.radius = max(map(lambda e: e.p_dist + e.radius, self.entries))

        for entry in self.entries:
            entry.subtree.parent_node = self

    def could_contain_results(self, rq, entry, d_parent_query):
        if self.is_root():
            return True
        return abs(d_parent_query - entry.p_dist) <= rq + entry.radius
            
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(nn.rq, entry, d_parent_query):
                d_entry_query = self.d(entry.obj, query_obj)
                entry_dmin = max(d_entry_query - entry.radius, 0)
                if entry_dmin <= nn.rq:
                    heappush(pr, PrObject(entry.subtree, entry_dmin, d_entry_query))
                    entry_dmax = d_entry_query + entry.radius
                    if entry_dmax < nn.rq:
                        nn.update(None, entry_dmax)
