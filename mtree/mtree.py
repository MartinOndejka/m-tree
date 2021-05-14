import abc
from heapq import heappush, heappop
import collections
from itertools import islice, combinations
import math
import random
from operator import itemgetter


def euclidean_distance(a, b):
    return math.sqrt(
        sum(
            map(lambda a, b: (a - b) ** 2, *[a, b])
        )    
    )


def m_lb_dist_promote(entries, old, d):
    if old is None or any(e.distance_to_parent is None for e in entries):
        o1, o2 = random.sample(entries, 2)
        return o1.obj, o2.obj
    
    new_entry = max(entries, key=lambda e: e.distance_to_parent)
    return old.obj, new_entry.obj
    

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
            distances
            o1_set.add(distances[-1]["obj"])
        else:
            distances.sort(key=itemgetter("o2"), reverse=True)
            o2_set.add(distances[-1]["obj"])

        distances.pop()

    return o1_set, o2_set


class MTree(object):
    def __init__(self,
                 d=euclidean_distance,
                 max_node_size=4,
                 promote=m_lb_dist_promote,
                 partition=balanced_distribution):
        """
        Create a new MTree.
        Arguments:
        d: distance function.
        max_node_size: optional. Maximum number of entries in a node of
            the M-tree
        promote: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines given the set of entries which two entries should be
            used as routing object to represent the two nodes in the
            parent node.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        partition: optional. Used during insertion of a new value when
            a node of the tree is split in two.
            Determines to which of the two routing object each entry of the
            split node should go.
            This is delving pretty far into implementation choices of
            the Mtree. If you don't understand all this just swallow
            the blue pill, use the default value and you'll be fine.
        """
        if not callable(d):
            #Why the hell did I put this?
            #This is python, we use dynamic typing and assumes the user
            #of the API is smart enough to pass the right parameters.
            raise TypeError('d is not a function')
        if max_node_size < 2:
            raise ValueError('max_node_size must be >= 2 but is %d' %
                             max_node_size)
        self.d = d
        self.max_node_size = max_node_size
        self.promote = promote
        self.partition = partition
        self.size = 0
        self.root = LeafNode(self)

    def __len__(self):
        return self.size

    def add(self, obj):
        """
        Add an object into the M-tree
        """
        self.root.add(obj)
        self.size += 1

    def add_all(self, iterable):
        """
        Add all the elements in the M-tree
        """
        #TODO: implement using the bulk-loading algorithm
        for obj in iterable:
            self.add(obj)

    def search(self, query_obj, k=1):
        """Return the k objects the most similar to query_obj.
        Implementation of the k-Nearest Neighbor algorithm.
        Returns a list of the k closest elements to query_obj, ordered by
        distance to query_obj (from closest to furthest).
        If the tree has less objects than k, it will return all the
        elements of the tree."""
        k = min(k, len(self))
        if k == 0: return []

        #priority queue of subtrees not yet explored ordered by dmin
        pr = []
        heappush(pr, PrEntry(self.root, 0, 0))

        #at the end will contain the results 
        nn = NN(k)

        while pr:
            prEntry = heappop(pr)
            if(prEntry.dmin > nn.search_radius()):
                #best candidate is too far, we won't have better a answer
                #we can stop
                break
            prEntry.tree.search(query_obj, pr, nn, prEntry.d_query)

            #could prune pr here
            #(the paper prunes after each entry insertion, instead whe could
            #prune once after handling all the entries of a node)
            
        return nn.result_list()

    
NNEntry = collections.namedtuple('NNEntry', 'obj dmax')
class NN(object):
    def __init__(self, size):
        self.elems = [NNEntry(None, float("inf"))] * size
        #store dmax in NN as described by the paper
        #but it would be more logical to store it separately
        self.dmax = float("inf")

    def __len__(self):
        return len(self.elems)

    def search_radius(self):
        """The search radius of the knn search algorithm.
        aka dmax
        The search radius is dynamic."""
        return self.dmax

    def update(self, obj, dmax):
        if obj == None:
            #internal node
            self.dmax = min(self.dmax, dmax)
            return
        self.elems.append(NNEntry(obj, dmax))
        for i in range(len(self)-1, 0, -1):
            if self.elems[i].dmax < self.elems[i-1].dmax:
                self.elems[i-1], self.elems[i] = self.elems[i], self.elems[i-1]
            else:
                break
        self.elems.pop()

    def result_list(self):
        result = map(lambda entry: entry.obj, self.elems)
        return result

    def __repr__(self):
        return "NN(%r)" % self.elems
            

class PrEntry(object):
    def __init__(self, tree, dmin, d_query):
        """
        Constructor.
        arguments:
        d_query: distance d to searched query object
        """
        self.tree = tree
        self.dmin = dmin
        self.d_query = d_query

    def __lt__(self, other):
        return self.dmin < other.dmin

    def __repr__(self):
        return "PrEntry(tree:%r, dmin:%r)" % (self.tree, self.dmin)

    
class Entry(object):
    """
    
    The leafs and internal nodes of the M-tree contain a list of instances of
    this class.
    The distance to the parent is None if the node in which this entry is
    stored has no parent.
    radius and subtree are None if the entry is contained in a leaf.
    Used in set and dict even tough eq and hash haven't been redefined
    """
    def __init__(self,
                 obj,
                 distance_to_parent=None,
                 radius=None,
                 subtree=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent
        self.radius = radius
        self.subtree = subtree

    def __repr__(self):
        return "Entry(obj: %r, dist: %r, radius: %r, subtree: %r)" % (
            self.obj,
            self.distance_to_parent,
            self.radius,
            self.subtree.repr_class() if self.subtree else self.subtree)


class AbstractNode(object):
    """An abstract leaf of the M-tree.
    Concrete class are LeafNode and InternalNode
    We need to keep a reference to mtree so that we can know if a given node
    is root as well as update the root.
    
    We need to keep both the parent entry and the parent node (i.e. the node
    in which the parent entry is) for the split operation. During a split
    we may need to remove the parent entry from the node as well as adding
    a new entry to the node."""

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):
        #There will be an empty node (entries set is empty) when the tree
        #is empty and there only is an empty root.
        #May also be empty during construction (create empty node then add
        #the entries later).
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set()

    def __repr__(self): # pragma: no cover
        #entries might be big. Only prints the first few elements
        entries_str = '%s' % list(islice(self.entries, 2))
        if len(self.entries) > 2:
            entries_str = entries_str[:-1] + ', ...]'
            
        return "%s(parent_node: %s, parent_entry: %s, entries:%s)" % (
            self.__class__.__name__,
            self.parent_node.repr_class() \
                if self.parent_node else self.parent_node,
            self.parent_entry,
            entries_str
            
    )

    def repr_class(self): # pragma: no cover
        return self.__class__.__name__ + "()"

    def __len__(self): 
        return len(self.entries)

    @property
    def d(self):
        return self.mtree.d

    def is_full(self):
        return len(self) == self.mtree.max_node_size

    def is_empty(self):
        return len(self) == 0

    def is_root(self):
        return self is self.mtree.root

    def remove_entry(self, entry):
        """Removes the entry from this node
        Raise KeyError if the entry is not in this node
        """
        self.entries.remove(entry)

    def add_entry(self, entry):
        """Add an entry to this node.
        Raise ValueError if the node is full.
        """
        if self.is_full():
            raise ValueError('Trying to add %s into a full node' % str(entry))
        self.entries.add(entry)

    #TODO recomputes d(leaf, parent)!
    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        self.entries = new_entries
        self.parent_entry = new_parent_entry
        self.parent_entry.radius = self.update_radius(self.parent_entry.obj)
        self._update_entries_distance_to_parent()

    #wastes d computations if parent hasn't changed.
    #How to avoid? -> check if the new routing_object is the same as the old
    # (compare id(obj) not obj directly to prevent == assumption about object?)
    def _update_entries_distance_to_parent(self):
        if self.parent_entry:
            for entry in self.entries:
                entry.distance_to_parent = self.d(entry.obj,
                                                  self.parent_entry.obj)

    @abc.abstractmethod
    def add(self, obj): # pragma: no cover
        """Add obj into this subtree"""
        pass

    @abc.abstractmethod         
    def update_radius(self, obj): # pragma: no cover
        """Compute the radius needed for obj to cover the entries of this node.
        """
        pass

    @abc.abstractmethod
    def search(self, query_obj, pr, nn, d_parent_query):
        pass
        

class LeafNode(AbstractNode):
    """A leaf of the M-tree"""
    def __init__(self,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

        AbstractNode.__init__(self,
                              mtree,
                              parent_node,
                              parent_entry,
                              entries)
    def add(self, obj):
        distance_to_parent = self.d(obj, self.parent_entry.obj) \
            if self.parent_entry else None
        new_entry = Entry(obj, distance_to_parent)
        if not self.is_full():
            self.entries.add(new_entry)
        else:
            split(self, new_entry, self.d)
        assert self.is_root() or self.parent_node        

    def update_radius(self, obj):
        """Compute minimal radius for obj so that it covers all the objects
        of this node.
        """
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj), self.entries))

    def could_contain_results(self,
                              query_obj,
                              search_radius,
                              distance_to_parent, 
                              d_parent_query):
        """Determines without any d computation if there could be
        objects in the subtree belonging to the result.
        """
        if self.is_root():
            return True
        
        return abs(d_parent_query - distance_to_parent)\
                <= search_radius
        
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(query_obj,
                                          nn.search_radius(),
                                          entry.distance_to_parent,
                                          d_parent_query):
                distance_entry_to_q = self.d(entry.obj, query_obj)
                if distance_entry_to_q <= nn.search_radius():
                    nn.update(entry.obj, distance_entry_to_q)


class InternalNode(AbstractNode):
    """An internal node of the M-tree"""

    def __init__(self,
                 mtree,
                 parent_node=None,
                 parent_entry=None,
                 entries=None):

        AbstractNode.__init__(self,
                              mtree,
                              parent_node,
                              parent_entry,
                              entries)

    #TODO: apply optimization that uses the d of the parent to reduce the
    #number of d computation performed. cf M-Tree paper 3.3
    def add(self, obj):     
        #put d(obj, e) in a dict to prevent recomputation 
        #I guess memoization could be used to make code clearer but that is
        #too magic for me plus there is potentially a very large number of
        #calls to memoize
        dist_to_obj = {}
        for entry in self.entries:
            dist_to_obj[entry] = self.d(obj, entry.obj)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = [e for e in self.entries
                             if dist_to_obj[e] <= e.radius]
            return min(valid_entries, key=dist_to_obj.get) \
                if valid_entries else None
                
        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries,
                             key=lambda e: dist_to_obj[e] - e.radius)
            #enlarge radius so that obj is in the covering radius of e 
            entry.radius = dist_to_obj[entry]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or \
            find_best_entry_minimizing_radius_increase()
        entry.subtree.add(obj)
        assert self.is_root() or self.parent_node

    def update_radius(self, obj):
        """Compute minimal radius for obj so that it covers the radiuses
        of all the routing objects of this node
        """
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj) + e.radius,
                           self.entries))

    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        AbstractNode.set_entries_and_parent_entry(self,
                                                  new_entries,
                                                  new_parent_entry)
        for entry in self.entries:
            entry.subtree.parent_node = self

    def could_contain_results(self,
                              query_obj,
                              search_radius,
                              entry,
                              d_parent_query):
        """Determines without any d computation if there could be
        objects in the subtree belonging to the result.
        """
        if self.is_root():
            return True
        
        parent_obj = self.parent_entry.obj
        return abs(d_parent_query - entry.distance_to_parent)\
                <= search_radius + entry.radius
            
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(query_obj,
                                          nn.search_radius(),
                                          entry,
                                          d_parent_query):
                d_entry_query = self.d(entry.obj, query_obj)
                entry_dmin = max(d_entry_query - \
                                     entry.radius, 0)
                if entry_dmin <= nn.search_radius():
                    heappush(pr, PrEntry(entry.subtree, entry_dmin, d_entry_query))
                    entry_dmax = d_entry_query + entry.radius
                    if entry_dmax < nn.search_radius():
                        nn.update(None, entry_dmax)


def split(node, entry, d):
    # Union node entries with new entry
    entries = node.entries.copy()
    entries.add(entry)

    # let op be the parent entry of node
    op = node.parent_entry

    # Allocating new node
    m_tree = node.mtree
    new_node = None
    if isinstance(node, InternalNode):
        new_node = InternalNode(mtree=m_tree)
    elif isinstance(node, LeafNode):
        new_node = LeafNode(mtree=m_tree)

    # promoting o1, o2
    o1, o2 = m_tree.promote(entries, node.parent_entry, d)
    # partition all entries
    entries1, entries2 = m_tree.partition(entries, o1, o2, d)

    # Create routing entries for objects o1, o2
    o1_entry = Entry(o1, None, None, node)
    o2_entry = Entry(o2, None, None, new_node)

    # Store partitioned entries into nodes
    node.set_entries_and_parent_entry(entries1, o1_entry)
    new_node.set_entries_and_parent_entry(entries2, o2_entry)

    # Store promoted routing entries into parent node
    if node.is_root():
        new_root_node = InternalNode(node.mtree)

        node.parent_node = new_root_node
        new_node.parent_node = new_root_node

        new_root_node.add_entry(o1_entry)
        new_root_node.add_entry(o2_entry)
        
        m_tree.root = new_root_node
    else:
        parent_node = node.parent_node

        if not parent_node.is_root():
            # parent node has itself a parent, therefore the two entries we add
            # in the parent must have distance_to_parent set appropriately
            o1_entry.distance_to_parent = \
                d(o1_entry.obj, parent_node.parent_entry.obj)
            o2_entry.distance_to_parent = \
                d(o2_entry.obj, parent_node.parent_entry.obj)

        parent_node.remove_entry(op)
        parent_node.add_entry(o1_entry)
        
        if parent_node.is_full():
            split(parent_node, o2_entry, d)
        else:
            parent_node.add_entry(o2_entry)
            new_node.parent_node = parent_node
    assert node.is_root() or node.parent_node
    assert new_node.is_root() or new_node.parent_node
