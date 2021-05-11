
DEFAULT_NODE_SIZE = 5

class MTree:

    def __init__(self, node_size=DEFAULT_NODE_SIZE):
        self.root = None
        self.node_size = node_size

    def add(self, obj):
        self._insert(self.root, obj)
    
    def add_bulk(self, l):
        for i in l:
            self.add(i)

    def _insert(self, node, obj):
        pass

    def _split(self, node, obj):
        pass