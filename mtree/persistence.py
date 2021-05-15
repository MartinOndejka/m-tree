import pickle
from .mtree import euclidean_distance


def save_tree(mtree, file_path):
    mtree.d = None

    with open(file_path, "wb") as f:
        pickle.dump(mtree, f)


def load_tree(file_path, distance_function=euclidean_distance):
    with open(file_path, "rb") as f:
        mtree = pickle.load(f)
    
    mtree.d = distance_function

    return mtree