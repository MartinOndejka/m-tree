import pickle
from .mtree import euclidean_distance


class LoadError(pickle.UnpicklingError):
    pass


def save_tree(mtree, file_path):
    mtree.d = None

    with open(file_path, "wb") as f:
        pickle.dump(mtree, f)


def load_tree(file_path, distance_function=euclidean_distance):
    with open(file_path, "rb") as f:
        try:
            mtree = pickle.load(f)
        except pickle.UnpicklingError:
            raise LoadError

    mtree.d = distance_function

    return mtree
