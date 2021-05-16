import mtree
from contextlib import contextmanager
from time import time
from random import sample


@contextmanager
def timeit(db):
    start = time()
    yield
    end = time()
    print(f"Time elapsed: {end - start}s")
    print(f"Distance function called: {db.dcall_counter} times")
    

def main():
    db = mtree.MTree(node_size=10, dimensions=4)

    print("Indexing database")
    with timeit(db):
        db.seed_db(4, 10000, 0, 1000)

    query = tuple(sample(range(0, 1000), 4))

    print("Random range query with radius 30")
    with timeit(db):
        db.range_search(query, 30)
    
    print("Sequential")
    with timeit(db):
        db.sequential_range_search(query, 30)
    
    print("Random kNN query with k 15")
    with timeit(db):
        db.knn(query, 15)
    
    print("Sequential")
    with timeit(db):
        db.sequential_knn(query, 15)


if __name__ == '__main__':
    main()