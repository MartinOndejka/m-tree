from mtree import MTree


def main():
    d = lambda a, b: abs(a - b)

    db = MTree(d=d, node_size=10)
    
    db.add_bulk([62, 10, 30, 44, 46, 76, 58, 52, 45, 73, 66, 23, 60, 82, 61, 90, 81, 95, 20, 74, 22, 37, 31, 78, 88, 92, 6, 55, 24, 63, 83, 2, 68, 3, 56, 69, 38, 1, 77, 64])

    assert list(db.knn(19,5)) == [20, 22, 23, 24, 10]

    print(db.range_search(20, 10))

if __name__ == '__main__':
    main()
