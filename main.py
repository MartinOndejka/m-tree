from mtree import MTree


def main():
    d = lambda a, b: abs(a - b)

    db = MTree(d=d, max_node_size=3)

    db.add_all([11, 13, 3, 27, 24, 1, 36, 6, 10, 2])

    print(
        list(db.search(20,3))
    )


if __name__ == '__main__':
    main()
