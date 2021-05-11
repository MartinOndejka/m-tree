from mtree import MTree, mtree


def main():
    tree = MTree()

    tree.add(1)

    print(
        mtree.euclidean_distance((-7, -4), (17, 6.5))
    )

if __name__ == '__main__':
    main()
