from mtree import MTree


def main():
    db = MTree(node_size=2)

    db.add_bulk(
        [
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
        ]
    )

    print(db)

if __name__ == '__main__':
    main()
