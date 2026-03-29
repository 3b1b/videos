import random


def compute_join_sequence(n, seed):
    random.seed(seed)

    parent = list(range(2 * n))
    rank = [0] * (2 * n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    for i in range(n):
        union(2 * i, 2 * i + 1)

    loose = list(range(2 * n))
    random.shuffle(loose)

    seq = []
    lc = 0
    while len(loose) >= 2:
        a = loose.pop(random.randrange(len(loose)))
        b = loose.pop(random.randrange(len(loose)))
        fl = find(a) == find(b)
        union(a, b)
        if fl:
            lc += 1
        seq.append((a, b, fl, lc))
    return seq


def ensure_box_self_loop(seq, n_box_joins):
    for i in range(len(seq)):
        if seq[i][2]:
            if i >= n_box_joins - 1:
                break
            loop_entry = seq[i]
            seq = seq[:]
            seq.pop(i)
            seq.insert(n_box_joins - 1, loop_entry)
            break
    return seq


def first_loop_index_in_2d(seq, n_box_joins):
    for i in range(n_box_joins, len(seq)):
        if seq[i][2]:
            return i - n_box_joins
    return None


if __name__ == "__main__":
    n = 10
    n_box_joins = 4
    search_range = 1000  # adjust as needed; higher values will take more time but may result in better seeds

    print(
        f"Searching seeds 0–{search_range - 1} for n={n}, n_box_joins={n_box_joins}\n"
    )
    print(f"{'Seed':>6}  {'First 2D loop at join':>22}  {'2D loop sequence'}")
    print("-" * 60)

    for seed in range(search_range):
        seq = compute_join_sequence(n, seed)
        seq = ensure_box_self_loop(seq, n_box_joins)
        idx = first_loop_index_in_2d(seq, n_box_joins)
        if idx is not None and idx <= 2:
            loop_positions = [
                i - n_box_joins for i in range(n_box_joins, len(seq)) if seq[i][2]
            ]
            print(f"{seed:>6}  {'join ' + str(idx):>22}  {loop_positions}")
