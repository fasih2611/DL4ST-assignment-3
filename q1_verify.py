"""Q1(b): verify the base-(K+1) message encoding on every allowed neighbourhood."""
import itertools
import numpy as np

K = 8
B = K + 1                                    # base: each count is at most K, so digits 0..K never carry
phi = {-1: 1, 0: B, +1: B**2}                # phi(-1)=1, phi(0)=K+1, phi(+1)=(K+1)^2  -> 1, 9, 81


def decode(s):
    """Recover (slowing, steady, speeding) from the summed message."""
    return s % B, (s // B) % B, s // B**2


triples = [(a, b, c) for a, b, c in itertools.product(range(K + 1), repeat=3)
           if 1 <= a + b + c <= K]          # every allowed (n_slow, n_steady, n_fast)
counts = np.array(triples)                   # shape (164, 3)
sums = counts @ np.array([phi[-1], phi[0], phi[+1]])

assert len(np.unique(sums)) == len(sums), "two neighbourhoods collide"
decoded = np.array([decode(s) for s in sums])
assert (decoded == counts).all(), "decoding failed"
print(f"K={K}: {len(triples)} count triples, {len(np.unique(sums))} distinct sums, "
      f"sum range [{sums.min()}, {sums.max()}], all decoded correctly")

# also check by summing per-neighbour messages over every ordered neighbourhood of size 1..K
n_checked = 0
for n in range(1, K + 1):
    for nbhd in itertools.product((-1, 0, 1), repeat=n) if n <= 6 else []:  # 3^6=729; larger n covered by triples
        s = sum(phi[x] for x in nbhd)
        assert decode(s) == (nbhd.count(-1), nbhd.count(0), nbhd.count(1))
        n_checked += 1
print(f"explicit neighbourhood check: {n_checked} ordered neighbourhoods (size 1-6) OK")

# simpler target: count non-steady neighbours with phi(x) = x^2
for nbhd in itertools.product((-1, 0, 1), repeat=5):
    assert sum(x**2 for x in nbhd) == sum(x != 0 for x in nbhd)
print("phi(x)=x^2 counts non-steady neighbours: OK")
