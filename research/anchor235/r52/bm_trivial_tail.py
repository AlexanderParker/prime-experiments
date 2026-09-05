"""Manager check (2026-09-06): the block budget eta = sum_g (4/g^2) rho_g^2 must stay below 1.
Head gears (g < G) have rho = 1 exactly where the lower period times g fits the window.
Tail with the TRIVIAL bound rho_g <= 1/delta_{<g} (every strike of g lands on a survivor).
Prints, per starting gear G, the exact head, the room 1 - head, the trivial tail and the ratio."""
def primes(n):
    s = bytearray([1]) * (n + 1); s[0] = s[1] = 0
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n + 1) if s[i]]
P = [p for p in primes(2_000_000) if p >= 5]
delta = 1.0; cum = []
for p in P:
    cum.append((p, 4 / p ** 2, 4 / p ** 2 / delta ** 2, delta)); delta *= 1 - 2 / p
head = lambda G: sum(t for p, t, _, _ in cum if p < G)
tail = lambda G: sum(t for p, _, t, _ in cum if p >= G)
for G in (19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101):
    d = [d for p, _, _, d in cum if p == G][0]
    print(f"G={G:3d} head={head(G):.4f} room={1-head(G):.4f} tail_trivial={tail(G):.4f} ratio={tail(G)/(1-head(G)):.2f} delta_<G={d:.4f}")
