"""One-off (round 25): the scan-free dictionary vs Mechanic's scan-derived
4-tuple censuses (research/data/gap_tuples_{23,29}_4.csv).  Any difference is
a cyclic-seam question - the CRT dictionary is seam-free by construction
(it counts k mod P), a linear scan is not."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crt_dict, scanfree_dict


def main():
    for y in (23, 29):
        M = set()
        with open('research/data/gap_tuples_%d_4.csv' % y) as f:
            r = csv.reader(f); hdr = next(r)
            for row in r:
                M.add(tuple(int(x) for x in row[:4]))
        D, Fj, und = scanfree_dict.build(y, 4, 5,
                                         cap=crt_dict.KNOWN_F[y] + 20,
                                         verbose=False)
        S = set(D[4])
        print('m%d  header %s' % (y, hdr))
        print('  mechanic %d   scan-free %d' % (len(M), len(S)))
        print('  in scan-free NOT in mechanic: %s' % sorted(S - M))
        print('  in mechanic NOT in scan-free: %s' % sorted(M - S)[:10],
              flush=True)


if __name__ == "__main__":
    main()
