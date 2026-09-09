"""Read research/stack/r2/results/generated.json and print the Q1-Q4 table of Part II."""
import json
import math
import os

P = os.path.join("research", "stack", "r2", "results", "generated.json")


def main():
    d = json.load(open(P))
    print(f"N = {d['N']:,}; bases {sorted(int(b) for b in d['chains'])}; total {d.get('seconds')} s")
    print()
    print("== Q1 (construction = primes) and Q3 (the ends) ==")
    print("base  section                              full  gears   coprime30    members    mismatches  gmpy2(bad mem/bad non)  smallest=p_{k+1}  first twin +  last twin gap")
    nfull = npref = 0
    mism = 0
    for b, ch in d["chains"].items():
        for s in ch["sections"]:
            G = s["sets"]["G"]
            full = s["full"]
            nfull += full
            npref += not full
            mism += s["mismatches"]
            g2 = s["gmpy2_samples"]
            print(f"{b:>4}  [{s['section'][0]:,}, {s['section'][1]:,})".ljust(44)
                  + f"{'full' if full else 'pref'}  {s['gears_used']:6,}  {s['coprime30_numbers']:10,}  {G['size']:10,}  "
                    f"{s['mismatches']:10d}  {g2['members_not_prime']:>3}/{g2['non_members_prime']:<3} of {g2['members']}/{g2['non_members']}   "
                    f"{str(G['smallest_is_first_prime_at_or_above_cut']):>6}  {str(G['first_twin_offset']):>10}  {str(G.get('last_twin_gap_to_end')):>10}")
    print(f"full sections {nfull}, prefixes {npref}, total mismatches {mism}")
    print()
    print("== Q4 (the record) ==")
    print("base  section                              slots        struck-run  twin-free run  equal  ratio")
    for b, ch in d["chains"].items():
        for s in ch["sections"]:
            G = s["sets"]["G"]
            print(f"{b:>4}  [{s['section'][0]:,}, {s['section'][1]:,})".ljust(44)
                  + f"{G['slots']:11,}  {s['struck_slot_record']:10d}  {G['record_slots']:13d}  "
                    f"{str(s['struck_slot_record'] == G['record_slots']):>5}  {G['record_ratio']:.3e}")
    print()
    print("== Q2 (the properties table; full sections only) ==")
    for b, ch in d["chains"].items():
        for s in ch["sections"]:
            if not s["full"]:
                continue
            print(f"-- base {b}, section [{s['section'][0]:,}, {s['section'][1]:,}), "
                  f"{s['coprime30_numbers']:,} numbers coprime to the lower product")
            print("  set   size        pairs@2   missingM   div-by-lower   class0(7/11/13)  triples  (2,4)|(4,2)  "
                  "census maxdev sd(7/11/13)  ties  chi(7/11/13)  mirror hits/chance")
            for k in ("G", "B", "C", "D"):
                if k not in s["sets"]:
                    continue
                S = s["sets"][k]
                c0 = S["class0"]
                dev = S["census_max_dev_sd"]
                ties = S["census_ties"]
                chi = S["chi"]
                print(f"  {k:<4}  {S['size']:10,}  {S['pairs_at_2']:8,}  {str(S.get('missing_coprime')):>9}  "
                      f"{str(S.get('members_divisible_by_lower_gear')):>12}   "
                      f"{c0['7']}/{c0['11']}/{c0['13']}".ljust(17)
                      + f"  {S['triples']:7d}  {S['pattern_2_4']:11d}  "
                      + f"{dev['7']}/{dev['11']}/{dev['13']}".ljust(26)
                      + f"  {ties['7']}/{ties['11']}/{ties['13']}  "
                      + f"{chi['7']}/{chi['11']}/{chi['13']}".ljust(14)
                      + f"  {S['mirror_hits']}/{S['mirror_chance']} = {S['mirror_ratio']}")


if __name__ == "__main__":
    main()
