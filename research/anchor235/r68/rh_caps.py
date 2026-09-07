"""rh_caps.py -- the sharpened E2 as a function of T = floor((F(M+q') - 2)/q') at every rung, in
three forms: per class with all gears (2 Omega^full(T+1) - 1), the joint two-class form
(max_s Omega^(2)(T; s) - 1, both classes on [0, T]) with gears {5, 7} and with all gears of M.
The corpus rungs have T <= 2; the table says what the pullback buys once F(M+q') passes 3q'.

    uv run python research/anchor235/r68/rh_caps.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh_core import OUT, gears_of, letter_data, next_gear, omega

RUNGS = [19, 23, 29, 31, 37, 41, 43, 47, 53]
TMAX = 9
FILE11 = {T: 2 * T + 1 for T in range(1, TMAX + 1)}
E2_57_UNIFORM = {1: 3, 2: 5, 3: 5, 4: 5, 5: 7, 6: 9, 7: 11, 8: 11, 9: 11}


def main():
    om = json.load(open(os.path.join(OUT, "omega.json")))
    res = {}
    print("T:            " + " ".join(f"{T:3d}" for T in range(1, TMAX + 1)))
    print("file 11:      " + " ".join(f"{FILE11[T]:3d}" for T in range(1, TMAX + 1)))
    print("E2 {5,7} uni: " + " ".join(f"{E2_57_UNIFORM[T]:3d}" for T in range(1, TMAX + 1)))
    for y in RUNGS:
        q = next_gear(y)
        gears = gears_of(y)
        Ld = letter_data(q)
        row = {"q": q, "per_class_full": {}, "joint_57": {}, "joint_full": {}}
        for T in range(1, TMAX + 1):
            row["per_class_full"][T] = 2 * om[f"m{y}"]["omega_full"][str(T + 1)] - 1
            b57 = ball = 0
            for s in (Ld["a"], Ld["b"]):
                slots = [m * q for m in range(T + 1)] + [s + m * q for m in range(T + 1)]
                b57 = max(b57, omega(slots, [5, 7], want_args=False)[0])
                ball = max(ball, omega(slots, gears, want_args=False)[0])
            row["joint_57"][T] = b57 - 1
            row["joint_full"][T] = ball - 1
        res[f"m{y}"] = row
        print(f"m{y} q'={q:2d} per-class full: " + " ".join(f"{row['per_class_full'][T]:3d}" for T in range(1, TMAX + 1)))
        print(f"          joint {{5,7}}:    " + " ".join(f"{row['joint_57'][T]:3d}" for T in range(1, TMAX + 1)))
        print(f"          joint full:     " + " ".join(f"{row['joint_full'][T]:3d}" for T in range(1, TMAX + 1)))
    with open(os.path.join(OUT, "caps.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
