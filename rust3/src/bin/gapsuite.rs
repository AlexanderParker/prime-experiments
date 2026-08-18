//! gapsuite — command line for the slot-frame gap algorithms.
//!
//! ```text
//! gapsuite next <n>              next prime after n
//! gapsuite prev <n>              previous prime before n
//! gapsuite gaps <from> <count>   prime gaps
//! gapsuite twins <from> <count>  twin pairs and their gaps (in slots)
//! gapsuite maxgap <from> <to>    largest prime gap in a range
//! gapsuite ladder <y>            record gaps F of machines up to gear y
//! gapsuite merge <y> <q>         F(M(y) + q) from the old machine alone
//! gapsuite bench <n>             sieve throughput to n
//! ```

use std::env;
use std::time::Instant;

use gearsuite::machine::{f_next, gears_upto, ladder, openings, period, spectrum};
use gearsuite::sieve::{gears_for, next_prime, prev_prime, Segment};
use gearsuite::slot::{hi, lo, slot_of};
use gearsuite::{PrimeGaps, TwinGaps};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
        return;
    }
    let num = |i: usize| -> u64 {
        args.get(i)
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                eprintln!("expected a number in position {i}");
                std::process::exit(2)
            })
    };

    match args[1].as_str() {
        "next" => println!("{}", next_prime(num(2))),
        "prev" => match prev_prime(num(2)) {
            Some(p) => println!("{p}"),
            None => println!("none"),
        },
        "gaps" => {
            let (from, count) = (num(2), num(3) as usize);
            for (p, g) in PrimeGaps::new(from).take(count) {
                println!("{p} {g}");
            }
        }
        "twins" => {
            let (from, count) = (num(2), num(3) as usize);
            let start = slot_of(from.max(5));
            for (k, g) in TwinGaps::new(start).take(count) {
                println!("{k} ({}, {}) {g}", lo(k), hi(k));
            }
        }
        "maxgap" => {
            let (from, to) = (num(2), num(3));
            let mut best = (0u64, 0u64);
            for (p, g) in PrimeGaps::new(from) {
                if p >= to {
                    break;
                }
                if g > best.1 {
                    best = (p, g);
                }
            }
            println!("largest gap {} after prime {}", best.1, best.0);
        }
        "ladder" => {
            let y = num(2);
            println!("{:>5} {:>10} {:>10} {:>12}", "gear", "F(slot)", "F(adj)", "openings");
            for (g, f_slot, f_adj) in ladder(y) {
                let gs = gears_upto(g);
                println!("{:>5} {:>10} {:>10} {:>12}", g, f_slot, f_adj, openings(&gs).len());
            }
        }
        "merge" => {
            let (y, q) = (num(2), num(3));
            let gs = gears_upto(y);
            let ops = openings(&gs);
            let p = period(&gs);
            let t0 = Instant::now();
            let (f, w) = f_next(&ops, p, q);
            let dt = t0.elapsed();
            println!("machine {{5..{y}}}: period {p}, openings {}, F {}", ops.len(), gearsuite::machine::f_max_gap(&ops, p));
            println!("spectrum {:?}", spectrum(&ops, p, 6));
            println!("F(M + {q}) = {f}   [{} kills, {}, span {}]  in {:?}",
                w.kills,
                if w.padded { "padded" } else { "literal" },
                w.span,
                dt);
        }
        "bench" => {
            let n = num(2);
            let top = slot_of(n);
            let gears = gears_for(n);
            let t0 = Instant::now();
            let mut primes = 2usize; // 2 and 3 live outside the frame
            let mut twins = 0usize;
            let mut base = 1u64;
            let seg = 1 << 18;
            while base <= top {
                let len = ((top - base + 1) as usize).min(seg);
                let s = Segment::sieve(base, len, &gears);
                for i in 0..len {
                    if s.lo_prime(i) {
                        primes += 1;
                    }
                    if s.hi_prime(i) {
                        primes += 1;
                    }
                    if s.is_twin(i) {
                        twins += 1;
                    }
                }
                base += len as u64;
            }
            let dt = t0.elapsed();
            println!("to {n}: {primes} primes, {twins} twin pairs in {dt:?}");
            println!("{:.1} M slots/s", top as f64 / dt.as_secs_f64() / 1e6);
        }
        _ => usage(),
    }
}

fn usage() {
    eprintln!(
        "gapsuite — prime and twin gaps in the slot frame

  next <n>              next prime after n
  prev <n>              previous prime before n
  gaps <from> <count>   prime gaps
  twins <from> <count>  twin pairs and gaps (slots)
  maxgap <from> <to>    largest prime gap in a range
  ladder <y>            record gaps F of machines up to gear y
  merge <y> <q>         F(M(y) + q) from the old machine alone
  bench <n>             sieve throughput to n"
    );
}
