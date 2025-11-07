use std::time::Instant;

struct PrimeFinder {
    known_primes: Vec<u64>,
    prime_roots: Vec<u64>,
}

impl PrimeFinder {
    fn new() -> Self {
        Self {
            known_primes: vec![2, 3],
            prime_roots: vec![2, 2],
        }
    }

    fn last_known_prime(&self) -> u64 {
        self.known_primes[self.known_primes.len() - 1]
    }

    fn calculate_next_known_prime(&mut self) -> u64 {
        let max_range = ((self.last_known_prime() as f64).sqrt().ceil()) as u64;
        let mut gaps = vec![true; (max_range + 1) as usize];
        let mut even_slot_found = false;

        // Check first cycle of remainders to see if all blocked slots are odd
        for &prime_number in &self.known_primes[0..self.known_primes.len() - 1] {
            if prime_number > max_range {
                break;
            }
            let blocked_prime_gap = prime_number - (self.last_known_prime() % prime_number);
            if blocked_prime_gap > max_range {
                break;
            }
            gaps[blocked_prime_gap as usize] = false;
            if blocked_prime_gap % 2 == 0 {
                even_slot_found = true;
            }
        }

        if even_slot_found {
            for &prime_number in &self.known_primes[0..self.known_primes.len() - 1] {
                if prime_number > max_range {
                    break;
                }
                let mut cycle = 1u64;
                loop {
                    let blocked_prime_gap = (prime_number
                        - (self.last_known_prime() % prime_number))
                        + (prime_number * cycle);
                    if blocked_prime_gap > max_range {
                        break;
                    }
                    gaps[blocked_prime_gap as usize] = false;
                    cycle += 1;
                }
            }
        } else {
            let next_prime = self.last_known_prime() + 2;
            self.known_primes.push(next_prime);
            return next_prime;
        }

        let mut check_empty_slot = 2;
        while check_empty_slot <= max_range && !gaps[check_empty_slot as usize] {
            check_empty_slot += 2;
        }

        let next_prime = self.last_known_prime() + check_empty_slot;
        self.known_primes.push(next_prime);
        self.prime_roots
            .push(((next_prime as f64).sqrt().ceil()) as u64);
        next_prime
    }

    fn next_prime_any(&mut self, n: u64) -> u64 {
        // Return nearest highest prime to n if it's a known prime
        for &p in &self.known_primes {
            if n < p {
                return p;
            }
        }

        let max_range = ((n as f64).sqrt().ceil()) as u64;

        // Ensure we know all primes to max_range
        while self.last_known_prime() <= max_range {
            self.calculate_next_known_prime();
        }

        let mut gaps = vec![true; (max_range + 1) as usize];
        let mut even_slot_found = false;

        for &prime_number in &self.known_primes[0..self.known_primes.len() - 1] {
            if prime_number > max_range {
                break;
            }
            let blocked_prime_gap = (prime_number - (n % prime_number)) % prime_number;
            gaps[blocked_prime_gap as usize] = false;
            if blocked_prime_gap % 2 == 0 {
                even_slot_found = true;
            }
        }

        if even_slot_found == true {
            for &prime_number in &self.known_primes[0..self.known_primes.len() - 1] {
                if prime_number > max_range {
                    break;
                }
                let mut cycle = 1u64;
                loop {
                    let blocked_prime_gap =
                        (prime_number - (n % prime_number)) % prime_number + (prime_number * cycle);
                    if blocked_prime_gap > max_range {
                        break;
                    }
                    gaps[blocked_prime_gap as usize] = false;
                    cycle += 1;
                }
            }
        }

        let mut check_empty_slot = if n % 2 == 0 { 1 } else { 2 };
        while check_empty_slot <= max_range && !gaps[check_empty_slot as usize] {
            check_empty_slot += 2;
        }

        n + check_empty_slot
    }
}

fn main() {
    let mut finder = PrimeFinder::new();

    let test_n = 7213393222u64;
    //let test_n = 2305843009213693950; // Next prime is 2305843009213693951 (m9)
    let start = Instant::now();
    let our_guess = finder.next_prime_any(test_n);
    let our_time = start.elapsed();
    println!(
        "Our result: {} (time: {:.6} seconds)",
        our_guess,
        our_time.as_secs_f64()
    );
}

/*

int | mask  | Cycle mask
Cumulative  | 00
2   = 10    | 10
------------------------
Combined    | 10
New         | 01 (3)
Cumulative  | 11 (2, 3)

Cumulative: | 110
2   = 10    | 101
3   = 100   | 100
------------------------
Combined:   | 111
Progressive | 101110

*/
