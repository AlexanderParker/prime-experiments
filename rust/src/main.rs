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

    fn last_known_prime_root(&self) -> u64 {
        self.prime_roots[self.prime_roots.len() - 1]
    }

    fn calculate_next_known_prime(&mut self) -> u64 {
        let mut gaps = vec![true; (self.last_known_prime_root() + 1) as usize];
        
        for &prime_number in &self.known_primes[0..self.known_primes.len() - 1] {
            if self.last_known_prime() * self.last_known_prime() > prime_number {
                break;
            }
            let mut cycle = 0u64;
            loop {
                let blocked_prime_gap = (prime_number - (self.last_known_prime() % prime_number))
                    % prime_number
                    + (prime_number * cycle);
                if self.last_known_prime() * self.last_known_prime() > blocked_prime_gap {
                    break;
                }
                gaps[blocked_prime_gap as usize] = false;
                cycle += 1;
            }
        }

        let mut check_empty_slot = 2;
        while !gaps[check_empty_slot as usize] {
            check_empty_slot += 2;
        }
        
        let next_prime = self.last_known_prime() + check_empty_slot;
        self.known_primes.push(next_prime);
        self.prime_roots.push(((next_prime as f64).sqrt().ceil()) as u64);
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
        
        for &prime_number in &self.known_primes[0..self.known_primes.len() - 1] {
            
            if prime_number > max_range {
                break;
            }
            let mut cycle = 0u64;
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

        let mut check_empty_slot = if n % 2 == 0 { 1 } else { 2 };
        while check_empty_slot <= max_range && !gaps[check_empty_slot as usize] {
            check_empty_slot += 2;
        }
        
        n + check_empty_slot
    }
}

fn main() {
    let mut finder = PrimeFinder::new();
    //let test_n = 543210; 

    let test_n = 2305843009213693950; // Next prime is 2305843009213693951 (m9)
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
int | mask 
1   = 0
2   = 01
3   = 001
*/