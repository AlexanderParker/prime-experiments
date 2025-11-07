use malachite::base::num::arithmetic::traits::SqrtRem;
use malachite::Natural;
use std::env;
use std::fs;
use std::sync::Mutex;
use std::time::Instant;

lazy_static::lazy_static! {
    static ref KNOWN_PRIMES: Mutex<Vec<Natural>> = Mutex::new(vec![Natural::from(2u32), Natural::from(3u32)]);
}

fn sqrt_ceil(n: &Natural) -> Natural {
    let (sqrt, remainder) = n.sqrt_rem();
    if remainder == 0u32 {
        sqrt
    } else {
        sqrt + Natural::from(1u32)
    }
}

fn populate_primes_up_to_segmented(limit: &Natural) {
    println!("Using optimised prime generation up to limit");
    let start_time = Instant::now();

    if *limit < Natural::from(2u32) {
        return;
    }

    let mut primes = vec![Natural::from(2u32)];
    let mut candidate = Natural::from(3u32);

    while candidate <= *limit {
        if is_prime_trial_division(&candidate, &primes) {
            primes.push(candidate.clone());
        }
        candidate += Natural::from(2u32);

        if primes.len() % 10000 == 0 {
            println!(
                "Found a batch of 10000 primes",
                primes.len()
            );
        }
    }

    {
        let mut known_primes = KNOWN_PRIMES.lock().unwrap();
        *known_primes = primes;
    }

    let prime_count = {
        let known_primes = KNOWN_PRIMES.lock().unwrap();
        known_primes.len()
    };

    println!(
        "Prime generation completed in {:.2}s, found {} primes",
        start_time.elapsed().as_secs_f64(),
        prime_count
    );
}

fn populate_primes_sequential_from(start: &Natural, limit: &Natural) {
    // Reset known primes to start from current position
    let mut current = start.clone();

    // Make sure current is odd
    if &current % Natural::from(2u32) == Natural::from(0u32) {
        current += Natural::from(1u32);
    }

    let mut primes = vec![Natural::from(2u32)];

    // Add odd numbers and check primality
    while current <= *limit {
        if is_prime_trial_division(&current, &primes) {
            primes.push(current.clone());
        }
        current += Natural::from(2u32);

        if primes.len() % 1000 == 0 {
            println!(
                "Sequential: found {} primes, current: {}",
                primes.len(),
                current
            );
        }
    }

    {
        let mut known_primes = KNOWN_PRIMES.lock().unwrap();
        *known_primes = primes;
    }
}

fn is_prime_trial_division(n: &Natural, known_primes: &[Natural]) -> bool {
    let sqrt_n = sqrt_ceil(n);

    for prime in known_primes {
        if prime > &sqrt_n {
            break;
        }
        if n % prime == Natural::from(0u32) {
            return false;
        }
    }
    true
}

fn populate_primes_sequential(limit: &Natural) {
    let mut last_progress_time = Instant::now();

    loop {
        let (last_prime, prime_count) = {
            let known_primes = KNOWN_PRIMES.lock().unwrap();
            (known_primes.last().unwrap().clone(), known_primes.len())
        };

        if last_prime > *limit {
            println!("Found all {} primes up to limit", prime_count);
            break;
        }

        let elapsed = last_progress_time.elapsed();
        if elapsed.as_secs() >= 2 {
            println!("Found {} primes", prime_count);
            last_progress_time = Instant::now();
        }

        get_next_prime();
    }
}

fn get_next_prime_gap() -> Natural {
    let known_primes = KNOWN_PRIMES.lock().unwrap();
    let p1 = known_primes.last().unwrap().clone();

    let p1_sqrt = sqrt_ceil(&p1);

    let prime_divisors: Vec<Natural> = known_primes
        .iter()
        .filter(|d| *d <= &p1_sqrt)
        .cloned()
        .collect();

    let blocked_gaps: Vec<Natural> = prime_divisors.iter().map(|p| (p - (&p1 % p)) % p).collect();

    if !blocked_gaps.contains(&Natural::from(2u32)) {
        return Natural::from(2u32);
    }

    let mut gap_buckets = blocked_gaps.clone();
    let mut test_gap = Natural::from(2u32);

    loop {
        let mut test_gap_blocked = false;
        test_gap += Natural::from(2u32);

        if gap_buckets.contains(&test_gap) {
            continue;
        } else {
            let mut prime_divisors_index = 1;

            while prime_divisors_index < prime_divisors.len()
                && prime_divisors[prime_divisors_index] < test_gap
            {
                while gap_buckets[prime_divisors_index] < test_gap {
                    gap_buckets[prime_divisors_index] += &prime_divisors[prime_divisors_index];
                    if gap_buckets[prime_divisors_index] == test_gap {
                        test_gap_blocked = true;
                        break;
                    }
                }

                if test_gap_blocked {
                    break;
                }

                prime_divisors_index += 1;

                if prime_divisors_index >= prime_divisors.len() {
                    return test_gap;
                }
            }
        }

        if !test_gap_blocked {
            return test_gap;
        }
    }
}

fn get_next_prime() -> Natural {
    let gap = get_next_prime_gap();
    let mut known_primes = KNOWN_PRIMES.lock().unwrap();
    let next_prime = known_primes.last().unwrap() + &gap;
    known_primes.push(next_prime.clone());
    next_prime
}

fn get_prime_after(n: &Natural) -> Natural {
    println!("Checking if already known...");
    {
        let known_primes = KNOWN_PRIMES.lock().unwrap();
        if n < known_primes.last().unwrap() {
            for p in known_primes.iter() {
                if p > n {
                    return p.clone();
                }
            }
        }
    }

    println!("Calculating sqrt - this may take a while for very large numbers...");
    let start_sqrt = Instant::now();
    let n_sqrt = sqrt_ceil(n);
    println!(
        "Square root calculated in {:.2}s",
        start_sqrt.elapsed().as_secs_f64()
    );

    println!("Finding primes up to sqrt");

    populate_primes_up_to_segmented(&n_sqrt);

    println!("Starting gap search...");

    let known_primes = KNOWN_PRIMES.lock().unwrap();
    let prime_divisors: Vec<Natural> = known_primes
        .iter()
        .filter(|d| *d <= &n_sqrt)
        .cloned()
        .collect();

    let blocked_gaps: Vec<Natural> = prime_divisors.iter().map(|p| (p - (n % p)) % p).collect();

    let first_gap = if (n % Natural::from(2u32)) == Natural::from(0u32) {
        Natural::from(1u32)
    } else {
        Natural::from(2u32)
    };

    if !blocked_gaps.contains(&first_gap) {
        return n + &first_gap;
    }

    let mut gap_buckets = blocked_gaps.clone();
    let mut test_gap = first_gap;
    let mut gaps_checked = 0;
    let mut last_gap_progress_time = Instant::now();

    loop {
        let mut test_gap_blocked = false;
        test_gap += Natural::from(2u32);
        gaps_checked += 1;

        if gaps_checked % 10000 == 0 {
            let elapsed = last_gap_progress_time.elapsed();
            if elapsed.as_secs() >= 3 {
                println!("Checked {} gaps, current gap: {}", gaps_checked, test_gap);
                last_gap_progress_time = Instant::now();
            }
        }

        if gap_buckets.contains(&test_gap) {
            continue;
        } else {
            let mut prime_divisors_index = 1;

            while prime_divisors_index < prime_divisors.len()
                && prime_divisors[prime_divisors_index] < test_gap
            {
                while gap_buckets[prime_divisors_index] < test_gap {
                    gap_buckets[prime_divisors_index] += &prime_divisors[prime_divisors_index];
                    if gap_buckets[prime_divisors_index] == test_gap {
                        test_gap_blocked = true;
                        break;
                    }
                }

                if test_gap_blocked {
                    break;
                }

                prime_divisors_index += 1;

                if prime_divisors_index >= prime_divisors.len() {
                    println!("Found prime after {} gaps", gaps_checked);
                    return n + &test_gap;
                }
            }
        }

        if !test_gap_blocked {
            println!("Found prime after {} gaps", gaps_checked);
            return n + &test_gap;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let (number, output_to_file, output_filename) = if args.len() > 1 {
        let filename = &args[1];

        println!("Reading file: {}", filename);
        let start_read = Instant::now();
        let content = fs::read_to_string(filename).expect("Failed to read file");
        println!(
            "File read in {:.2}s ({} bytes)",
            start_read.elapsed().as_secs_f64(),
            content.len()
        );

        println!("Parsing number...");
        let start_parse = Instant::now();
        let cleaned_content = content
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>();

        if cleaned_content.is_empty() {
            panic!("No digits found in file");
        }

        let number = cleaned_content
            .parse::<Natural>()
            .expect("Failed to parse number from file");
        println!(
            "Number parsed in {:.2}s ({} digits)",
            start_parse.elapsed().as_secs_f64(),
            cleaned_content.len()
        );

        let output_filename = format!("{}.out.txt", filename);
        (number, true, output_filename)
    } else {
        (Natural::from(7213393222u64), false, String::new())
    };

    println!("Starting prime search...");
    let start_search = Instant::now();
    let result = get_prime_after(&number);
    println!(
        "Prime search completed in {:.2}s",
        start_search.elapsed().as_secs_f64()
    );

    if output_to_file {
        println!("Writing result to file...");
        let start_write = Instant::now();
        fs::write(&output_filename, result.to_string()).expect("Failed to write output file");
        println!(
            "Result written to {} in {:.2}s",
            output_filename,
            start_write.elapsed().as_secs_f64()
        );
    } else {
        println!("{}", result);
    }
}
