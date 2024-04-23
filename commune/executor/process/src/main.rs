use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let start_time = Instant::now();
    println!("Rayon Start");
    let nums: Vec<u128> = (1..=100_000_000).collect();
    let _nums_squared: u128 = nums
        .par_iter()
        .map(|&x| x * x)
        .sum();
    
    println!("Rayon End");
    let end_time = Instant::now();
    let elapsed_time = end_time - start_time;
    println!("Elapsed time: {} milliseconds", elapsed_time.subsec_millis());
    print("Elapsed time:", elapsed_time, "seconds")
}