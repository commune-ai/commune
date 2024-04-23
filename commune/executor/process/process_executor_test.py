import concurrent.futures
import time

def calculate_sum_of_squares(start, end):
    return sum(x*x for x in range(start, end+1))

if __name__ == "__main__":
    start_time = time.time()  # Record start time
    print("Python ProcessPoolExecutor Start")
    total_numbers = 10**8
    chunk_size = 10**6  # Adjust chunk size as needed for performance
    total_sum = 0

    # Create a ProcessPoolExecutor with maximum workers
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Divide the range of numbers into chunks
        chunks = [(i, min(i+chunk_size-1, total_numbers)) for i in range(1, total_numbers+1, chunk_size)]

        # Submit jobs for each chunk to be calculated in parallel
        futures = [executor.submit(calculate_sum_of_squares, start, end) for start, end in chunks]

        # Get results from the futures
        for future in concurrent.futures.as_completed(futures):
            total_sum += future.result()

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print("Python ProcessPoolExecutor End")
    print("Elapsed time:", round(elapsed_time, 2), "milliseconds")
