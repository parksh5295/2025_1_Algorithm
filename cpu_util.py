import time

def busy_loop(intensity_level=0.000001): # Default intensity
    """
    Continuously performs a simple operation to keep the CPU busy.
    A short sleep is included to prevent total CPU monopolization and 
    allow for graceful interruption if needed.
    Adjust intensity_level (sleep duration) based on desired idleness/
    responsiveness. A smaller value (or 0) means more CPU work.
    """
    print(f"Starting CPU busy loop with intensity_level={intensity_level}... Press Ctrl+C to exit.")
    counter = 0
    try:
        while True:
            # Perform a computationally non-trivial but simple task
            _ = 12345.6789 * 98765.4321 / (counter + 1.0) # Ensure float division
            counter += 1
            if counter > 1_000_000_000: # Reset counter to prevent potential overflow/precision issues
                counter = 0
            
            # Only sleep if intensity_level is positive and greater than a very small threshold
            # to effectively disable sleep for intensity_level=0
            if intensity_level > 1e-9: # A very small positive number; effectively sleep only if intensity_level is meant to cause a noticeable pause
                time.sleep(intensity_level)
            # If intensity_level is 0 or very close to 0, no sleep occurs, maximizing CPU usage for this loop.

    except KeyboardInterrupt:
        print("CPU busy loop stopped by user.")
    except Exception as e:
        print(f"An error occurred in busy_loop: {e}")
    finally:
        print("Exiting busy_loop.")

if __name__ == "__main__":
    # To make the CPU usage more aggressive and ensure it's above 1%,
    # we call busy_loop with intensity_level=0.
    # This will make the loop spin more calculations and not sleep at all.
    print("cpu_util.py: Configuring for higher CPU usage (intensity_level=0).")
    busy_loop(intensity_level=0) 