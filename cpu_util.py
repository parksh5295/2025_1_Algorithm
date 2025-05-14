import time

def busy_loop(intensity_level=0.00000001):
    """
    Continuously performs a simple operation to keep the CPU busy.
    A short sleep is included to prevent total CPU monopolization and 
    allow for graceful interruption if needed, though for pure CPU busy work
    this sleep might be counterproductive if the goal is max CPU usage.
    Adjust intensity_level (sleep duration) based on desired idleness/
    responsiveness. A smaller value means more CPU work.
    """
    print("Starting CPU busy loop... Press Ctrl+C to exit.")
    counter = 0
    try:
        while True:
            # Perform a computationally non-trivial but simple task
            # Example: a few floating point operations or string manipulations
            # This is a placeholder; more complex math can be added.
            _ = 12345.6789 * 98765.4321 / (counter + 1) 
            counter += 1
            if counter > 1_000_000_000: # Reset counter to prevent overflow if run for extreme durations
                counter = 0
            
            # Optional: sleep for a very short period to yield CPU slightly
            # This can make the system more responsive if this script is too aggressive.
            # If the goal is to maximize CPU usage to prevent session closure due to *any* idleness,
            # this sleep might be removed or made extremely small.
            # For preventing session closure, continuous activity is key.
            time.sleep(intensity_level) 

    except KeyboardInterrupt:
        print("\nCPU busy loop stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You can adjust the intensity_level here.
    # A very small number (e.g., 0.000001 or even 0) will make the CPU busier.
    # A larger number (e.g., 0.1) will make it less busy.
    busy_loop(intensity_level=0.00000001) 