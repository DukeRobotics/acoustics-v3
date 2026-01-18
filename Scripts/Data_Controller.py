import os
import time
import Logic as LOGIC

def run_data_controller(
        sampling_freq,
        selected,
        epochs,
        capture_time,
        capture_format,
        capture_base_path,
        test_name,
        capture_close_logic):

    # Parameter validation
    if sampling_freq <= 0:
        raise ValueError(f"sampling_freq must be positive, got {sampling_freq}")
    
    if len(selected) != 4:
        raise ValueError(f"selected must have exactly 4 elements, got {len(selected)}")
    
    if not any(selected):
        raise ValueError("At least one hydrophone must be selected (True in selected array)")
    
    if capture_format not in [".bin", ".csv", "both"]:
        raise ValueError(f"capture_format must be '.bin', '.csv', or 'both', got '{capture_format}'")
    
    if capture_time <= 0:
        raise ValueError(f"capture_time must be positive, got {capture_time}")
    
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")

    # Ensure base directory exists
    if not os.path.exists(capture_base_path):
        os.makedirs(capture_base_path)
    
    # Create test-specific directory with timestamp
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
    test_folder = f"{test_name}_{time_stamp}"
    test_path = os.path.join(capture_base_path, test_folder)
    os.makedirs(test_path)

    print(f"Created test directory: {test_path}")
    print(f"Running {epochs} epochs of data collection...")

    # Initialize Logic connection
    logic = LOGIC.Logic(sampling_freq=sampling_freq)
    logic.print_saleae_status()

    # Run data collection for specified number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        epoch_time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
        epoch_name = f"{test_name}_epoch_{epoch + 1:03d}_{epoch_time_stamp}"
        
        if capture_format == "both":
            capture_path, csv_path = logic.export_binary_and_csv_capture(capture_time, test_path, epoch_name)
            print(f"  Captured: {capture_path} and {csv_path}")
        elif capture_format == ".csv":
            capture_path = logic.start_csv_capture(capture_time, test_path, epoch_name)
            print(f"  Captured: {capture_path}")
        else:
            capture_path = logic.export_binary_capture(capture_time, test_path, epoch_name)
            print(f"  Captured: {capture_path}")
    
    if capture_close_logic:
        logic.kill_logic()
    
    print("=" * 50)
    print(f"Data collection complete!")
    print(f"Total epochs: {epochs}")
    print(f"Data saved to: {test_path}")
    print("=" * 50)

if __name__ == "__main__":
    SAMPLING_FREQ = 781250                          # Sampling frequency in Hz for data acquisition
    SELECTED = [True, True, True, True]             # Which hydrophones to analyze (array of 4 booleans)
    
    EPOCHS = 10                                     # Number of data collection runs to perform
    CAPTURE_TIME = 2                                # Duration of each capture in seconds
    CAPTURE_FORMAT = ".bin"                         # Format for capture: ".bin", ".csv", or "both"
    CAPTURE_BASE_PATH = "Temp_Data"                 # Base directory path for saving captured data
    TEST_NAME = "0"                                  # Name for this test session (timestamp will be appended)
                                                    # Please name test 0, 1, 2, ... 7. to indicate truth.
    CAPTURE_CLOSE_LOGIC = True                      # Whether to close Logic software after all captures

    run_data_controller(
        sampling_freq=SAMPLING_FREQ,
        selected=SELECTED,
        epochs=EPOCHS,
        capture_time=CAPTURE_TIME,
        capture_format=CAPTURE_FORMAT,
        capture_base_path=CAPTURE_BASE_PATH,
        test_name=TEST_NAME,
        capture_close_logic=CAPTURE_CLOSE_LOGIC
    )

