import os
import time
import Logic as LOGIC
import Hydrophone_Array

def run_controller(        
        sampling_freq,
        selected,
        capture,
        capture_time,
        capture_format,
        capture_base_path,
        capture_prefix,
        capture_close_logic,
        historical_path,
        plot_envelope,
        plot_gcc):

    # Parameter validation
    if sampling_freq <= 0:
        raise ValueError(f"sampling_freq must be positive, got {sampling_freq}")
    
    if len(selected) != 4:
        raise ValueError(f"selected must have exactly 4 elements, got {len(selected)}")
    
    if not any(selected):
        raise ValueError("At least one hydrophone must be selected (True in selected array)")
    
    if capture_format not in [".bin", ".csv", "both"]:
        raise ValueError(f"capture_format must be '.bin', '.csv', or 'both', got '{capture_format}'")
    
    if capture and capture_time <= 0:
        raise ValueError(f"capture_time must be positive when capturing, got {capture_time}")

    if capture:
        # Ensure base directory exists
        if not os.path.exists(capture_base_path):
            os.makedirs(capture_base_path)
        
        time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
        folder = capture_prefix + time_stamp

        base_path = os.path.join(capture_base_path, folder)
        os.makedirs(base_path)

        logic = LOGIC.Logic(sampling_freq=sampling_freq)
        logic.print_saleae_status()

        if capture_format == "both":
            capture_path, csv_path = logic.export_binary_and_csv_capture(capture_time, base_path)
        elif capture_format == ".csv":
            capture_path = logic.start_csv_capture(capture_time, base_path)
        else:
            capture_path = logic.export_binary_capture(capture_time, base_path)
        
        if capture_close_logic:
            logic.kill_logic()

    elif historical_path != "":
        if not os.path.exists(historical_path):
            raise FileNotFoundError(f"Historical path does not exist: {historical_path}")
        if not os.path.isfile(historical_path):
            raise ValueError(f"Historical path must be a file, not a directory: {historical_path}")
        capture_path = historical_path
    else:
        raise ValueError("Either capture must be True or historical_path must be provided")

    hydrophone_array = Hydrophone_Array.HydrophoneArray(sampling_freq=sampling_freq)
    hydrophone_array.load_from_path(capture_path)

    hydrophone_array.estimate_selected_by_envelope(selected)
    hydrophone_array.print_envelope_toas()
    if plot_envelope:
        hydrophone_array.plot_selected_envelope(selected, show_frequency_domain=True)
    print("=" * 30)

    hydrophone_array.estimate_selected_by_gcc(selected, use_multi_reference=True)
    hydrophone_array.print_gcc_tdoa(selected)
    if plot_gcc:
        hydrophone_array.plot_selected_gcc(selected)
    print("=" * 30)

if __name__ == "__main__":
    LOGIC_PATH = ""                                 # LOGIC path. Leave Blank for default
    SAMPLING_FREQ = 781250                          # Sampling frequency in Hz for data acquisition
    SELECTED = [True, True, True, True]             # Which hydrophones to analyze (array of 4 booleans)
    
    CAPTURE = True                                  # If True, capture new data; if False, use historical data
    CAPTURE_TIME = 2                                # Duration of capture in seconds
    CAPTURE_FORMAT = ".bin"                         # Format for capture: ".bin", ".csv", or "both"
    CAPTURE_BASE_PATH = "Temp_Data"                 # Base directory path for saving captured data
    CAPTURE_PREFIX = ""                             # Prefix for capture folder name (timestamp will be appended)
    CAPTURE_CLOSE_LOGIC = True                      # Whether to close Logic software after capture
    
    HISTORICAL_PATH = ""                            # Path to historical data file (used when CAPTURE is False)
    
    PLOT_ENVELOPE = False                           # Whether to plot envelope analysis results
    PLOT_GCC = False                                # Whether to plot GCC (Generalized Cross-Correlation) results

    run_controller(
        sampling_freq=SAMPLING_FREQ,
        selected=SELECTED,
        capture=CAPTURE,
        capture_time=CAPTURE_TIME,
        capture_format=CAPTURE_FORMAT,
        capture_base_path=CAPTURE_BASE_PATH,
        capture_prefix=CAPTURE_PREFIX,
        capture_close_logic=CAPTURE_CLOSE_LOGIC,
        historical_path=HISTORICAL_PATH,
        plot_envelope=PLOT_ENVELOPE,
        plot_gcc=PLOT_GCC
    )