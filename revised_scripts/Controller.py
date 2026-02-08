"""Controller module for hydrophone data acquisition and analysis."""
import os
import time
from logic import logic
from hydrophones import hydrophone_array

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
        plot,
        plot_option
        ):
    """Run the hydrophone controller for data capture and analysis.

    Args:
        sampling_freq: Sampling frequency in Hz
        selected: List of 4 booleans indicating which hydrophones to use
        capture: Whether to capture new data
        capture_time: Duration of capture in seconds
        capture_format: Format for capture ('.bin', '.csv', or 'both')
        capture_base_path: Base directory for saving captured data
        capture_prefix: Prefix for capture folder name
        capture_close_logic: Whether to close Logic software after capture
        historical_path: Path to historical data file
        plot: Whether to plot the data
        plot_option: Type of plot to display
    """
    # Parameter validation
    if sampling_freq <= 0:
        raise ValueError(
            f"sampling_freq must be positive, got {sampling_freq}"
        )

    if len(selected) != 4:
        raise ValueError(
            f"selected must have exactly 4 elements, got {len(selected)}"
        )

    if not any(selected):
        msg = "At least one hydrophone must be selected"
        raise ValueError(msg)

    if capture_format not in [".bin", ".csv", "both"]:
        msg = f"capture_format must be '.bin', '.csv', or 'both', got '{capture_format}'"
        raise ValueError(msg)

    if capture and capture_time <= 0:
        msg = f"capture_time must be positive when capturing, got {capture_time}"
        raise ValueError(msg)

    if capture:
        # Ensure base directory exists
        if not os.path.exists(capture_base_path):
            os.makedirs(capture_base_path)

        time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
        folder = capture_prefix + time_stamp

        base_path = os.path.join(capture_base_path, folder)
        os.makedirs(base_path)

        logic_interface = logic.Logic(sampling_freq=sampling_freq)
        logic_interface.print_saleae_status()

        if capture_format == "both":
            capture_path, _ = logic_interface.export_binary_and_csv_capture(
                capture_time, base_path
            )
        elif capture_format == ".csv":
            capture_path = logic_interface.start_csv_capture(
                capture_time, base_path
            )
        else:
            capture_path = logic_interface.export_binary_capture(
                capture_time, base_path
            )

        if capture_close_logic:
            logic_interface.kill_logic()

    elif historical_path != "":
        if not os.path.exists(historical_path):
            msg = f"Historical path does not exist: {historical_path}"
            raise FileNotFoundError(msg)
        if not os.path.isfile(historical_path):
            msg = f"Historical path must be a file: {historical_path}"
            raise ValueError(msg)
        capture_path = historical_path
    else:
        msg = "Either capture must be True or historical_path must be provided"
        raise ValueError(msg)

    array = hydrophone_array.HydrophoneArray(
        sampling_freq=sampling_freq, selected=selected
    )
    array.load_from_path(capture_path)

    if plot:
        array.plot_hydrophones(option=plot_option)

if __name__ == "__main__":
    # LOGIC path. Leave Blank for default
    LOGIC_PATH = ""
    # Sampling frequency in Hz for data acquisition
    SAMPLING_FREQ = 781250
    # Which hydrophones to analyze (array of 4 booleans)
    SELECTED = [True, True, True, True]

    # If True, capture new data; if False, use historical data
    CAPTURE = False
    # Duration of capture in seconds
    CAPTURE_TIME = 2
    # Format for capture: ".bin", ".csv", or "both"
    CAPTURE_FORMAT = ".bin"
    # Base directory path for saving captured data
    CAPTURE_BASE_PATH = "Temp_Data"
    # Prefix for capture folder name (timestamp will be appended)
    CAPTURE_PREFIX = ""
    # Whether to close Logic software after capture
    CAPTURE_CLOSE_LOGIC = True

    # Path to historical data file (used when CAPTURE is False)
    HISTORICAL_PATH = "Temp_Data/2026-01-19--15-35-37/TEMP.bin"

    # If True, plot
    PLOT = True
    # Options: signal, filtered_signal, frequency, filtered_frequency
    PLOT_OPTION = "filtered_frequency"

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
        plot=PLOT,
        plot_option=PLOT_OPTION,
    )
