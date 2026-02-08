"""Controller module for hydrophone data acquisition and analysis."""
import os
import time
from logic import logic
from hydrophones import hydrophone_array
from analyzers import TOAEnvelopeAnalyzer


def run_controller(
        hydrophone_array,
        analyzers=None
        ):
    """Run analysis on hydrophone array.

    Args:
        hydrophone_array: HydrophoneArray with loaded data
        analyzers: Optional analyzer instance or list of analyzer instances to run

    Returns:
        Single analysis result dict if one analyzer, or list of dicts if multiple
    """
    # Handle single analyzer or list
    if analyzers is None:
        return None
    
    results = []
    for analyzer in analyzers:
        print(f"\n{'='*60}")
        analysis_result = analyzer.analyze_array(hydrophone_array)

        # Print results
        print(f"\n{analysis_result['analyzer']}")
        print(f"Center Frequency: {analysis_result['center_frequency']:.2f} Hz")
        print(f"\nRelative TOA (ref: hydrophone {analysis_result['reference_idx']}):")
        for i, rel_time in enumerate(analysis_result['relative_times']):
            print(f"  Hydrophone {i}: {rel_time*1e6:8.2f} μs")
        
        results.append(analysis_result)

    return results


def capture_data(
        sampling_freq,
        capture_time,
        capture_format,
        output_dir,
        close_logic_after=True
        ):
    """Capture new data from Logic hardware.

    Args:
        sampling_freq: Sampling frequency in Hz
        capture_time: Duration of capture in seconds
        capture_format: '.bin', '.csv', or 'both'
        output_dir: Directory to save capture files
        close_logic_after: Whether to close Logic software after capture

    Returns:
        Path to captured data file
    """
    os.makedirs(output_dir, exist_ok=True)

    logic_interface = logic.Logic(sampling_freq=sampling_freq)
    logic_interface.print_saleae_status()

    if capture_format == "both":
        capture_path, _ = logic_interface.export_binary_and_csv_capture(
            capture_time, output_dir
        )
    elif capture_format == ".csv":
        capture_path = logic_interface.start_csv_capture(
            capture_time, output_dir
        )
    else:
        capture_path = logic_interface.export_binary_capture(
            capture_time, output_dir
        )

    if close_logic_after:
        logic_interface.kill_logic()

    return capture_path


def load_hydrophone_data(
        data_path,
        sampling_freq,
        selected_hydrophones,
        plot_data
        ):
    """Load hydrophone data from file.

    Args:
        data_path: Path to data file (.bin or .csv)
        sampling_freq: Sampling frequency in Hz
        selected_hydrophones: List of 4 bools indicating which to load

    Returns:
        HydrophoneArray with loaded data
    """
    array = hydrophone_array.HydrophoneArray(
        sampling_freq=sampling_freq,
        selected=selected_hydrophones
    )
    array.load_from_path(data_path)
    
    # Plot raw data if requested
    if plot_data:
        array.plot_hydrophones()
    return array


if __name__ == "__main__":
    # Whether to capture new data from Logic hardware (True) or use existing file (False)
    CAPTURE_NEW_DATA = True

    # Path to existing data file (used when CAPTURE_NEW_DATA = False)
    DATA_FILE = ""

    # Duration of capture in seconds (only used if CAPTURE_NEW_DATA = True)
    CAPTURE_TIME = 2

    # Format for capture: '.bin', '.csv', or 'both' (only used if CAPTURE_NEW_DATA = True)
    CAPTURE_FORMAT = ".bin"

    # Output directory for captured data (only used if CAPTURE_NEW_DATA = True)
    CAPTURE_OUTPUT = "Temp_Data"

    # Sampling frequency in Hz for data acquisition
    SAMPLING_FREQ = 781250

    # Which hydrophones to load/analyze (array of 4 booleans)
    SELECTED = [True, True, True, True]

    # Whether to plot raw signal and frequency spectrum
    PLOT_DATA = True

    # Analyzer(s) for TOA detection (set to None to skip analysis)
    # Can be a single analyzer or a list of analyzers to run in sequence
    ANALYZERS = [
        TOAEnvelopeAnalyzer(
            search_band_min=25000,
            search_band_max=40000,
            use_narrow_band=True,
            narrow_band_width=100,
            reference_hydrophone=0,
            plot_results=True,
            threshold_sigma=5
        ),
    ]

    # Step 1: Get data (capture new or load existing)
    if CAPTURE_NEW_DATA:
        timestamp = time.strftime('%Y-%m-%d--%H-%M-%S')
        output_dir = os.path.join(CAPTURE_OUTPUT, timestamp)
        DATA_PATH = capture_data(
            sampling_freq=SAMPLING_FREQ,
            capture_time=CAPTURE_TIME,
            capture_format=CAPTURE_FORMAT,
            output_dir=output_dir
        )
    else:
        DATA_PATH = DATA_FILE

    # Step 2: Load data into hydrophone array
    hydrophone_array_obj = load_hydrophone_data(
        data_path=DATA_PATH,
        sampling_freq=SAMPLING_FREQ,
        selected_hydrophones=SELECTED,
        plot_data=PLOT_DATA
    )

    # Step 3: Run analysis
    analysis_results = run_controller(
        hydrophone_array=hydrophone_array_obj,
        analyzers=ANALYZERS
    )

