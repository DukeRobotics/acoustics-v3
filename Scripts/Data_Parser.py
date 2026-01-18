import Hydrophone_Array
import os

def run_data_parser(
        sampling_freq,
        selected,
        data_directories,
        data_sample_output_dir,
        plot_envelope,
        plot_gcc):

    # Parameter validation
    if sampling_freq <= 0:
        raise ValueError(f"sampling_freq must be positive, got {sampling_freq}")
    
    if len(selected) != 4:
        raise ValueError(f"selected must have exactly 4 elements, got {len(selected)}")
    
    if not any(selected):
        raise ValueError("At least one hydrophone must be selected (True in selected array)")
    
    if not data_directories:
        raise ValueError("data_directories cannot be empty")
    
    # Verify all directories exist
    for directory in data_directories:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Data directory does not exist: {directory}")
        if not os.path.isdir(directory):
            raise ValueError(f"Path must be a directory: {directory}")
    
    # Initialize hydrophone array
    hydrophone_array = Hydrophone_Array.HydrophoneArray(
        sampling_freq=sampling_freq,
        enable_data_sample=True,
        data_sample_out_dir=data_sample_output_dir
    )

    # Process each directory
    for base_path in data_directories:
        print(f"\nProcessing directory: {base_path}")
        print("=" * 50)
        
        filenames = os.listdir(base_path)
        bin_files = [f for f in filenames if os.path.splitext(f)[1].lower() == ".bin"]
        
        if not bin_files:
            print(f"No .bin files found in {base_path}")
            continue
        
        print(f"Found {len(bin_files)} .bin file(s)")
        
        # Process each .bin file
        for filename in bin_files:
            path = os.path.join(base_path, filename)
            print(f"\nProcessing: {path}")

            hydrophone_array.load_from_path(path)

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
            
            hydrophone_array.data_sample()
    
    print("\n" + "=" * 50)
    print("Data parsing complete!")
    print("=" * 50)

if __name__ == "__main__":
    SAMPLING_FREQ = 781250                          # Sampling frequency in Hz for data acquisition
    SELECTED = [True, True, True, True]             # Which hydrophones to analyze (array of 4 booleans)
    
    DATA_DIRECTORIES = [                            # List of directories containing .bin files to process
        "Temp_Data/0_2026-01-17--20-02-54",
    ]
    
    DATA_SAMPLE_OUTPUT_DIR = "Temp_Data/Test_Set"     # Directory to save extracted data samples
    
    PLOT_ENVELOPE = False                           # Whether to plot envelope analysis results
    PLOT_GCC = False                                # Whether to plot GCC (Generalized Cross-Correlation) results

    run_data_parser(
        sampling_freq=SAMPLING_FREQ,
        selected=SELECTED,
        data_directories=DATA_DIRECTORIES,
        data_sample_output_dir=DATA_SAMPLE_OUTPUT_DIR,
        plot_envelope=PLOT_ENVELOPE,
        plot_gcc=PLOT_GCC
    )


