"""Batch processing script for hydrophone data analysis."""
import os
import time
import csv
from controller import run_controller, load_hydrophone_data, check_all_valid
from analyzers import TOAEnvelopeAnalyzer, NearbyAnalyzer


def process_sample(array_obj, sample_name, truth, OUTPUT_PATH, SELECTED, confusion, results_list):
    """Process a single hydrophone array sample and return results."""
    try:
        results = run_controller(
            hydrophone_array=array_obj,
            analyzers=results_list
        )
        
        # Find closest hydrophone by earliest TOA time
        toa_results = results[0]['results']
        earliest_time = float('inf')
        predicted = None
        
        for result in toa_results:
            idx = result['hydrophone_idx']
            toa_time = result.get('toa_time')
            
            if toa_time is not None and toa_time < earliest_time:
                earliest_time = toa_time
                predicted = idx
        
        # Extract TOA times and validity for CSV
        toa_dict = {r['hydrophone_idx']: r['toa_time'] for r in toa_results}
        valid_dict = {r['hydrophone_idx']: r.get('is_valid', False) for r in toa_results}
        validity_reason_dict = {r['hydrophone_idx']: r.get('validity_reason', 'UNKNOWN') for r in toa_results}
        nearby_results = results[1]['results']
        nearby_dict = {r['hydrophone_idx']: r['nearby'] for r in nearby_results}
        nearby_valid_dict = {r['hydrophone_idx']: r.get('is_valid', False) for r in nearby_results}
        delta_t_dict = {r['hydrophone_idx']: r.get('delta_t', None) for r in nearby_results}
        
        toas = [toa_dict.get(i) for i in range(4)]
        valid_status = [valid_dict.get(i) for i in range(4)]
        validity_reasons = [validity_reason_dict.get(i) for i in range(4)]
        nearby_status = [nearby_dict.get(i) for i in range(4)]
        nearby_valid_status = [nearby_valid_dict.get(i) for i in range(4)]
        delta_ts = [delta_t_dict.get(i) for i in range(4)]
        
        # Check if all selected hydrophones are valid
        all_valid = check_all_valid(toa_results, SELECTED)
        
        # Write to CSV
        row = [sample_name, truth, predicted] + toas + [all_valid] + valid_status + validity_reasons + nearby_status + nearby_valid_status + delta_ts
        with open(OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        
        # Update confusion matrix only for valid samples
        if all_valid and truth is not None and predicted is not None:
            confusion[truth][predicted] += 1
            valid_files_count = 1 if all_valid else 0
        else:
            valid_files_count = 0
        
        print(f"\n{'='*60}")
        print(f"Processed: {sample_name} | Predicted: H{predicted} | Truth: H{truth}")
        return valid_files_count
        
    except Exception as e:
        print(f"Error: {sample_name} - {e}")
        return 0


if __name__ == "__main__":
    # Configuration
    IS_LOGIC_2 = True
    SAMPLING_FREQ = 781250
    SELECTED = [True, False, False, False]
    PLOT_DATA = False
    
    DATA_PATHS = [
        "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08",
        "data/2.22.2026/H0_Closest_10FT_2026-02-22--15-35-26",
        "data/2.22.2026/H0_Closest_20FT_2026-02-22--15-41-55",
        "data/2.22.2026/H0_Closest_30FT_2026-02-22--15-47-28",
        "data/2.22.2026/H0_Closest_40FT_2026-02-22--15-53-23",
    ]
    
    ANALYZERS = [
        TOAEnvelopeAnalyzer(
            threshold_sigma=5,
            raw_signal_threshold=0.5,
            margin_front=0.1,
            margin_end=0.1,
            filter_order=0,
            search_band_min=25000,
            search_band_max=40000,
            plot_results=False
        ),
        NearbyAnalyzer(
            ping_width_threshold=0.1,
            crossing_std_dev=5,
            filter_order=0,
            search_band_min=25000,
            search_band_max=40000,
            plot_results=False
        ),
    ]
    
    # Setup output CSV
    timestamp = time.strftime('%Y-%m-%d--%H-%M-%S')
    OUTPUT_PATH = os.path.join("analysis", f"analysis_{timestamp}.csv")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    HEADERS = ["PATH", "TRUTH", "PREDICTED",
               "H0 TOA", "H1 TOA", "H2 TOA", "H3 TOA",
               "ALL_VALID",
               "H0 VALID", "H1 VALID", "H2 VALID", "H3 VALID",
               "H0 REASON", "H1 REASON", "H2 REASON", "H3 REASON",
               "H0 NEARBY", "H1 NEARBY", "H2 NEARBY", "H3 NEARBY",
               "H0 NEARBY_VALID", "H1 NEARBY_VALID", "H2 NEARBY_VALID", "H3 NEARBY_VALID",
               "H0 DELTA_T", "H1 DELTA_T", "H2 DELTA_T", "H3 DELTA_T"]
    
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(HEADERS)
    
    # Initialize tracking
    confusion = {i: {j: 0 for j in range(4)} for i in range(4)}
    total_files = 0
    valid_files = 0
    errors = 0
    
    # Process all data files
    for data_dir in DATA_PATHS:
        dir_name = os.path.basename(os.path.normpath(data_dir))
        try:
            truth = int(dir_name.split("_")[0])
        except (ValueError, IndexError):
            truth = None
        
        # Get list of items to process (epoch dirs for Logic 2, files for Logic 1)
        items = []
        if IS_LOGIC_2:
            for epoch_dir in os.listdir(data_dir):
                epoch_path = os.path.join(data_dir, epoch_dir)
                if os.path.isdir(epoch_path):
                    items.append((epoch_dir, epoch_path))
            # Sort items numerically by epoch number
            items.sort(key=lambda x: int(x[0].split('_')[-1]))
        else:
            for filename in os.listdir(data_dir):
                if filename.endswith('.bin'):
                    filepath = os.path.join(data_dir, filename)
                    items.append((filename, filepath))
            items.sort()
        
        # Process each item
        for item_name, item_path in items:
            total_files += 1
            
            array = load_hydrophone_data(
                data_path=item_path,
                sampling_freq=SAMPLING_FREQ,
                selected_hydrophones=SELECTED,
                is_logic_2=IS_LOGIC_2,
                plot_data=PLOT_DATA
            )
            
            valid_count = process_sample(array, item_name, truth, OUTPUT_PATH, SELECTED, confusion, ANALYZERS)
            valid_files += valid_count
            if valid_count == 0 and item_name:
                errors += 1
    
    # Print metrics
    correct = sum(confusion[i][i] for i in range(4))
    accuracy = 100 * correct / valid_files if valid_files > 0 else 0
    
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Errors: {errors}")
    print(f"Accuracy: {correct}/{valid_files} ({accuracy:.1f}%)")
    
    print("\nConfusion Matrix:")
    print("       Predicted")
    print("       0   1   2   3")
    print("     " + "-" * 17)
    for i in range(4):
        counts = [confusion[i][j] for j in range(4)]
        if sum(counts) > 0:
            print(f"True {i} | {counts[0]:2d}  {counts[1]:2d}  {counts[2]:2d}  {counts[3]:2d}")
    
    print(f"\n{'='*60}")
    print(f"Results: {OUTPUT_PATH}")
    print(f"{'='*60}")