"""Batch processing script for hydrophone data analysis."""
import os
import time
import csv
from controller import run_controller, load_hydrophone_data, check_all_valid, get_analyzers, find_closest_hydrophone


def process_sample(array_obj, sample_name, truth, distance_truth, OUTPUT_PATH, SELECTED, confusion, results_list):
    """Process a single hydrophone array sample and return results."""
    try:
        results = run_controller(
            hydrophone_array=array_obj,
            analyzers=results_list
        )
        
        # Find closest hydrophone and nearby status using controller's logic
        predicted, detected_nearby, all_valid = find_closest_hydrophone(results, SELECTED)
        
        # Extract results indexed by hydrophone (for CSV columns)
        toa_results = results[0]['results']
        toas = [None] * 4
        valid_status = [None] * 4
        validity_reasons = [None] * 4
        nearby_status = [None] * 4
        nearby_valid_status = [None] * 4
        delta_ts = [None] * 4
        
        for r in toa_results:
            idx = r['hydrophone_idx']
            toas[idx] = r['toa_time']
            valid_status[idx] = r.get('is_valid', False)
            validity_reasons[idx] = r.get('validity_reason', 'UNKNOWN')
        
        nearby_results = results[1]['results']
        for r in nearby_results:
            idx = r['hydrophone_idx']
            nearby_status[idx] = r['nearby']
            nearby_valid_status[idx] = r.get('is_valid', False)
            delta_ts[idx] = r.get('delta_t', None)
        
        # Build CSV row
        row = [sample_name, truth, distance_truth, predicted, detected_nearby]
        row.extend(toas)
        row.append(all_valid)
        row.extend(valid_status)
        row.extend(validity_reasons)
        row.extend(nearby_status)
        row.extend(nearby_valid_status)
        row.extend(delta_ts)
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
        return valid_files_count, detected_nearby
        
    except Exception as e:
        print(f"Error: {sample_name} - {e}")
        return 0, None


if __name__ == "__main__":
    # Start timing
    script_start_time = time.perf_counter()
    
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
    
    ANALYZERS = get_analyzers()
    
    # Setup output CSV
    timestamp = time.strftime('%Y-%m-%d--%H-%M-%S')
    OUTPUT_PATH = os.path.join("analysis", f"analysis_{timestamp}.csv")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    HEADERS = ["PATH", "TRUTH", "DISTANCE_TRUTH", "PREDICTED", "DETECTED_NEARBY",
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
    nearby_correct = 0
    nearby_total = 0
    sample_times = []
    
    # Process all data files
    for data_dir in DATA_PATHS:
        dir_name = os.path.basename(os.path.normpath(data_dir))
        try:
            truth = int(dir_name.split("_")[0])
        except (ValueError, IndexError):
            truth = None
        
        # Extract distance from directory name (e.g., "20FT" from "H0_Closest_20FT_...")
        # Find "FT" in the string, then extract the number between underscores
        try:
            ft_idx = dir_name.index("FT")
            # Find the underscore before FT
            start_idx = dir_name.rfind("_", 0, ft_idx)
            # Find the underscore after FT
            end_idx = dir_name.find("_", ft_idx)
            if end_idx == -1:
                end_idx = len(dir_name)
            # Extract the distance string (e.g., "20FT")
            distance_str = dir_name[start_idx+1:end_idx]
            # Extract numeric part
            distance_ft = int(distance_str.replace("FT", ""))
        except (ValueError, AttributeError):
            distance_str = "UNKNOWN"
            distance_ft = None
        
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
            sample_start_time = time.perf_counter()
            
            array = load_hydrophone_data(
                data_path=item_path,
                sampling_freq=SAMPLING_FREQ,
                selected_hydrophones=SELECTED,
                is_logic_2=IS_LOGIC_2,
                plot_data=PLOT_DATA
            )
            
            valid_count, detected_nearby = process_sample(array, item_name, truth, distance_str, OUTPUT_PATH, SELECTED, confusion, ANALYZERS)
            valid_files += valid_count
            
            sample_elapsed_time = time.perf_counter() - sample_start_time
            sample_times.append(sample_elapsed_time)
            
            # Check nearby detection accuracy
            if distance_ft is not None and detected_nearby is not None:
                # nearby=True if distance <= 20FT, False otherwise
                is_nearby_true = distance_ft <= 20
                if detected_nearby == is_nearby_true:
                    nearby_correct += 1
                nearby_total += 1
            
            if valid_count == 0 and item_name:
                errors += 1
    
    # Print metrics
    correct = sum(confusion[i][i] for i in range(4))
    accuracy = 100 * correct / valid_files if valid_files > 0 else 0
    nearby_accuracy = 100 * nearby_correct / nearby_total if nearby_total > 0 else 0
    
    # Calculate timing metrics
    script_elapsed_time = time.perf_counter() - script_start_time
    avg_sample_time = sum(sample_times) / len(sample_times) if sample_times else 0
    
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
    print("NEARBY DETECTION (threshold: 20FT)")
    print(f"{'='*60}")
    print(f"Nearby Correct: {nearby_correct}/{nearby_total} ({nearby_accuracy:.1f}%)")
    print(f"  - nearby=True (<= 20FT): distances 0FT, 10FT, 20FT")
    print(f"  - nearby=False (> 20FT): distances 30FT, 40FT")
    
    print(f"\n{'='*60}")
    print("TIME ANALYSIS")
    print(f"{'='*60}")
    print(f"Average time per sample: {avg_sample_time:.3f}s")
    print(f"Total script execution time: {script_elapsed_time:.1f}s ({script_elapsed_time/60:.1f}m)")
    print(f"Total samples processed: {len(sample_times)}")
    if sample_times:
        print(f"Min sample time: {min(sample_times):.3f}s")
        print(f"Max sample time: {max(sample_times):.3f}s")
    
    print(f"\n{'='*60}")
    print(f"Results: {OUTPUT_PATH}")
    print(f"{'='*60}")