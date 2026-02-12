"""Parser module for evaluating analyzer performance against ground truth."""
import os
import csv
import re
from pathlib import Path
from typing import List, Dict, Any
from hydrophones import hydrophone_array
from analyzers import TOAEnvelopeAnalyzer, NearbyAnalyzer


def extract_truth_label(directory_name: str) -> int | None:
    """Extract ground truth hydrophone index from directory name.
    
    Args:
        directory_name: Directory name like "0_2026-02-07--15-24-04"
        
    Returns:
        Hydrophone index (0-3) or None if no valid prefix found
    """
    match = re.match(r'^(\d+)_', os.path.basename(directory_name))
    if match:
        idx = int(match.group(1))
        if 0 <= idx <= 3:
            return idx
    return None


def find_closest_hydrophone_toa(analysis_results: Dict, selected: List[bool]) -> int | None:
    """Find which selected hydrophone had the earliest TOA.
    
    Args:
        analysis_results: Results dict from analyzer
        selected: Which hydrophones to consider
        
    Returns:
        Index of hydrophone with earliest TOA among selected ones
    """
    results = analysis_results.get('results', [])
    if not results:
        return None
    
    earliest_time = float('inf')
    earliest_idx = None
    
    for result in results:
        hydro_idx = result['hydrophone_idx']
        if selected[hydro_idx]:
            toa_time = result.get('toa_time')
            if toa_time is not None and toa_time < earliest_time:
                earliest_time = toa_time
                earliest_idx = hydro_idx
    
    return earliest_idx


def get_csv_fieldnames(analyzers: List, selected: List[bool]) -> List[str]:
    """Generate CSV column names based on analyzers and selected hydrophones.
    
    Args:
        analyzers: List of analyzer instances
        selected: Which hydrophones are selected
        
    Returns:
        List of CSV field names
    """
    fieldnames = ['directory', 'file', 'truth_label']
    
    for analyzer in analyzers:
        name = analyzer.get_name()
        fieldnames.append(f'{name}_predicted')
        fieldnames.append(f'{name}_center_freq')
        
        # Add fields for each selected hydrophone
        for idx, is_selected in enumerate(selected):
            if is_selected:
                fieldnames.extend([
                    f'{name}_h{idx}_toa_time',
                    f'{name}_h{idx}_toa_idx',
                    f'{name}_h{idx}_nearby'
                ])
    
    return fieldnames


def run_parser(
    data_directories: List[str],
    sampling_freq: float = 781250,
    selected_hydrophones: List[bool] = None,
    analyzers: List = None,
    output_csv: str = "analysis_results.csv",
    verbose: bool = True
):
    """Parse data directories, run analyzers, and evaluate performance.
    
    Args:
        data_directories: List of directory paths with truth label prefixes
        sampling_freq: Sampling frequency in Hz
        selected_hydrophones: Which hydrophones to analyze (4 bools)
        analyzers: List of analyzer instances to run
        output_csv: Path to output CSV file
        verbose: Whether to print detailed progress
    """
    if selected_hydrophones is None:
        selected_hydrophones = [True, True, True, True]
    
    if analyzers is None:
        print("Warning: No analyzers provided. Using default TOAEnvelopeAnalyzer.")
        analyzers = [
            TOAEnvelopeAnalyzer(
                search_band_min=25000,
                search_band_max=40000,
                use_narrow_band=True,
                narrow_band_width=100,
                filter_order=0,
                plot_results=False
            )
        ]
    
    # Validate inputs
    if len(selected_hydrophones) != 4:
        raise ValueError("selected_hydrophones must have exactly 4 elements")
    
    if not any(selected_hydrophones):
        raise ValueError("At least one hydrophone must be selected")
    
    # Store all results for final metrics
    all_results = []
    
    # Prepare CSV file and get fieldnames
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    # Determine CSV columns
    fieldnames = get_csv_fieldnames(analyzers, selected_hydrophones)
    
    # Open CSV file for writing
    csvfile = open(output_csv, 'w', newline='')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    
    # Process each directory
    for directory in data_directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
        
        # Extract ground truth
        truth_label = extract_truth_label(directory)
        if truth_label is None:
            print(f"Warning: Could not extract truth label from: {directory}")
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {directory}")
            print(f"Ground Truth: Hydrophone {truth_label} was closest")
            print(f"{'='*60}")
        
        # Find all .bin files in directory
        bin_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.bin'):
                bin_files.append(os.path.join(directory, filename))
        
        if not bin_files:
            print(f"Warning: No .bin files found in {directory}")
            continue
        
        # Process each file
        for bin_file in bin_files:
            if verbose:
                print(f"\nProcessing file: {os.path.basename(bin_file)}")
            
            # Load data
            array = hydrophone_array.HydrophoneArray(
                sampling_freq=sampling_freq,
                selected=selected_hydrophones
            )
            
            try:
                array.load_from_path(bin_file)
            except Exception as e:
                print(f"Error loading {bin_file}: {e}")
                continue
            
            # Run each analyzer
            file_result = {
                'directory': os.path.basename(directory),
                'file': os.path.basename(bin_file),
                'truth_label': truth_label,
            }
            
            for analyzer in analyzers:
                analyzer_name = analyzer.get_name()
                
                try:
                    print(f"\n{'-'*60}")
                    analysis_result = analyzer.analyze_array(array)
                    
                    # Print results like controller.py does
                    analyzer.print_results(analysis_result)
                    
                    # Extract closest hydrophone prediction
                    predicted_closest = find_closest_hydrophone_toa(
                        analysis_result, selected_hydrophones
                    )
                    
                    file_result[f'{analyzer_name}_predicted'] = predicted_closest
                    file_result[f'{analyzer_name}_center_freq'] = analysis_result.get('center_frequency')
                    
                    # Extract individual hydrophone results
                    for result in analysis_result.get('results', []):
                        hydro_idx = result['hydrophone_idx']
                        prefix = f'{analyzer_name}_h{hydro_idx}'
                        
                        # Store TOA time if available
                        if 'toa_time' in result:
                            file_result[f'{prefix}_toa_time'] = result['toa_time']
                        
                        # Store TOA index if available
                        if 'toa_idx' in result:
                            file_result[f'{prefix}_toa_idx'] = result['toa_idx']
                        
                        # Store nearby detection if available
                        if 'is_nearby' in result:
                            file_result[f'{prefix}_nearby'] = result['is_nearby']
                    
                    print(f"Prediction: H{predicted_closest} | Truth: H{truth_label} | "
                          f"{'✓ CORRECT' if predicted_closest == truth_label else '✗ INCORRECT'}")
                
                except Exception as e:
                    print(f"Error running {analyzer_name} on {bin_file}: {e}")
                    file_result[f'{analyzer_name}_predicted'] = None
            
            # Write this result to CSV immediately
            all_results.append(file_result)
            writer.writerow(file_result)
            csvfile.flush()  # Ensure it's written to disk
    
    # Close CSV file
    csvfile.close()
    
    # Calculate and print metrics
    print_metrics(all_results, analyzers, selected_hydrophones)
    
    return all_results


def print_metrics(results: List[Dict], analyzers: List, selected: List[bool]):
    """Calculate and print performance metrics.
    
    Args:
        results: List of result dictionaries
        analyzers: List of analyzer instances used
        selected: Which hydrophones were selected for analysis
    """
    if not results:
        print("\nNo results to analyze")
        return
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Selected hydrophones: {[i for i, s in enumerate(selected) if s]}")
    
    for analyzer in analyzers:
        analyzer_name = analyzer.get_name()
        pred_key = f'{analyzer_name}_predicted'
        
        # Filter results that have predictions for this analyzer
        valid_results = [r for r in results if pred_key in r and r[pred_key] is not None]
        
        if not valid_results:
            print(f"\n{analyzer_name}:")
            print("  No valid predictions")
            continue
        
        # Calculate accuracy considering only selected hydrophones
        correct = 0
        total = 0
        
        # Confusion matrix for selected hydrophones
        selected_indices = [i for i, s in enumerate(selected) if s]
        confusion = {i: {j: 0 for j in selected_indices} for i in selected_indices}
        
        for result in valid_results:
            truth = result['truth_label']
            pred = result[pred_key]
            
            # Only count if both truth and prediction are in selected hydrophones
            if truth in selected_indices and pred in selected_indices:
                total += 1
                if truth == pred:
                    correct += 1
                confusion[truth][pred] += 1
        
        if total == 0:
            print(f"\n{analyzer_name}:")
            print("  No predictions for selected hydrophones")
            continue
        
        accuracy = correct / total * 100
        
        print(f"\n{analyzer_name}:")
        print(f"  Valid samples: {len(valid_results)}")
        print(f"  Samples with selected hydrophones: {total}")
        print(f"  Correct predictions: {correct}")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        # Print confusion matrix
        if len(selected_indices) > 1:
            print(f"\n  Confusion Matrix (selected hydrophones only):")
            print(f"  {'Truth\\Pred':<12}", end='')
            for pred_idx in selected_indices:
                print(f"H{pred_idx:>4}", end='')
            print()
            
            for truth_idx in selected_indices:
                print(f"  H{truth_idx:<11}", end='')
                for pred_idx in selected_indices:
                    count = confusion[truth_idx][pred_idx]
                    print(f"{count:>5}", end='')
                print()
        
        # Per-class accuracy
        print(f"\n  Per-Hydrophone Accuracy:")
        for hydro_idx in selected_indices:
            hydro_total = sum(confusion[hydro_idx].values())
            hydro_correct = confusion[hydro_idx][hydro_idx]
            if hydro_total > 0:
                hydro_acc = hydro_correct / hydro_total * 100
                print(f"    H{hydro_idx}: {hydro_correct}/{hydro_total} ({hydro_acc:.2f}%)")
            else:
                print(f"    H{hydro_idx}: No samples")


if __name__ == "__main__":

    DATA_DIRECTORIES = [
        "0_2026-02-07--15-24-04",
        "2_2026-02-07--15-38-56",
    ]
    
    SAMPLING_FREQ = 781250
    
    SELECTED_HYDROPHONES = [True, False, True, False]
    
    # Analyzers to evaluate
    ANALYZERS = [
        TOAEnvelopeAnalyzer(
            search_band_min=25000,
            search_band_max=40000,
            use_narrow_band=True,
            narrow_band_width=100,
            reference_hydrophone=0,
            filter_order=0,
            plot_results=False,
            threshold_sigma=5
        ),
        NearbyAnalyzer(
            threshold=1.0,
            search_band_min=25000,
            search_band_max=40000,
            use_narrow_band=True,
            narrow_band_width=100,
            filter_order=0,
            plot_results=False
        ),
    ]
    
    # Output CSV path
    OUTPUT_CSV = "Temp_Data/analyzer_performance.csv"
    
    # Run parser
    run_parser(
        data_directories=DATA_DIRECTORIES,
        sampling_freq=SAMPLING_FREQ,
        selected_hydrophones=SELECTED_HYDROPHONES,
        analyzers=ANALYZERS,
        output_csv=OUTPUT_CSV,
        verbose=True
    )
