"""Controller module for hydrophone data acquisition and analysis."""
import os
import time
from logic.logic2 import Logic2
from hydrophones import hydrophone_array
from analyzers import TOAEnvelopeAnalyzer, NearbyAnalyzer

# Whether to capture new data from Logic hardware (True) or use existing file (False)
CAPTURE_NEW_DATA = True

# Path to existing data file (used when CAPTURE_NEW_DATA = False)
DATA_FILE = ""

# Whether to use mock device for Logic 2 (True) or real device (False)
USE_MOCK_DEVICE = True

# Duration of capture in seconds (only used if CAPTURE_NEW_DATA = True)
CAPTURE_TIME = 2

# Format for capture: 'bin' and/or 'csv' (only used if CAPTURE_NEW_DATA = True)
CAPTURE_FORMAT = ["bin"]

# Output directory for captured data (only used if CAPTURE_NEW_DATA = True)
CAPTURE_OUTPUT_DIR = "Temp_Data"

# Sampling frequency in Hz for data acquisition
SAMPLING_FREQ = 781250

# Which hydrophones to load/analyze (array of 4 booleans)
SELECTED = [True, False, False, False]

# Whether to plot raw signal and frequency spectrum
PLOT_DATA = False

# Analyzer(s) for TOA detection (set to None to skip analysis)
ANALYZERS = [
    TOAEnvelopeAnalyzer(
        threshold_sigma=5,
        raw_signal_threshold=0.5,
        margin_front=0.1,
        margin_end=0.1,
        filter_order=6,
        search_band_min=30000,
        search_band_max=34000,
        plot_results_flag=False
    ),
    NearbyAnalyzer(
        ping_width_threshold=0.0147,
        crossing_std_dev=5,
        filter_order=6,
        search_band_min=30000,
        search_band_max=34000,
        plot_results_flag=False
    ),
]

# TODO: Write a comment
SALEAE = Logic2(is_mock=USE_MOCK_DEVICE)

# TODO: Write a comment
ARRAY = hydrophone_array.HydrophoneArray(
    sampling_freq=SAMPLING_FREQ,
    selected=SELECTED
)

def capture_data(prefix: str = ""):
    if not prefix:
        prefix = time.strftime('%Y-%m-%d--%H-%M-%S')
    
    _, data_path = SALEAE.capture(
        seconds=CAPTURE_TIME,
        prefix=prefix,
        base_dir=CAPTURE_OUTPUT_DIR,
        sample_rate=int(SAMPLING_FREQ),
        formats=CAPTURE_FORMAT
    )
    return data_path

def load_hydrophone_data(data_path: str):
    ARRAY.load_from_path(data_path, True)
    if PLOT_DATA:
        ARRAY.plot_hydrophones()

def run_analyzers():
    """Run all configured analyzers on hydrophone array.
    
    Returns:
        List of analysis results from each analyzer
    """
    results = []
    for analyzer in ANALYZERS:
        print(f"\n{'='*60}")
        analysis_result = analyzer.analyze_array(ARRAY)
        analyzer.print_results(analysis_result)
        results.append(analysis_result)
    return results


def valid_sample(toa_results):
    """Check if all selected hydrophones are valid.
    
    Args:
        toa_results: List of TOA analysis results
        
    Returns:
        True only if all selected hydrophones have is_valid=True
    """
    for idx, is_selected in enumerate(SELECTED):
        if is_selected:
            result = next((r for r in toa_results if r['hydrophone_idx'] == idx), None)
            if result is None or not result.get('is_valid', False):
                return False
    return True


def nearby(nearby_results):
    """Check if any selected hydrophone detects nearby signal.
    
    Args:
        nearby_results: List of nearby analysis results
        
    Returns:
        True if any selected and valid hydrophone detects nearby
    """
    for result in nearby_results:
        idx = result['hydrophone_idx']
        if SELECTED[idx] and result.get('is_valid', False):
            if result.get('nearby', False):
                return True
    return False


def orchestration_for_one_sample():
    """Run single sample through capture and analysis pipeline.
    
    Returns:
        Tuple of (is_nearby, is_valid) from the single sample
    """
    data_path = capture_data()
    load_hydrophone_data(data_path)
    
    results = run_analyzers()
    
    if not results:
        return (False, False)
    
    toa_results = results[0]['results']
    is_valid = valid_sample(toa_results)
    
    is_nearby_val = False
    if len(results) > 1:
        nearby_results = results[1]['results']
        is_nearby_val = nearby(nearby_results)
    
    return (is_nearby_val, is_valid, toa_results)


def run_voting_ensemble(num_votes_needed=5):
    """Run samples until one outcome gets enough votes.
    
    Collects votes until reaching num_votes_needed of one outcome (True or False).
    Max samples = 3 * num_votes_needed. Returns False if max samples reached.
    
    Args:
        num_votes_needed: Number of votes needed to win
        
    Returns:
        Dict with is_nearby, votes list, nearby_count, and num_valid_samples
    """
    true_count = 0
    false_count = 0
    votes = []
    max_samples = num_votes_needed * 3
    
    print(f"Starting voting ensemble ({num_votes_needed} votes needed, max {max_samples} samples)...")
    
    while len(votes) < max_samples:
        is_nearby_val, is_valid, _ = orchestration_for_one_sample()
        
        if is_valid:
            votes.append(is_nearby_val)
            if is_nearby_val:
                true_count += 1
            else:
                false_count += 1
            
            print(f"  Vote {len(votes)}: {is_nearby_val} (True: {true_count}, False: {false_count})")
            
            if true_count >= num_votes_needed:
                print(f"Result: True ({true_count} votes)")
                SALEAE.close()
                return {
                    'is_nearby': True,
                    'votes': votes,
                    'nearby_count': true_count,
                    'num_valid_samples': len(votes)
                }
            
            if false_count >= num_votes_needed:
                print(f"Result: False ({false_count} votes)")
                SALEAE.close()
                return {
                    'is_nearby': False,
                    'votes': votes,
                    'nearby_count': true_count,
                    'num_valid_samples': len(votes)
                }
        else:
            print(f"  Invalid sample, retrying...")
    
    print(f"Max samples reached. Returning False.")
    SALEAE.close()
    return {
        'is_nearby': False,
        'votes': votes,
        'nearby_count': true_count,
        'num_valid_samples': len(votes)
    }


if __name__ == "__main__":
    result = orchestration_for_one_sample()
    SALEAE.close()
