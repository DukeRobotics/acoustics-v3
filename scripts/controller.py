"""Controller module for hydrophone data acquisition and analysis."""
import time
import threading
import queue
from logic.logic2 import Logic2
from hydrophones import hydrophone_array
from analyzers import TOAEnvelopeAnalyzer, NearbyAnalyzer

# Limit concurrent analysis threads to prevent GIL contention and excessive context switching
MAX_CONCURRENT_ANALYSIS_THREADS = 3

# Whether to capture new data from Logic hardware (True) or use existing file (False)
CAPTURE_NEW_DATA = True

# Path to existing data file (used when CAPTURE_NEW_DATA = False)
DATA_FILE = ""

# Whether to use mock device for Logic 2 (True) or real device (False)
USE_MOCK_DEVICE = False

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
        model_path='scripts/artifacts/proximity_classifier_10ft_threshold_2026-04-12--23-04-00.pkl',
        filter_order=6,
        search_band_min=30000,
        search_band_max=34000,
        plot_results_flag=False
    ),
]

# Create a global logic object.
SALEAE = Logic2(is_mock=USE_MOCK_DEVICE)

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

def load_hydrophone_data(data_path: str, array):
    array.load_from_path(data_path, True)
    if PLOT_DATA:
        array.plot_hydrophones()

def run_analyzers(array):
    """Run all configured analyzers on hydrophone array.
    
    Returns:
        List of analysis results from each analyzer
    """
    results = []
    for analyzer in ANALYZERS:
        print(f"\n{'='*60}")
        analysis_result = analyzer.analyze_array(array)
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
        True if any selected hydrophone detects nearby
    """
    for result in nearby_results:
        idx = result['hydrophone_idx']
        if SELECTED[idx] and result.get('is_nearby', False):
            return True
    return False


def analyze_one_sample(data_path: str):
    """Run single sample through the analysis pipeline.
    
    Returns:
        Tuple of (is_nearby, is_valid, toa_results, nearby_results)
    """
    array = hydrophone_array.HydrophoneArray(
        sampling_freq=SAMPLING_FREQ,
        selected=SELECTED
    )
    load_hydrophone_data(data_path, array)
    
    results = run_analyzers(array)
    
    if not results:
        return (False, False, [], [])
    
    toa_results = results[0]['results']
    is_valid = valid_sample(toa_results)
    
    nearby_results = []
    is_nearby_val = False
    if len(results) > 1:
        nearby_results = results[1]['results']
        is_nearby_val = nearby(nearby_results)
    
    return (is_nearby_val, is_valid, toa_results, nearby_results)


def orchestration_for_one_sample():
    """Run single sample through capture and analysis pipeline.
    
    Returns:
        Tuple of (is_nearby, is_valid) from the single sample
    """
    data_path = capture_data()
    return analyze_one_sample(data_path=data_path)

def threaded_capture_data(capture_data_paths_queue, stop_event, num_captures_list):
    """Continuously capture data until stop_event is set."""
    while not stop_event.is_set():
        try:
            data_path = capture_data()
            capture_data_paths_queue.put(data_path)
            num_captures_list[0] += 1
            time.sleep(0.5)
        except Exception:
            break  


def threaded_analyze_one_sample(data_path, results_queue, semaphore):
    """Analyze one sample and put result in queue."""
    semaphore.acquire()
    try:
        result = analyze_one_sample(data_path)
        results_queue.put(result)
    finally:
        semaphore.release()

def cleanup(start_time, votes, confidences, stop_event, analysis_threads, capture_data_thread, num_captures_list):
    """Clean up resources and print timing statistics."""
    end_time = time.time()
    stop_event.set()
    # Wait for capture thread to finish
    capture_data_thread.join(timeout=2)
    # Wait for analysis threads to finish (max 1 second per thread)
    for thread in analysis_threads:
        thread.join(timeout=1)
    SALEAE.close()
    print(f"Total Time = {(end_time - start_time):.2f}s")
    print(f"Number of Captures = {num_captures_list[0]}")
    print(f"Total Votes = {len(votes)}")
    print(f"Votes = {votes}")
    print(f"Confidences = {confidences}")
    if votes:
        print(f"Average Time = {(((end_time - start_time)/len(votes))):.2f}s")

def run_voting_ensemble(num_votes_needed=3, timeout=60):
    start_time = time.time()
    SALEAE.open()
    is_nearby = False
    votes = []
    confidences = []
    num_captures_list = [0]
    capture_data_paths_queue = queue.Queue()
    results_queue = queue.Queue()
    analysis_threads = []

    analysis_thread_semaphore = threading.Semaphore(MAX_CONCURRENT_ANALYSIS_THREADS)
    stop_event = threading.Event()
    
    def timeout_handler():
        stop_event.set()
    
    timeout_timer = threading.Timer(timeout, timeout_handler)
    timeout_timer.daemon = True
    timeout_timer.start()

    capture_data_thread = threading.Thread(
        target=threaded_capture_data, 
        args=(capture_data_paths_queue, stop_event, num_captures_list), 
        daemon=True
    )
    capture_data_thread.start()

    while True:
        # Check if timeout fired
        if stop_event.is_set():
            cleanup(start_time, votes, confidences, stop_event, analysis_threads, capture_data_thread, num_captures_list)
            return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
        
        while True:
            if results_queue.empty():
                break

            is_nearby_val, is_valid, _, nearby_results = results_queue.get()
            
            confidence = None
            if is_valid and nearby_results:
                confidence = nearby_results[0].get('confidence', None)
            
            if is_valid:
                votes.append(is_nearby_val)
                confidences.append(confidence)
            else:
                votes.append(None)
                confidences.append(None)
            
            true_count = votes.count(True)
            false_count = votes.count(False)
            
            if is_valid:
                conf_str = f" [confidence: {confidence:.2%}]" if confidence is not None else ""
                print(f"  Vote {len(votes)}: {is_nearby_val}{conf_str} (True: {true_count}, False: {false_count})")
            else:
                print("  Invalid sample")
            
            if true_count >= num_votes_needed:
                print(f"Result: True ({true_count} votes)")
                is_nearby = True
                cleanup(start_time, votes, confidences, stop_event, analysis_threads, capture_data_thread, num_captures_list)
                return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
            if false_count >= num_votes_needed:
                print(f"Result: False ({false_count} votes)")
                is_nearby = False
                cleanup(start_time, votes, confidences, stop_event, analysis_threads, capture_data_thread, num_captures_list)
                return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
            
        if not capture_data_paths_queue.empty():
            analyze_thread = threading.Thread(target=threaded_analyze_one_sample, args=(capture_data_paths_queue.get(),results_queue,analysis_thread_semaphore), daemon=True)
            analysis_threads.append(analyze_thread)
            analyze_thread.start()
        time.sleep(0.5)

if __name__ == "__main__":
    run_voting_ensemble()
