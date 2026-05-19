"""Controller module for hydrophone data acquisition and analysis."""
import time
import threading
import queue
import warnings
import sys
from io import StringIO
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from logic.logic2 import Logic2
from hydrophones import hydrophone_array
from analyzers import TOAEnvelopeAnalyzer, NearbyAnalyzer

# Suppress sklearn warnings about nested parallelism
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.ensemble._base')

# Limit concurrent analysis threads to prevent GIL contention and excessive context switching
MAX_CONCURRENT_ANALYSIS_THREADS = 4

# Whether to suppress verbose analyzer output during voting ensemble (improves speed)
QUIET_MODE = True

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
    """Load hydrophone data, optionally suppressing output."""
    if QUIET_MODE:
        # Suppress stdout during loading
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            array.load_from_path(data_path, True)
        finally:
            sys.stdout = old_stdout
    else:
        array.load_from_path(data_path, True)
    
    if PLOT_DATA:
        array.plot_hydrophones()

def run_analyzers(array):
    """Run all configured analyzers on hydrophone array.
    
    Analyzers run in parallel using ThreadPoolExecutor since they are
    independent and may have I/O or light processing that benefits from
    concurrent execution.
    
    Returns:
        List of analysis results from each analyzer
    """
    results = []
    
    def run_single_analyzer(analyzer):
        """Run a single analyzer and optionally print results."""
        if not QUIET_MODE:
            print(f"\n{'='*60}")
        analysis_result = analyzer.analyze_array(array)
        if not QUIET_MODE:
            analyzer.print_results(analysis_result)
        return analysis_result
    
    # Run analyzers in parallel using thread pool
    # (independent, can benefit from concurrent I/O)
    with ThreadPoolExecutor(max_workers=len(ANALYZERS)) as executor:
        futures = [executor.submit(run_single_analyzer, analyzer) for analyzer in ANALYZERS]
        for future in futures:
            results.append(future.result())
    
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


def _analyze_worker(data_path):
    """Worker process function for multiprocessing - analyzes one sample.
    
    Must be at module level to be pickleable by multiprocessing.Pool.
    """
    return analyze_one_sample(data_path)


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


def threaded_capture_data(capture_data_paths_queue, stop_event, num_captures_list):
    """Continuously capture data until stop_event is set."""
    while not stop_event.is_set():
        try:
            data_path = capture_data()
            capture_data_paths_queue.put(data_path)
            num_captures_list[0] += 1
        except Exception:
            break

def cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool):
    """Clean up resources and print timing statistics."""
    end_time = time.time()
    stop_event.set()
    # Wait for capture thread to finish
    capture_data_thread.join(timeout=2)
    # Close the process pool
    process_pool.close()
    process_pool.join()
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
    pending_results = {}
    
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

    with Pool(processes=MAX_CONCURRENT_ANALYSIS_THREADS) as process_pool:
        while True:
            if stop_event.is_set():
                cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool)
                return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
            
            completed_results = []
            for async_result in list(pending_results.keys()):
                if async_result.ready():
                    try:
                        is_nearby_val, is_valid, _, nearby_results = async_result.get(timeout=1)
                        
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
                            cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool)
                            return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
                        if false_count >= num_votes_needed:
                            print(f"Result: False ({false_count} votes)")
                            is_nearby = False
                            cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool)
                            return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
                        
                        completed_results.append(async_result)
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        completed_results.append(async_result)
            
            for result in completed_results:
                del pending_results[result]
            
            if not capture_data_paths_queue.empty() and len(pending_results) < MAX_CONCURRENT_ANALYSIS_THREADS:
                data_path = capture_data_paths_queue.get()
                async_result = process_pool.apply_async(_analyze_worker, (data_path,))
                pending_results[async_result] = data_path
            
            time.sleep(0.1)

if __name__ == "__main__":
    run_voting_ensemble()
