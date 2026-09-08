"""Controller module for hydrophone data acquisition and analysis."""
import time
import threading
import queue
import sys
from io import StringIO
from multiprocessing import Pool
from logic.logic2 import Logic2
from hydrophones import hydrophone_array
from analyzers import TOAEnvelopeAnalyzer, NearbyAnalyzer

# Keep worker processes bounded to avoid competing with capture and device work.
MAX_CONCURRENT_ANALYSIS_PROCESSES = 4

# Quiet mode keeps repeated live votes readable and avoids unnecessary console I/O.
QUIET_MODE = True

# Mock mode is useful when developing without attached hardware.
USE_MOCK_DEVICE = True

# Capture settings are shared by live capture and the Logic 2 adapter.
CAPTURE_TIME = 2
CAPTURE_FORMAT = ["bin"]
CAPTURE_OUTPUT_DIR = "Temp_Data"

SAMPLING_FREQ = 781250
SELECTED = [True, False, False, False]
PLOT_DATA = False

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

SALEAE = Logic2(is_mock=USE_MOCK_DEVICE)


def capture_data(prefix: str = ""):
    """Capture one Logic 2 recording and return its directory path."""
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
    """Load a capture into an array and optionally plot selected signals."""
    if QUIET_MODE:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            array.load_from_path(data_path)
        finally:
            sys.stdout = old_stdout
    else:
        array.load_from_path(data_path)

    if PLOT_DATA:
        array.plot_hydrophones()


def run_analyzers(array):
    """Run each configured analyzer and return results by analyzer name."""
    results = {}

    for analyzer in ANALYZERS:
        if not QUIET_MODE:
            print(f"\n{'='*60}")
        analysis_result = analyzer.analyze_array(array)
        if not QUIET_MODE:
            analyzer.print_results(analysis_result)
        results[analysis_result['analyzer']] = analysis_result

    return results


def valid_sample(toa_results):
    """Return whether every selected hydrophone has a valid TOA result."""
    for idx, is_selected in enumerate(SELECTED):
        if is_selected:
            result = next((r for r in toa_results if r['hydrophone_idx'] == idx), None)
            if result is None or not result.get('is_valid', False):
                return False
    return True


def nearby(nearby_results):
    """Return whether any selected hydrophone is classified as nearby."""
    for result in nearby_results:
        idx = result['hydrophone_idx']
        if SELECTED[idx] and result.get('is_nearby', False):
            return True
    return False


def _analyze_worker(data_path):
    """Analyze one capture from a multiprocessing worker."""
    return analyze_one_sample(data_path)


def analyze_one_sample(data_path: str):
    """Load and analyze one capture, returning aggregated results."""
    array = hydrophone_array.HydrophoneArray(
        sampling_freq=SAMPLING_FREQ,
        selected=SELECTED
    )
    load_hydrophone_data(data_path, array)

    results = run_analyzers(array)

    toa_analysis = results.get('TOA Envelope Detection')
    nearby_analysis = results.get('ML-based Nearby Detection (10ft)')
    if toa_analysis is None:
        return (False, False, [], [])

    toa_results = toa_analysis['results']
    is_valid = valid_sample(toa_results)

    nearby_results = nearby_analysis['results'] if nearby_analysis else []
    is_nearby_val = nearby(nearby_results)

    return (is_nearby_val, is_valid, toa_results, nearby_results)


def threaded_capture_data(capture_data_paths_queue, stop_event, num_captures_list):
    """Capture recordings until the stop event is set or capture fails."""
    while not stop_event.is_set():
        try:
            data_path = capture_data()
            while not stop_event.is_set():
                try:
                    capture_data_paths_queue.put(data_path, timeout=0.1)
                    break
                except queue.Full:
                    continue
            else:
                break
            num_captures_list[0] += 1
        except Exception:
            break


def cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool, timeout_timer):
    """Stop active workers, close resources, and print run statistics."""
    end_time = time.time()
    stop_event.set()
    timeout_timer.cancel()
    capture_data_thread.join(timeout=2)
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


def run_voting_ensemble(num_votes_needed=2, timeout=60):
    """Capture, analyze, and vote until a result or timeout is reached."""
    start_time = time.time()
    SALEAE.open()
    is_nearby = False
    votes = []
    confidences = []
    num_captures_list = [0]
    capture_data_paths_queue = queue.Queue(maxsize=MAX_CONCURRENT_ANALYSIS_PROCESSES)
    pending_results = set()

    stop_event = threading.Event()

    def timeout_handler():
        """Stop the ensemble when its time budget expires."""
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

    with Pool(processes=MAX_CONCURRENT_ANALYSIS_PROCESSES) as process_pool:
        while True:
            if stop_event.is_set():
                cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool, timeout_timer)
                return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}

            completed_results = []
            for async_result in list(pending_results):
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
                            cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool, timeout_timer)
                            return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}
                        if false_count >= num_votes_needed:
                            print(f"Result: False ({false_count} votes)")
                            is_nearby = False
                            cleanup(start_time, votes, confidences, stop_event, capture_data_thread, num_captures_list, process_pool, timeout_timer)
                            return {'is_nearby': is_nearby, 'votes': votes, 'confidences': confidences}

                        completed_results.append(async_result)
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        completed_results.append(async_result)

            for result in completed_results:
                pending_results.discard(result)

            if not capture_data_paths_queue.empty() and len(pending_results) < MAX_CONCURRENT_ANALYSIS_PROCESSES:
                data_path = capture_data_paths_queue.get()
                async_result = process_pool.apply_async(_analyze_worker, (data_path,))
                pending_results.add(async_result)

            time.sleep(0.1)

if __name__ == "__main__":
    run_voting_ensemble()
