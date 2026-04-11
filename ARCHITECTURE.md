# Acoustics v3 Codebase Architecture

## Executive Summary

The codebase follows a **pipeline architecture** for hydrophone signal analysis:

```
Raw Data (Bin/CSV) → HydrophoneArray (loads & normalizes) → Multiple Analyzers (extract features) → CSV Results
```

The system is built around three core abstractions:
- **HydrophoneArray/Hydrophone**: Data containers that load raw sensor signals
- **BaseAnalyzer hierarchy**: Pluggable feature extraction algorithms
- **Controller**: Orchestrates the pipeline; Parser leverages it for batch processing

---

## 1. ANALYZER ARCHITECTURE

### Inheritance Hierarchy

```
BaseAnalyzer (ABC) ← TOAEnvelopeAnalyzer
                   ← NearbyAnalyzer
                   ← GarbageDetector (utility, not a full analyzer)
```

### BaseAnalyzer (Abstract Base Class)

**Location**: `scripts/analyzers/base_analyzer.py`

**Purpose**: Provides common infrastructure for all signal processing algorithms

**Key Responsibilities**:
- Bandpass filtering (Butterworth, configurable order and frequency range)
- Plotting infrastructure for visualization
- Array-level analysis orchestration (invokes individual hydrophone analyzers)
- Result aggregation and printing

**Key Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `analyze_array(hydrophone_array, selected)` | Main entry point for analyzing multiple hydrophones | `{'results': [...], 'analyzer': name}` |
| `_analyze_single(hydrophone, sampling_freq)` | **ABSTRACT** - Analyze one hydrophone | `dict` with analyzer-specific fields |
| `_plot_single_signal(ax_time, ax_freq, ...)` | **ABSTRACT** - Plot one hydrophone's results | None |
| `get_name()` | **ABSTRACT** - Return analyzer identifier | `str` |
| `apply_bandpass(signal, sampling_freq, band_min, band_max)` | Common utility - filter signal | filtered `np.ndarray` |

**Configuration Parameters**:
- `search_band_min`: Lower frequency bound (Hz) - default 25000 Hz
- `search_band_max`: Upper frequency bound (Hz) - default 40000 Hz
- `filter_order`: Butterworth filter order - default 8
- `plot_results_flag`: Whether to visualize results - default False

### TOAEnvelopeAnalyzer

**Location**: `scripts/analyzers/toa_envelope_analyzer.py`

**Purpose**: Detect **Time Of Arrival (TOA)** of a signal ping using Hilbert envelope detection

**Algorithm**:
1. Bandpass filter the signal (25-40 kHz by default)
2. Compute Hilbert transform → extract analytical signal's envelope
3. Calculate detection threshold: `mean(envelope) + threshold_sigma * std(envelope)`
4. Find first sample where envelope exceeds threshold → **TOA**
5. Validate using GarbageDetector (signal strength, timing margins)

**Returns per Hydrophone**:
```python
{
    'toa_time': float,               # Detected TOA in seconds
    'toa_idx': int,                  # Sample index of TOA
    'filtered_signal': np.ndarray,   # Bandpass filtered signal
    'processed_signal': np.ndarray,  # Hilbert envelope
    'is_valid': bool,                # Passes garbage detection
    'validity_reason': str,          # VALID | WEAK_SIGNAL | TOO_EARLY | TOO_LATE
    'threshold': float,              # Detection threshold value
    'band_min': float,               # Lower freq bound (Hz)
    'band_max': float,               # Upper freq bound (Hz)
    'hydrophone_idx': int            # Added by BaseAnalyzer.analyze_array()
}
```

**Configuration Parameters**:
- `threshold_sigma`: Std dev multiplier for threshold - default 5
- `raw_signal_threshold`: Minimum signal amplitude - default 3 (used by GarbageDetector)
- `margin_front`: Min seconds from start - default 0.1 (used by GarbageDetector)
- `margin_end`: Min seconds from end - default 0.1 (used by GarbageDetector)
- Plus all BaseAnalyzer parameters

**Example Usage in controller.py**:
```python
TOAEnvelopeAnalyzer(
    threshold_sigma=5,
    raw_signal_threshold=0.5,
    margin_front=0.1,
    margin_end=0.1,
    filter_order=6,
    search_band_min=30000,
    search_band_max=34000,
    plot_results_flag=False
)
```

### NearbyAnalyzer

**Location**: `scripts/analyzers/nearby_analyzer.py`

**Purpose**: Detect if a signal source is **nearby** vs. far away using **ping width** analysis

**Algorithm**:
1. Bandpass filter the signal
2. Compute envelope: `np.abs(filtered_signal)`
3. Calculate threshold: `mean(envelope) + crossing_std_dev * std(envelope)`
4. Find all samples where envelope exceeds threshold
5. Compute ping width: `(last_crossing - first_crossing) / sampling_freq`
6. Determine nearby: `delta_t <= ping_width_threshold`

**Physics Intuition**: 
- Nearby signals have shorter pulse duration (narrow time window of high amplitude)
- Far signals have longer pulse duration (spread over more samples)

**Returns per Hydrophone**:
```python
{
    'nearby': bool,                  # True if ping_width <= threshold
    'delta_t': float | None,         # Ping width in seconds (None if < 2 crossings)
    'filtered_signal': np.ndarray,   # Bandpass filtered signal
    'threshold': float,              # Envelope threshold
    'band_min': float,               # Lower freq bound (Hz)
    'band_max': float,               # Upper freq bound (Hz)
    'hydrophone_idx': int            # Added by BaseAnalyzer.analyze_array()
}
```

**Configuration Parameters**:
- `ping_width_threshold`: Detection threshold in seconds - default 0.014657 (~14.6 ms)
- `crossing_std_dev`: Std dev multiplier for envelope threshold - default 5
- Plus all BaseAnalyzer parameters

**Example Usage in controller.py**:
```python
NearbyAnalyzer(
    ping_width_threshold=0.014657,
    crossing_std_dev=5,
    filter_order=6,
    search_band_min=30000,
    search_band_max=34000,
    plot_results_flag=False
)
```

### GarbageDetector (Validation Utility)

**Location**: `scripts/analyzers/garbage_detector.py`

**Purpose**: Validate individual TOA measurements to filter out invalid/unreliable detections

**Validation Rules**:
```python
def validate_hydrophone_toa(signal_value, toa_time, recording_start, recording_end):
    # Rule 1: Signal amplitude must exceed threshold
    if abs(signal_value) < raw_signal_threshold:
        return False, "WEAK_SIGNAL"
    
    # Rule 2: TOA must not be too close to start (could be noise)
    if toa_time < recording_start + margin_front:
        return False, "TOO_EARLY"
    
    # Rule 3: TOA must not be too close to end (incomplete signal)
    if toa_time > recording_end - margin_end:
        return False, "TOO_LATE"
    
    return True, "VALID"
```

**Used by**: TOAEnvelopeAnalyzer (to mark results as valid/invalid)

---

## 2. DATA CONTAINERS

### Hydrophone Class

**Location**: `scripts/hydrophones/hydrophone.py`

**Purpose**: Container for a single sensor's data

**Data Fields**:
```python
class Hydrophone:
    # Time domain
    times: np.ndarray                # Time points (seconds)
    signal: np.ndarray               # Raw signal (amplitude)
    filtered_signal: np.ndarray      # Processed signal (optional)
    
    # Frequency domain
    freqs: np.ndarray                # Frequency bins (Hz)
    frequency: np.ndarray            # FFT magnitude
    filtered_frequency: np.ndarray   # FFT of filtered signal (optional)
    
    # Metadata
    sampling_period: float           # 1/sampling_freq (hydrophone-specific)
```

**Key Feature**: Each hydrophone can have its own `sampling_period` for multi-rate systems

### HydrophoneArray Class

**Location**: `scripts/hydrophones/hydrophone_array.py`

**Purpose**: Container for 4 hydrophones with data loading/preprocessing

**Initialization**:
```python
ARRAY = HydrophoneArray(
    sampling_freq=781250,      # Hz
    selected=[True, False, False, False]  # Which to analyze
)
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `load_from_path(path, is_logic_2=False)` | Auto-detect format (.bin/.csv, Logic 1/2) and load |
| `_load_from_bin(path)` | Parse binary file (header: samples, channels, period) |
| `_load_from_csv(path)` | Parse CSV (time in col 0, channels in cols 1-4) |
| `_load_from_logic2_directory(dir)` | Parse Logic 2 directory (looks for .csv or _0/_1/_2/_3.bin files) |
| `plot_hydrophones()` | Visualize raw signals and FFT |

**Processing Pipeline during Load**:
```
Raw input → Demean signal (subtract mean) → Compute FFT → Store in Hydrophone objects
```

**Key Data Transformation**:
```python
def _update_hydrophone(self, hydro, times, signal):
    hydro.times = times
    hydro.signal = signal - np.mean(signal)  # Center signal
    hydro.freqs = fftfreq(len(hydro.signal), hydro.sampling_period)
    hydro.frequency = fft(hydro.signal)
```

---

## 3. CONTROLLER.PY FLOW

**Location**: `scripts/controller.py`

**Purpose**: Orchestrates the entire analysis pipeline - bridges hardware capture and data processing

### Configuration

All settings are module-level constants:

```python
CAPTURE_NEW_DATA = True                    # Capture or use existing file
DATA_FILE = ""                             # Path if not capturing
USE_MOCK_DEVICE = False                    # Mock Logic 2 device for testing
CAPTURE_TIME = 2                           # Seconds
CAPTURE_FORMAT = ["bin"]                   # Output format
CAPTURE_OUTPUT_DIR = "Temp_Data"           # Where to save
SAMPLING_FREQ = 781250                     # Hz
SELECTED = [True, False, False, False]     # Which hydrophones to analyze
PLOT_DATA = False                          # Plot raw data

ANALYZERS = [                              # List of analyzers to run
    TOAEnvelopeAnalyzer(...),
    NearbyAnalyzer(...)
]

SALEAE = Logic2(is_mock=USE_MOCK_DEVICE)   # Hardware interface
ARRAY = HydrophoneArray(...)               # Data container
```

### Core Functions

#### 1. `capture_data(prefix="")`
**Purpose**: Acquire new data from Logic 2 hardware

**Flow**:
1. Generate timestamp if no prefix provided
2. Call `SALEAE.capture()` → returns `(_, data_path)`
3. Data saved to `CAPTURE_OUTPUT_DIR / prefix_timestamp.bin`
4. Return path for next step

**Output**: Path string to captured data

#### 2. `load_hydrophone_data(data_path)`
**Purpose**: Load captured/existing data into memory

**Flow**:
1. `ARRAY.load_from_path(data_path, is_logic_2=True)` → loads .bin/.csv
2. If `PLOT_DATA=True`: visualize raw signals
3. Hydrophone objects now contain:
   - `times`: time array
   - `signal`: demeaned raw data
   - `frequency`: FFT

**Output**: ARRAY object modified in-place

#### 3. `run_analyzers()`
**Purpose**: Run all configured analyzers on the loaded data

**Flow**:
```python
results = []
for analyzer in ANALYZERS:
    analysis_result = analyzer.analyze_array(ARRAY)
    analyzer.print_results(analysis_result)
    results.append(analysis_result)
return results
```

Each analyzer's `analyze_array()` returns:
```python
{
    'results': [                           # List of per-hydrophone results
        {'hydrophone_idx': 0, 'toa_time': ..., 'is_valid': ..., ...},
        {'hydrophone_idx': 2, 'toa_time': ..., 'is_valid': ..., ...}
    ],
    'analyzer': 'TOA Envelope Detection'   # Analyzer name
}
```

**Output**: List of analyzer results

#### 4. `analyze_one_sample(data_path)`
**Purpose**: Main analysis pipeline for a single recording

**Flow**:
```
data_path (to .bin or directory)
  ↓
load_hydrophone_data() → ARRAY populated
  ↓
run_analyzers() → results = [{TOA results}, {nearby results}]
  ↓
valid_sample(toa_results) → is_valid (bool)
  ↓
nearby(nearby_results) → is_nearby (bool)
  ↓
return (is_nearby, is_valid, toa_results, nearby_results)
```

**Helper Functions**:
- `valid_sample(toa_results)`: Returns True only if ALL selected hydrophones have `is_valid=True`
- `nearby(nearby_results)`: Returns True if ANY selected hydrophone has `nearby=True`

**Output**: 
```python
(is_nearby: bool, is_valid: bool, toa_results: list, nearby_results: list)
```

#### 5. `orchestration_for_one_sample()`
**Purpose**: Convenience wrapper for capture + analysis

**Flow**:
```
capture_data() → data_path
  ↓
analyze_one_sample(data_path) → (is_nearby, is_valid, ...)
  ↓
return (is_nearby, is_valid)
```

#### 6. `run_voting_ensemble(num_votes_needed=5)`
**Purpose**: Robustness through majority voting

**Flow**:
```
Loop up to 3*num_votes_needed times:
    orchestration_for_one_sample() → (is_nearby, is_valid)
    if is_valid:
        votes.append(is_nearby)
    
    if count(True in votes) >= num_votes_needed:
        return True, break
    if count(False in votes) >= num_votes_needed:
        return False, break

return is_nearby (majority or timeout)
```

---

## 4. PARSER.PY FLOW

**Location**: `scripts/parser.py`

**Purpose**: Batch analysis of multiple recordings, CSV result generation

### Main Entry Point: `parse_recordings(paths_to_analyze, output_path="analysis")`

**Input**: List of parent directories (each containing epoch folders)

**Output**: Single CSV file with all results

### Flow

```
paths_to_analyze = ["data/2.22.2026/H0_Closest_0FT_...", ...]
  ↓
For each parent_path:
    extract_metadata_from_path_name()  # Closest_H0, Distance_0FT
    ↓
    For each epoch_folder in parent_path:
        data_path = parent_path / epoch_folder
        ↓
        analyze_one_sample(data_path)  # From controller.py
        ↓
        Extract from results:
        - TOA times, validity flags, validation reasons (from TOAEnvelopeAnalyzer)
        - Nearby flags, ping widths (from NearbyAnalyzer)
        ↓
        Build CSV row
        ↓
        Append to CSV (file opened, row added, file closed)
```

### CSV Structure

**Base Columns** (left side):
| Column | Source |
|--------|--------|
| PATH | epoch_folder name |
| CLOSEST_HYDROPHONE | Extracted from parent path (e.g., "H0") |
| DISTANCE | Extracted from parent path (e.g., "0FT") |
| ALL_VALID | Result of `valid_sample()` (bool) |
| PREDICTED | Always 0 (placeholder for model predictions) |
| IS_NEARBY | Result of `nearby()` (bool) |

**Hydrophone Columns** (repeated for H0, H1, H2, H3):
| Column | Source |
|--------|--------|
| H{i} TOA | `toa_results[i]['toa_time']` |
| H{i} VALID | `toa_results[i]['is_valid']` |
| H{i} REASON | `toa_results[i]['validity_reason']` |
| H{i} IS_NEARBY | `nearby_results[i]['nearby']` |
| H{i} PING_WIDTH | `nearby_results[i]['delta_t']` |

### Example CSV Header
```
PATH,CLOSEST_HYDROPHONE,DISTANCE,ALL_VALID,PREDICTED,IS_NEARBY,H0 TOA,H0 VALID,H0 REASON,H0 IS_NEARBY,H0 PING_WIDTH,H1 TOA,H1 VALID,H1 REASON,H1 IS_NEARBY,H1 PING_WIDTH,...
```

### Example CSV Row
```
2026-02-22--15-29-08,H0,0FT,True,0,False,1.234567,True,VALID,False,0.025123,,,,,,...
```

### Metadata Extraction

```python
parent_name = "H0_Closest_0FT_2026-02-22--15-29-08"

# Extract closest hydrophone
closest_h = re.search(r'(H\d)', parent_name)  # → "H0"

# Extract distance
distance = re.search(r'(\d+FT)', parent_name)  # → "0FT"
```

---

## 5. COMPLETE DATA FLOW: END-TO-END EXAMPLE

### Scenario
Analyze one recording at `data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08/2026-02-22--15-29-08_1`

### Step 1: Data Loading
```
File: data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08/2026-02-22--15-29-08_1/analog_0.bin
                                                                                    ↑
                                                                   hydrophone index 0
```

**Binary Format** (header):
```
8 bytes: num_samples (uint64)
4 bytes: num_channels (uint32)
8 bytes: sampling_period (double)
remaining: float32 samples (num_samples * num_channels)
```

**HydrophoneArray._load_from_bin()**:
- Reads 20 bytes header → num_samples=781250, channels=4, period=1.28e-6
- Reshapes data: (4, 781250) → one row per hydrophone
- For hydrophone 0 (SELECTED[0]=True):
  ```
  signal_raw = data[0]  # 781250 samples
  hydrophone[0].signal = signal_raw - mean(signal_raw)
  hydrophone[0].times = [0, 1.28e-6, 2.56e-6, ..., 1.0] (2 seconds total)
  hydrophone[0].frequency = FFT(hydrophone[0].signal)
  ```

### Step 2: TOA Analysis (TOAEnvelopeAnalyzer)
```
Input: hydrophone[0] = {
    signal: demeaned 781250-sample array
    times: time array
    sampling_period: 1.28e-6
}

Processing:
  ↓ apply_bandpass(signal, fs=781250, band=[30k, 34k])
  Butterworth filter: 6th order, keeps only 30-34 kHz energy
  → filtered_signal (781250 samples)
  
  ↓ compute Hilbert envelope
  analytic_signal = hilbert(filtered_signal)
  envelope = abs(analytic_signal)
  → envelope (781250 samples)
  
  ↓ threshold detection
  threshold = mean(envelope) + 5 * std(envelope)
  crossings = where(envelope > threshold)
  toa_idx = crossings[0]  # first crossing
  → toa_time = hydrophone.times[toa_idx]
  
  ↓ garbage detection
  call GarbageDetector.validate_hydrophone_toa(
      signal_value=max(raw_signal),
      toa_time=toa_time,
      recording_start=times[0],
      recording_end=times[-1]
  )
  Checks:
    - max(signal) > 0.5 (raw_signal_threshold) ✓
    - toa_time >= 0 + 0.1 (margin_front) ?
    - toa_time <= 2.0 - 0.1 (margin_end) ?
  → is_valid (bool), validity_reason (str)

Output:
{
    'hydrophone_idx': 0,
    'toa_time': 0.543210,
    'toa_idx': 424321,
    'is_valid': True,
    'validity_reason': 'VALID',
    'filtered_signal': array([...]),
    'processed_signal': array([...]),  # envelope
    'threshold': 2.345,
    ...
}
```

### Step 3: Nearby Analysis (NearbyAnalyzer)
```
Input: same hydrophone object

Processing:
  ↓ apply_bandpass(signal, fs=781250, band=[30k, 34k])
  → filtered_signal (same as TOA analyzer)
  
  ↓ compute simple envelope
  envelope = abs(filtered_signal)
  
  ↓ threshold detection
  threshold = mean(envelope) + 5 * std(envelope)
  crossings = where(envelope > threshold)
  
  if len(crossings) >= 2:
      first = crossings[0]
      last = crossings[-1]
      delta_t = (last - first) / 781250  # in seconds
      nearby = (delta_t <= 0.014657)  # ping_width_threshold
  else:
      delta_t = None
      nearby = False

Output:
{
    'hydrophone_idx': 0,
    'nearby': True,       # If delta_t was very small
    'delta_t': 0.005123,
    'threshold': 2.345,
    ...
}
```

### Step 4: Aggregation in Controller
```
toa_results = [
    {'hydrophone_idx': 0, 'toa_time': 0.543210, 'is_valid': True, ...}
    # (H1, H2, H3 not included because SELECTED=[True, False, False, False])
]

nearby_results = [
    {'hydrophone_idx': 0, 'nearby': True, 'delta_t': 0.005123, ...}
]

valid_sample(toa_results):
    For idx=0, SELECTED[0]=True:
        result = toa_results[0]
        result.is_valid = True ✓
    return True

nearby(nearby_results):
    For result in nearby_results:
        idx=0, SELECTED[0]=True, result.nearby=True ✓
    return True

Return: (is_nearby=True, is_valid=True, toa_results, nearby_results)
```

### Step 5: CSV Row Generation (Parser)
```
parent_path = "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08"
parent_name = "H0_Closest_0FT_2026-02-22--15-29-08"
closest_h = "H0"
distance = "0FT"
epoch_folder = "2026-02-22--15-29-08_1"

row = {
    'PATH': "2026-02-22--15-29-08_1",
    'CLOSEST_HYDROPHONE': "H0",
    'DISTANCE': "0FT",
    'ALL_VALID': True,
    'PREDICTED': 0,
    'IS_NEARBY': True,
    'H0 TOA': 0.543210,
    'H0 VALID': True,
    'H0 REASON': 'VALID',
    'H0 IS_NEARBY': True,
    'H0 PING_WIDTH': 0.005123,
    'H1 TOA': '',
    'H1 VALID': '',
    'H1 REASON': '',
    'H1 IS_NEARBY': '',
    'H1 PING_WIDTH': '',
    ... (repeat for H2, H3, all empty since not selected)
}

Write row to CSV
```

---

## 6. KEY ABSTRACTIONS & RESPONSIBILITIES

### Class Responsibility Map

| Class | File | Responsibility | Key Methods |
|-------|------|-----------------|-------------|
| **Hydrophone** | `hydrophones/hydrophone.py` | Container for one sensor's data | `reset()` |
| **HydrophoneArray** | `hydrophones/hydrophone_array.py` | Load from files, normalize, manage 4 hydrophones | `load_from_path()`, `plot_hydrophones()` |
| **BaseAnalyzer** | `analyzers/base_analyzer.py` | Common infrastructure (filtering, plotting, orchestration) | `analyze_array()`, `apply_bandpass()` |
| **TOAEnvelopeAnalyzer** | `analyzers/toa_envelope_analyzer.py` | Extract TOA using Hilbert envelope | `_analyze_single()` (returns TOA+validity) |
| **NearbyAnalyzer** | `analyzers/nearby_analyzer.py` | Detect nearby/far using ping width | `_analyze_single()` (returns nearby+delta_t) |
| **GarbageDetector** | `analyzers/garbage_detector.py` | Validate measurements | `validate_hydrophone_toa()` |
| **Logic2** | `logic/logic2.py` | Hardware interface (Saleae device) | `capture()`, `open()`, `close()` |

### Data Flow Responsibilities

| Component | Input | Responsibility | Output |
|-----------|-------|-----------------|--------|
| **Parser** | Directory paths | Orchestrate batch analysis, format CSV | CSV file |
| **Controller** | Data path (file/dir) | Coordinate capture, load, analyze | Flags + analyzer results |
| **HydrophoneArray** | Raw binary/CSV file | Load, demean, FFT | 4 Hydrophone objects with signal data |
| **Analyzers** | Hydrophone objects | Extract features (TOA, nearby status) | Per-hydrophone feature dicts |

---

## 7. CONFIGURATION & THRESHOLD MANAGEMENT

### Where Thresholds Are Defined

#### In `controller.py` (module level):
```python
ANALYZERS = [
    TOAEnvelopeAnalyzer(
        threshold_sigma=5,              # ← TOA threshold: mean + 5*std
        raw_signal_threshold=0.5,       # ← Min signal amplitude
        margin_front=0.1,               # ← Min seconds from start
        margin_end=0.1,                 # ← Min seconds from end
        filter_order=6,                 # ← Butterworth filter order
        search_band_min=30000,          # ← Freq range (Hz)
        search_band_max=34000,
        plot_results_flag=False
    ),
    NearbyAnalyzer(
        ping_width_threshold=0.014657,  # ← Nearby decision threshold (seconds)
        crossing_std_dev=5,             # ← Envelope threshold: mean + 5*std
        filter_order=6,
        search_band_min=30000,          # ← Freq range (Hz)
        search_band_max=34000,
        plot_results_flag=False
    ),
]
```

#### In `garbage_detector.py` (instantiated by TOAEnvelopeAnalyzer):
```python
self.garbage_detector = GarbageDetector(
    raw_signal_threshold=0.5,    # Passed from TOAEnvelopeAnalyzer
    margin_front=0.1,
    margin_end=0.1
)
```

### How Thresholds Are Used

| Threshold | Used By | Purpose | Example |
|-----------|---------|---------|---------|
| `threshold_sigma=5` | TOAEnvelopeAnalyzer | Envelope crossing threshold | When envelope > mean+5*std, mark as TOA |
| `ping_width_threshold=0.014657` | NearbyAnalyzer | Nearby decision rule | If pulse width < 14.6 ms, classify as nearby |
| `crossing_std_dev=5` | NearbyAnalyzer | Envelope for pulse detection | When envelope > mean+5*std, count as crossing |
| `raw_signal_threshold=0.5` | GarbageDetector | Min signal amplitude | Only valid if max(raw_signal) > 0.5 |
| `margin_front=0.1` | GarbageDetector | Minimum time from start | Reject if TOA < 0.1s (noise rejection) |
| `margin_end=0.1` | GarbageDetector | Minimum time from end | Reject if TOA > (duration - 0.1s) (incomplete signal) |
| `search_band_min/max=30000/34000` | Both analyzers | Frequency filtering | Only analyze 30-34 kHz band (ping frequency) |
| `filter_order=6` | Both analyzers | Butterworth sharpness | 6th order filter for frequency selectivity |

### Typical Tuning Considerations

For **feature extraction**, adjust these thresholds based on your signal characteristics:

```python
# For weak/far signals:
TOAEnvelopeAnalyzer(
    threshold_sigma=3,           # Lower threshold (more sensitive)
    raw_signal_threshold=0.1,    # Accept lower amplitudes
    margin_front=0.05, margin_end=0.05,  # Tighter margins
)

# For reducing false positives:
TOAEnvelopeAnalyzer(
    threshold_sigma=7,           # Higher threshold (more restrictive)
    raw_signal_threshold=1.0,    # Higher amplitude requirement
    margin_front=0.2, margin_end=0.2,   # Larger safety margins
)

# For varying ping characteristics:
NearbyAnalyzer(
    ping_width_threshold=0.020,  # Adjust based on actual widths
    crossing_std_dev=3,          # More sensitive detection
)
```

---

## 8. INFORMATION EXTRACTED AT EACH STAGE

### Raw Data Stage (HydrophoneArray)
- **Time array**: `hydrophone.times` (zero to ~2 seconds)
- **Raw signal**: `hydrophone.signal` (demeaned amplitude samples)
- **FFT**: `hydrophone.frequency` (frequency domain representation)
- **Metadata**: `hydrophone.sampling_period` (time between samples)

### TOA Extraction Stage
- **Time of arrival**: `toa_time` (seconds)
- **Sample index**: `toa_idx` (which sample exceeded threshold)
- **Envelope waveform**: `processed_signal` (Hilbert envelope)
- **Threshold value**: `threshold` (detection level used)
- **Validity flag**: `is_valid` (passed garbage detection)
- **Validity reason**: `validity_reason` (VALID/WEAK_SIGNAL/TOO_EARLY/TOO_LATE)

### Nearby Detection Stage
- **Nearby flag**: `nearby` (bool)
- **Pulse width**: `delta_t` (time between first and last threshold crossing)
- **Envelope threshold**: `threshold` (mean + crossing_std_dev*std)

### CSV Output Stage
- **Path identifier**: For data traceability
- **Ground truth labels**: Closest hydrophone, distance (from directory names)
- **Predictions**: TOA per hydrophone, validity flags, nearby status
- **Supporting data**: Ping widths, validation reasons

---

## 9. TYPICAL ANALYSIS WORKFLOW

### For Development/Debugging

```python
# controller.py: Single sample analysis
from controller import analyze_one_sample

data_path = "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08/2026-02-22--15-29-08_1"
is_nearby, is_valid, toa_results, nearby_results = analyze_one_sample(data_path)

# Inspect results
for toa in toa_results:
    print(f"H{toa['hydrophone_idx']}: TOA={toa['toa_time']:.6f}s, valid={toa['is_valid']}")

for nb in nearby_results:
    print(f"H{nb['hydrophone_idx']}: nearby={nb['nearby']}, delta_t={nb['delta_t']}")
```

### For Batch Processing

```python
# parser.py: Batch with CSV output
from parser import parse_recordings

paths = [
    "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08",
    "data/2.22.2026/H0_Closest_10FT_2026-02-22--15-35-26",
]

parse_recordings(paths, output_path="analysis")
# Creates: analysis/analysis_YYYY-MM-DD--HH-MM-SS.csv
```

### For Voting/Robustness

```python
# controller.py: Ensemble voting
from controller import run_voting_ensemble

result = run_voting_ensemble(num_votes_needed=5)
print(f"Final decision: {result['is_nearby']} (votes: {result['votes']})")
```

---

## 10. SUMMARY TABLE: Who Does What

| Task | Module | Class/Function | Input | Output |
|------|--------|-----------------|-------|--------|
| Capture hardware data | controller | `capture_data()` | prefix | data_path |
| Load from disk | controller | `load_hydrophone_data()` | data_path | ARRAY (global) |
| Parse binary format | hydrophones | `HydrophoneArray._load_from_bin()` | .bin file | 4 Hydrophone objects |
| Demean & FFT | hydrophones | `HydrophoneArray._update_hydrophone()` | raw signal | times, signal, frequency |
| Run TOA extraction | analyzers | `TOAEnvelopeAnalyzer.analyze_array()` | Hydrophone objects | List of TOA results |
| Run nearby detection | analyzers | `NearbyAnalyzer.analyze_array()` | Hydrophone objects | List of nearby results |
| Validate measurements | analyzers | `GarbageDetector.validate_hydrophone_toa()` | signal metadata | (is_valid, reason) |
| Single sample pipeline | controller | `analyze_one_sample()` | data_path | (is_nearby, is_valid, toa_results, nearby_results) |
| Batch analysis | parser | `parse_recordings()` | Directory list | CSV file |

---

## KEY INSIGHTS FOR FEATURE EXTRACTION

1. **Modularity**: Analyzers are pluggable via the `ANALYZERS` list in controller
2. **Extensibility**: To add new features, create new analyzer inheriting from `BaseAnalyzer`
3. **Bandpass filtering**: All analyzers share this common preprocessing (30-34 kHz)
4. **Envelope-based approach**: TOA uses Hilbert transform, nearby uses direct amplitude envelope
5. **Validation layer**: GarbageDetector provides crucial noise rejection
6. **Multi-rate capable**: Each hydrophone can have different `sampling_period`
7. **CSV generation**: Parser automates result collection and CSV formatting
