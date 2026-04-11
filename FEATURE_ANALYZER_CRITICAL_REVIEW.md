## CRITICAL REVIEW: FeatureAnalyzer Implementation

### Executive Summary
The FeatureAnalyzer contains **multiple critical bugs** that invalidate many of the features being extracted. The implementation is **NOT consistent** with the design intent, and comparison with existing analyzers reveals significant methodological inconsistencies.

**Status**: ❌ **DO NOT USE** for production until bugs are fixed

---

## CRITICAL BUGS

### BUG #1: Inverted Envelope Computation (CRITICAL)
**Severity**: HIGH - Invalidates most features  
**Location**: Lines 317-318

**Current Code**:
```python
raw_envelope = np.abs(scipy_signal.hilbert(raw_signal))
filtered_envelope = np.abs(filtered_signal)  # ❌ WRONG
```

**Problem**:
- Raw signal envelope uses Hilbert transform (correct analytical approach)
- Filtered signal envelope uses plain `np.abs()` on time-domain signal (inconsistent)
- TOA Analyzer (line 84) does: `envelope = np.abs(hilbert(filtered_signal))` ← CORRECT
- NearbyAnalyzer (line 73) does: `envelope = np.abs(filtered_signal)` ← Also not using Hilbert
- Feature Analyzer does the OPPOSITE: Hilbert on raw, np.abs() on filtered ← BACKWARDS

**Impact**:
- Filtered envelope not capturing true signal envelope (may be noisy/oscillating)
- All 26 FILTERED_* features based on this are potentially invalid
- Temporal features (rise_time, fall_time, FWHM, pulse_width, etc.) will be wrong
- Energy features (total_energy, early_window_energy) computed on wrong envelope
- Spectral features computed on wrong signal region identification

**Fix**:
```python
raw_envelope = np.abs(scipy_signal.hilbert(raw_signal))
filtered_envelope = np.abs(scipy_signal.hilbert(filtered_signal))  # ✓ Apply Hilbert to filtered too
```

---

### BUG #2: Temporal Features - Incorrect Threshold Crossing Logic
**Severity**: HIGH - Invalidates temporal features  
**Location**: Lines 65-89 in `_extract_temporal_features()`

**Current Code**:
```python
idx_10_before_up = None
for i in range(peak_idx):  # Searches from 0 to peak_idx
    if envelope[i] > thresh_10:
        idx_10_before_up = i
        break  # Stops at FIRST crossing, not LAST before peak
```

**Problem**:
- Finds the FIRST place where envelope > 10% threshold (possibly at start of signal)
- Should find the LAST place before peak where envelope < 10%
- This makes `rise_time` calculation span from signal start to peak, not actual pulse rise time
- Wrong for all threshold crossings (10%, 50%, 90%)

**Example Impact**:
- If signal starts at t=0 with noise, then ping arrives at t=0.5s
- Find first time envelope > 10% threshold: might be at t=0.001s
- Peak at t=0.7s
- Rise time calculated as: (0.7 - 0.001) * 1000 = 699ms ← WRONG
- Actual rise time of pulse: ~1-5ms ← Much shorter

**Affected Features**:
- rise_time_ms
- fall_time_ms  
- fwhm_ms
- pulse_width_ms
(×2 for RAW and FILTERED = 8 features affected)

**Fix**: Search backwards from peak:
```python
idx_10_before_up = None
for i in range(peak_idx, -1, -1):  # Search backwards from peak
    if envelope[i] < thresh_10:
        idx_10_before_up = i
        break
rise_time = ((peak_idx - idx_10_before_up) / sampling_freq * 1000) if idx_10_before_up is not None else None
```

---

### BUG #3: Rise Time vs Fall Time Inconsistency
**Severity**: MEDIUM - Inconsistent logic  
**Location**: Lines 72-86

**Current Code**:
```python
# Rise time - uses idx_10_before_up from buggy forward search
rise_time = ((peak_idx - idx_10_before_up) / sampling_freq * 1000)

# Fall time - searches forward correctly
idx_10_after = None
for i in range(peak_idx, len(envelope)):  # Forward search is correct here
    if envelope[i] < thresh_10:
        idx_10_after = i
        break
fall_time = ((idx_10_after - peak_idx) / sampling_freq * 1000)  # ✓ Correct formula
```

**Problem**:
- Rise time calculation searches FORWARD from 0 (bug #2)
- Fall time calculation searches FORWARD from peak (correct)
- These use entirely different methodologies
- Fall time will be reasonable, rise time will be garbage

**Impact**: Rise time feature is unreliable for distinguishing nearby vs far

---

### BUG #4: Single Peak Detection Applied to Both Envelopes (DESIGN FLAW)
**Severity**: MEDIUM - May miss true peaks  
**Location**: Line 319

**Current Code**:
```python
peak_idx = np.argmax(filtered_envelope)  # Peak from filtered domain only

# Same peak_idx used for BOTH:
temporal = self._extract_temporal_features(filtered_envelope, sampling_freq, peak_idx)
...
raw_temporal = self._extract_temporal_features(raw_envelope, sampling_freq, peak_idx)
```

**Problem**:
- Peak detected in filtered envelope (cleaner signal)
- Same peak index applied to raw envelope
- Raw signal may have peak at different location
- Envelopes may have different shapes/peaks

**Example Impact**:
- Filtered envelope peak: index 1,540,119 (clearest signal)
- Raw envelope peak: index 1,540,150 (30 samples offset)
- Using filtered peak for raw analysis misses true peak by 30 samples
- At 781.25 kHz = 38.4 µs offset

**Assessment**: 
- This is a design choice, but debatable
- May cause systematic bias in raw signal feature extraction
- Recommendation: Use separate peak detection for each envelope

**Better Approach**:
```python
peak_idx_filtered = np.argmax(filtered_envelope)
peak_idx_raw = np.argmax(raw_envelope)

# Extract features using appropriate peak for each envelope
temporal = self._extract_temporal_features(filtered_envelope, sampling_freq, peak_idx_filtered)
raw_temporal = self._extract_temporal_features(raw_envelope, sampling_freq, peak_idx_raw)
```

---

### BUG #5: Spectral Features - Parameter Naming Misleading
**Severity**: LOW - Confusing but currently works  
**Location**: Lines 186-232

**Current Code**:
```python
def _extract_spectral_features(self, filtered_signal, filtered_freqs):
    """Extract peak frequency, centroid, bandwidth, flatness."""
    spectrum = np.abs(fft(filtered_signal))  # Called on both RAW and FILTERED
```

**Problem**:
- Parameter named `filtered_signal` but used for both filtered and raw signals
- Docstring says "filtered_signal" but is actually generic
- Confusing to readers and maintainers

**Impact**: Low - code works, just misleading

**Fix**: Rename parameter:
```python
def _extract_spectral_features(self, signal, freqs):  # More generic
    """Extract peak frequency, centroid, bandwidth, flatness.
    
    Args:
        signal: Input signal (can be raw or filtered)
        freqs: Frequency bins from fftfreq
    """
    spectrum = np.abs(fft(signal))
```

---

### BUG #6: Energy Features - Mixed Semantics
**Severity**: LOW - Works but semantic issue  
**Location**: Lines 127-143

**Current Code**:
```python
total_energy = np.sum(envelope ** 2)  # Entire signal
early_energy = np.sum(envelope[peak_idx:early_end_idx] ** 2)  # Window from peak
energy_concentration = (early_energy / total_energy) * 100  # Ratio of part to whole
```

**Problem**:
- `total_energy` computed from index 0 (entire signal)
- `energy_concentration` is ratio of early window to entire signal
- For nearby detection, this might be fine (nearby = more energy in early window)
- But semantically mixing window-based analysis with full-signal basis

**Impact**: LOW - May still be discriminative, just semantically unclear

**Recommendation**: 
```python
# More semantically consistent:
early_energy = np.sum(envelope[peak_idx:early_end_idx] ** 2)
late_energy = np.sum(envelope[late_start_idx:late_end_idx] ** 2)
total_window_energy = early_energy + late_energy
energy_concentration = (early_energy / (total_window_energy + 1e-10)) * 100
```

---

### BUG #7: Envelope Computation Inconsistency Across Analyzers
**Severity**: MEDIUM - Design inconsistency  

**Codebase Analysis**:
```
TOAEnvelopeAnalyzer:
  envelope = np.abs(hilbert(filtered_signal))  ✓ Consistent

NearbyAnalyzer:
  envelope = np.abs(filtered_signal)  ← No Hilbert

FeatureAnalyzer:
  raw_envelope = np.abs(hilbert(raw_signal))  ✓ Correct
  filtered_envelope = np.abs(filtered_signal)  ✗ Inconsistent with TOA!
```

**Problem**:
- Three analyzers use three different envelope computation methods
- Makes it impossible to compare features across analyzers
- Creates systematic bias in feature values
- TOA uses Hilbert on filtered, NearbyAnalyzer doesn't, FeatureAnalyzer does backwards

**Impact**: 
- Feature values not comparable to other analyzer results
- ML models trained on these features may learn analyzer artifact, not true signal properties

---

## FEATURE COUNT DISCREPANCY

**Claimed**: "27+ features"  
**Actual**: 26 unique features per signal type

**Breakdown** (26 features × 2 signal types = 52 total):
1. rise_time_ms
2. fall_time_ms
3. fwhm_ms
4. pulse_width_ms
5. total_energy
6. early_window_energy
7. late_window_energy
8. early_late_ratio
9. energy_concentration_pct
10. peak_amplitude
11. dynamic_range
12. crest_factor
13. snr_db
14. peak_snr_db
15. noise_std
16. peak_frequency_hz
17. spectral_centroid_hz
18. spectral_bandwidth_hz
19. spectral_flatness
20. peak_floor_ratio
21. secondary_peak_count
22. secondary_peak_amplitude_ratio
23. time_to_secondary_peak_ms
24. envelope_skewness
25. envelope_kurtosis
26. envelope_compactness

**Minor Issue**: Update docstring from "27+ features" to "26 features per signal type"

---

## DATA VALIDATION - Test Results Show Wrong Values

**Example from test_features.py**:

Epoch 0 (H0):
```
FILTERED_rise_time_ms: 0.5158 ms
FILTERED_fall_time_ms: 0.0077 ms  
FILTERED_pulse_width_ms: 0.52352 ms
```

PROBLEM: rise_time (0.516 ms) < pulse_width (0.524 ms) ← WRONG!  
Pulse width should be rise + fall time, so:
```
pulse_width = rise_time + fall_time
0.524 ≟ 0.516 + 0.008 = 0.524 ✓ Actually matches!
```

But if pulse_width is correctly calculated as:
```
pulse_width_ms = (idx_10_after - idx_10_before_up) / sampling_freq * 1000
```

And rise_time uses FIRST crossing (bug), then:
- idx_10_before_up = very early index (wrong)
- idx_10_after = correct index  
- rise_time = peak_idx - idx_10_before_up (very large)
- pulse_width = idx_10_after - idx_10_before_up (very large)

The test values are UNREALISTICALLY SMALL. Typical pings have:
- Pulse width: several milliseconds (2-10 ms typical)
- Current values: 0.5-1.2 ms ← Could be real, but suspicious

**Hypothesis**: Threshold crossings being found correctly by chance, but logic is still wrong

---

## RECOMMENDATIONS

### IMMEDIATE FIXES (BEFORE USE):

1. **Fix Bug #1**: Apply Hilbert to filtered envelope
   ```python
   filtered_envelope = np.abs(scipy_signal.hilbert(filtered_signal))
   ```

2. **Fix Bug #2**: Correct threshold crossing logic
   ```python
   # Search backwards from peak for proper rise/fall times
   ```

3. **Fix Bug #4**: Use separate peak detection
   ```python
   peak_idx_filtered = np.argmax(filtered_envelope)
   peak_idx_raw = np.argmax(raw_envelope)
   ```

4. **Document Decision**: If bug #1 and #4 are deliberately chosen, explicitly justify in comments

### TESTING NEEDED:

1. Compare FILTERED feature values with TOA analyzer envelope
2. Verify rise_time + fall_time ≈ pulse_width for all samples
3. Validate that nearby signals have different feature values than far signals
4. Check for NaN/Inf values in edge cases
5. Verify feature ranges are physically reasonable (ms, dB, Hz scales)

### VALIDATION CHECKLIST:

- [ ] Envelope computation matches TOA analyzer
- [ ] Temporal features satisfy: rise_time + fall_time ≈ pulse_width
- [ ] SNR values are 10-50 dB range (typical for signals)
- [ ] Peak frequency within bandpass range (30-34 kHz)
- [ ] No negative energy values
- [ ] No NaN/Inf in any feature
- [ ] Temporal features monotonically increase from 0% to peak
- [ ] Features differ significantly between nearby and far samples

---

## CURRENT STATUS

**✗ Non-functional**: Multiple critical bugs prevent reliable feature extraction  
**✗ Untested**: Features have never been validated against ground truth  
**✗ Inconsistent**: Contradicts envelope computation in other analyzers  

**Next Step**: Fix bugs #1, #2, #4 and re-validate

