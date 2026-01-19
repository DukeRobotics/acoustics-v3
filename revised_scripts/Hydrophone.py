from typing import Optional
import numpy as np

class Hydrophone:
    times: Optional[np.ndarray] = None
    signal: Optional[np.ndarray] = None
    filtered_signal: Optional[np.ndarray] = None
    
    freqs: Optional[np.ndarray] = None
    frequency: Optional[np.ndarray] = None
    filtered_frequency: Optional[np.ndarray] = None
    
    def reset(self):
        self.times = None
        self.signal = None
        self.filtered_signal = None
        self.freqs = None
        self.frequency = None
        self.filtered_frequency = None