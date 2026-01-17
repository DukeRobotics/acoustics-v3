"""
logic_capture.py

Wrapper utilities for launching, configuring, and capturing data from
a Saleae Logic analyzer using the Saleae Python API.

This module abstracts:
- Launching the Saleae Logic software
- Device and channel configuration
- Timed data capture
- Exporting captures to CSV and/or binary formats

Intended for scripted, repeatable data acquisition workflows.
"""

import subprocess
import saleae
import time
import os 
import sys

class Logic():
    """
    High-level controller for a Saleae Logic analyzer.

    This class handles:
    - Launching the Saleae Logic application (if needed)
    - Selecting devices and channels
    - Configuring sampling rates
    - Capturing and exporting analog data

    Parameters
    ----------
    sampling_freq : int, optional
        Desired sampling frequency in Hz. Defaults to 781250.
    """

    def __init__(self, sampling_freq = 781250):
        """
        Initialize the Logic controller and connect to Saleae Logic.

        This constructor:
        - Determines the OS-specific Logic executable path
        - Launches the Logic application if required
        - Connects to the Saleae API server
        - Configures device, channels, and sampling rate

        Parameters
        ----------
        sampling_freq : int, optional
            Desired sampling frequency in Hz. Defaults to 781250.
        """
        self.LAUNCH_TIMEOUT = 15
        self.QUIET = False
        self.PORT = 10429
        self.HOST = 'localhost'

        if sys.platform == "win32":
            self.LOGIC_PATH = "Logic-1.2.40-Windows/Logic-1.2.40/Logic.exe"
        elif sys.platform == "darwin":
            self.LOGIC_PATH = "Logic-1.2.40-MacOS.dmg"
        elif sys.platform == "linux":
            self.LOGIC_PATH = "Logic-1.2.40-Linux.AppImage"
        else:
            print(f"Unknown OS: {sys.platform}")

        
        self.DEVICE_SELECTION = 1    # 0 for LOGIC PRO 16, 1 for LOGIC 8, 2 for LOGIC PRO 8
        self.SAMPLING_FREQ = sampling_freq
        self.H0_CHANNEL = 0
        self.H1_CHANNEL = 1
        self.H2_CHANNEL = 2
        self.H3_CHANNEL = 3
        self.CHANNELS = [self.H0_CHANNEL, self.H1_CHANNEL, self.H2_CHANNEL, self.H3_CHANNEL]
        
        if (not sys.platform == "linux"):
            self.start_logic()
        self.s = saleae.Saleae(host=self.HOST, port=self.PORT, quiet=self.QUIET)
        self.launch_configure()

    def start_logic(self): 
        """
        Launch the Saleae Logic application if it is not already running.

        Returns
        -------
        bool
            True if Logic is running or successfully launched.
        """
        if (not saleae.Saleae.is_logic_running()):
            return saleae.Saleae.launch_logic(timeout=self.LAUNCH_TIMEOUT, quiet=self.QUIET, 
                                              host=self.HOST, port=self.PORT, logic_path=self.LOGIC_PATH)
        return True

    def kill_logic(self):
        """
        Forcefully terminate the Saleae Logic application.
        """
        saleae.Saleae.kill_logic()

    def launch_configure(self):
        """
        Configure the connected Saleae device.

        This method:
        - Selects the active device
        - Enables the specified analog channels
        - Sets the sampling rate based on a minimum requirement
        """
        self.s.select_active_device(self.DEVICE_SELECTION)
        self.s.set_active_channels(digital=None, analog=self.CHANNELS)
        self.s.set_sample_rate_by_minimum(0,self.SAMPLING_FREQ)

    def print_saleae_status(self):
        """
        Print diagnostic information about the Saleae Logic state.

        Useful for debugging connection issues, configuration errors,
        or performance constraints.
        """
        print(f"DEBUG: IS LOGIC RUNNING: {self.s.is_logic_running()}")  
        print(f"DEBUG: CONNECTED DEVICE: {self.s.get_connected_devices()}")
        print(f"DEBUG: PERFORMANCE: {self.s.get_performance()}")  
        print(f"DEBUG: ACTIVE CHANNELS: {self.s.get_active_channels()}") 
        print(f"DEBUG: POSSIBLE SAMPLING RATES: {self.s.get_all_sample_rates()}")
        print(f"DEBUG: SAMPLING RATE: {self.s.get_sample_rate()}")
        print(f"DEBUG: POSSIBLE BANDWIDTH: {self.s.get_bandwidth(self.s.get_sample_rate())}")  
        print(f"DEBUG: ANALYZERS: {self.s.get_analyzers()}")  
        
    def start_csv_capture(self, seconds, output_dir):
        """
        Capture analog data and export it as a CSV file.

        Parameters
        ----------
        seconds : float
            Duration of the capture in seconds.
        output_dir : str
            Directory where the CSV file will be saved.

        Returns
        -------
        str
            Full path to the generated CSV file.
        """
        csv_path = os.path.join(output_dir,"TEMP.csv")
        self.s.set_capture_seconds(seconds)
        self.s.capture_start_and_wait_until_finished()
        self.s.export_data2(file_path_on_target_machine=csv_path, format='csv')
        while(not self.s.is_processing_complete()):
            time.sleep(0.5)
        return csv_path
    
    def export_binary_capture(self, seconds, output_dir, name = "TEMP.bin"):
        """
        Capture analog data and export it in binary format.

        Parameters
        ----------
        seconds : float
            Duration of the capture in seconds.
        output_dir : str
            Directory where the binary file will be saved.
        name : str, optional
            Name of the binary file. Defaults to "TEMP.bin".

        Returns
        -------
        str
            Full path to the generated binary file.
        """
        bin_path = os.path.join(output_dir, name)
        self.s.set_capture_seconds(seconds)
        self.s.capture_start_and_wait_until_finished()
        self.s.export_data2(file_path_on_target_machine=bin_path, analog_channels=self.CHANNELS, format='binary')
        while(not self.s.is_processing_complete()):
            time.sleep(0.5)
        return bin_path
    
    def export_binary_and_csv_capture(self, seconds, output_dir):
        """
        Capture analog data and export both binary and CSV versions.

        Parameters
        ----------
        seconds : float
            Duration of the capture in seconds.
        output_dir : str
            Directory where output files will be saved.

        Returns
        -------
        tuple of str
            Paths to the generated (binary_path, csv_path).
        """
        bin_path = os.path.join(output_dir, f"TEMP.bin")
        csv_path = os.path.join(output_dir,"TEMP.csv")

        self.s.set_capture_seconds(seconds)
        self.s.capture_start_and_wait_until_finished()
        self.s.export_data2(file_path_on_target_machine=bin_path, analog_channels=self.CHANNELS, format='binary')
        while(not self.s.is_processing_complete()):
            time.sleep(0.5)
       
        self.s.export_data2(file_path_on_target_machine=csv_path, format='csv')
        while(not self.s.is_processing_complete()):
            time.sleep(0.5)
            
        return bin_path, csv_path