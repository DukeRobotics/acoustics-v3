"""Logic 2 module for interfacing with Saleae Logic 2 hardware."""
import os
import sys
from pathlib import Path

from saleae.automation import (
    Manager,
    LogicDeviceConfiguration,
    TimedCaptureMode,
)


class Logic2:
    """Interface for Saleae Logic 2 data acquisition hardware."""
    
    def __init__(self, sampling_freq=781250, logic_path=""):
        self._sampling_freq = sampling_freq
        self._port = 10430  # Default Logic 2 port
        self._host = '127.0.0.1'
        self._launch_timeout = 15
        
        if logic_path != "":
            self._logic_path = logic_path
        else:
            # Let Manager.launch() auto-detect the application
            self._logic_path = None
        
        # Device configuration
        self._device_selection = None  # Will be set by get_devices()
        self._h0_channel = 0
        self._h1_channel = 1
        self._h2_channel = 2
        self._h3_channel = 3
        self._channels = [self._h0_channel, self._h1_channel,
                         self._h2_channel, self._h3_channel]
        
        self._manager = None
        self._current_capture = None
        
        # Start Logic 2 and initialize manager
        self.start_logic()
    
    def start_logic(self):
        """Start the Logic 2 software if not already running."""
        try:
            # Try to connect to existing instance first
            self._manager = Manager.connect(
                address=self._host,
                port=self._port,
                connect_timeout_seconds=5
            )
        except Exception:
            # If connection fails, launch a new instance
            if self._logic_path:
                self._manager = Manager.launch(
                    application_path=self._logic_path,
                    connect_timeout_seconds=self._launch_timeout,
                    port=self._port
                )
            else:
                self._manager = Manager.launch(
                    connect_timeout_seconds=self._launch_timeout,
                    port=self._port
                )
        
        self._configure_device()
        return True
    
    def kill_logic(self):
        """Close the Logic 2 manager (graceful shutdown)."""
        if self._current_capture is not None:
            try:
                self._current_capture.close()
            except Exception:
                pass
        
        if self._manager is not None:
            try:
                self._manager.close()
            except Exception:
                pass
    
    def _configure_device(self):
        """Configure the active Logic device with channels and sample rate."""
        # Get available devices
        devices = self._manager.get_devices()
        if not devices:
            raise RuntimeError("No Logic 2 devices found")
        
        # Use first device
        self._device_id = devices[0].device_id
    
    def _create_device_config(self):
        """Create LogicDeviceConfiguration for current setup."""
        return LogicDeviceConfiguration(
            enabled_analog_channels=self._channels,
            analog_sample_rate=self._sampling_freq
        )
    
    def start_csv_capture(self, seconds, output_dir):
        """Capture data and export to CSV format."""
        csv_path = os.path.join(output_dir, "TEMP.csv")
        
        # Create device configuration
        device_config = self._create_device_config()
        
        # Create capture configuration for timed capture
        capture_config = TimedCaptureMode(duration_seconds=seconds)
        
        # Start capture
        self._current_capture = self._manager.start_capture(
            device_id=self._device_id,
            device_configuration=device_config,
            capture_configuration=capture_config
        )
        
        # Wait for capture to complete
        self._current_capture.wait()
        
        # Export to CSV
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self._current_capture.export_raw_data_csv(
            directory=output_dir,
            analog_channels=self._channels
        )
        
        # Close the capture
        self._current_capture.close()
        self._current_capture = None
        
        return csv_path
    
    def export_binary_capture(self, seconds, output_dir, name="TEMP.bin"):
        """Capture data and export to binary format."""
        bin_path = os.path.join(output_dir, name)
        
        # Create device configuration
        device_config = self._create_device_config()
        
        # Create capture configuration for timed capture
        capture_config = TimedCaptureMode(duration_seconds=seconds)
        
        # Start capture
        self._current_capture = self._manager.start_capture(
            device_id=self._device_id,
            device_configuration=device_config,
            capture_configuration=capture_config
        )
        
        # Wait for capture to complete
        self._current_capture.wait()
        
        # Export to binary
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self._current_capture.export_raw_data_binary(
            directory=output_dir,
            analog_channels=self._channels
        )
        
        # Close the capture
        self._current_capture.close()
        self._current_capture = None
        
        return bin_path
    
    def export_binary_and_csv_capture(self, seconds, output_dir):
        """Capture data and export to both binary and CSV formats."""
        bin_path = os.path.join(output_dir, "TEMP.bin")
        csv_path = os.path.join(output_dir, "TEMP.csv")
        
        # Create device configuration
        device_config = self._create_device_config()
        
        # Create capture configuration for timed capture
        capture_config = TimedCaptureMode(duration_seconds=seconds)
        
        # Start capture
        self._current_capture = self._manager.start_capture(
            device_id=self._device_id,
            device_configuration=device_config,
            capture_configuration=capture_config
        )
        
        # Wait for capture to complete
        self._current_capture.wait()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to binary
        self._current_capture.export_raw_data_binary(
            directory=output_dir,
            analog_channels=self._channels
        )
        
        # Export to CSV
        self._current_capture.export_raw_data_csv(
            directory=output_dir,
            analog_channels=self._channels
        )
        
        # Close the capture
        self._current_capture.close()
        self._current_capture = None
        
        return bin_path, csv_path