import saleae.automation as automation
from saleae.automation import Manager
import os
import sys

class Logic():
    def __init__(self, sampling_freq = 781250, logic_path = ""):
        self.LAUNCH_TIMEOUT = 15
        self.QUIET = False
        self.PORT = 10429
        self.HOST = 'localhost'

        # TODO: Put in images for Logic 2
        if logic_path != "":
            self.LOGIC_PATH = logic_path
        elif sys.platform == "win32":
            self.LOGIC_PATH = "Logic_Software/Logic-1.2.40_WINDOWS/Logic.exe"
        elif sys.platform == "darwin":
            self.LOGIC_PATH = "Logic-1.2.40-MacOS.dmg"
        elif sys.platform == "linux":
            self.LOGIC_PATH = "Logic_Software/Logic-1.2.40-Linux.AppImage"
        else:
            print(f"Unknown OS: {sys.platform}")


        self.DEVICE_SELECTION = 3    # 6 for LOGIC PRO 16, 3 for LOGIC 8, 5 for LOGIC PRO 8
        self.SAMPLING_FREQ = sampling_freq
        self.H0_CHANNEL = 0
        self.H1_CHANNEL = 1
        self.H2_CHANNEL = 2
        self.H3_CHANNEL = 3
        self.CHANNELS = [self.H0_CHANNEL, self.H1_CHANNEL, self.H2_CHANNEL, self.H3_CHANNEL]
        self.DEVICE_CONFIGURATION = automation.LogicDeviceConfiguration(
            enabled_analog_channels=self.CHANNELS,
            analog_sample_rate=self.SAMPLING_FREQ
        )

        self.s = self.start_logic()

        for device in self.s.get_devices():
            if device.device_type == self.DEVICE_SELECTION:
                self.DEVICE_ID = device.device_id

        assert self.DEVICE_ID is not None

    def start_logic(self):
        try:
            return Manager.connect(port=self.PORT)
        except Exception:
            return Manager.launch(application_path=self.LOGIC_PATH,
                                  connect_timeout_seconds=self.LAUNCH_TIMEOUT,
                                  port=self.PORT)

    def kill_logic(self):
        self.s.close()

    def print_saleae_status(self):
        devices = self.s.get_devices()

        print("DEBUG: CONNECTED DEVICES:")
        for d in devices:
            print(f"  ID: {d.device_id}")
            print(f"  Type: {d.device_type}")

    def export_capture(self, seconds, output_dir, capture_binary, capture_csv):
        capture_configuration = automation.CaptureConfiguration(
            capture_mode=automation.TimedCaptureMode(duration_seconds=seconds)
        )

        with self.s.start_capture(
            device_id=self.DEVICE_ID,
            device_configuration=self.DEVICE_CONFIGURATION,
            capture_configuration=capture_configuration,
        ) as capture:
            capture.wait()

            if capture_csv:
                capture.export_raw_data_csv(directory=output_dir, analog_channels=self.CHANNELS)

            if capture_binary:
                capture.export_raw_data_binary(directory=output_dir, analog_channels=self.CHANNELS)

        return os.path.join(output_dir, "analog.bin"), os.path.join(output_dir, "analog.csv")
