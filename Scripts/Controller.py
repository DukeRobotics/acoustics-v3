import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array

SAMPLING_FREQ = 781250
SELECTED = [True, False, True, False]
prefix = ""
time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
folder = prefix + time_stamp

base_path = os.path.join("Scripts", folder)
os.mkdir(base_path)

logic = LOGIC.Logic(sampling_freq=SAMPLING_FREQ)
logic.print_saleae_status()

# path = logic.start_csv_capture(2, base_path)
path = logic.export_binary_capture(2, base_path)
# path, csv_path = logic.export_binary_and_csv_capture(2, base_path)

# logic.kill_logic()

# path = "Scripts/2025-10-09--18-18-31_1-0/SAMPLE.csv"
# path = "Scripts/2025-10-09--18-20-27_1-0/SAMPLE.csv"
# path = "Scripts/2025-10-09--18-21-04_1-0/SAMPLE.csv"
# path = "Scripts/2025-10-09--18-43-46_0-1/SAMPLE.csv"
# path = "Scripts/2025-10-09--18-47-46_0-1/SAMPLE.csv"
# path = "Scripts/2025-11-11--00-25-09/TEMP.bin"
# path = "Scripts/0_Data_Collection/0_2025-11-16--15-31-44.bin"
# path = "Scripts/0_Data_Collection/0_2025-11-16--15-32-19.bin"
# path = "Scripts/2_Data_Collection/2_2025-11-16--15-19-34.bin"
# path = "Scripts/2_Data_Collection/2_2025-11-16--15-27-41.bin"
# path = "Scripts/Acoustics_Thruster_Data_Amp_Then_Filter_11_20.csv"
# path = "Scripts/Acoustics_Thruster_Data_Filter_Then_Amp_11_20.csv"
# path = "Scripts/Acoustics_Thruster_Data_Raw_Hydrophone_11_20.csv"

hydrophone_array = Hydrophone_Array.HydrophoneArray(sampling_freq=SAMPLING_FREQ)
hydrophone_array.load_from_path(path)

hydrophone_array.estimate_selected_by_envelope(SELECTED)
hydrophone_array.print_envelope_toas()
hydrophone_array.plot_selected_envelope(SELECTED, show_frequency_domain=True)
print("=" * 30)

hydrophone_array.estimate_selected_by_gcc(SELECTED, use_multi_reference=True)
hydrophone_array.print_gcc_tdoa(SELECTED)
# hydrophone_array.plot_selected_gcc(SELECTED)
print("=" * 30)


