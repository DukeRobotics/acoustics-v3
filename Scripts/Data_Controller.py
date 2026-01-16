import os
import matplotlib.pyplot as plt
import time
import Logic as LOGIC
import Hydrophone_Array

SAMPLING_FREQ = 781250
SELECTED = [True, True, True, True]
# what is the truth of the hydrophone
prefix = "testgi_"
data_collection = prefix + "Data_Collection"
base_path = os.path.join("Scripts", data_collection)
os.makedirs(base_path, exist_ok = True)

logic = LOGIC.Logic(sampling_freq=SAMPLING_FREQ)
logic.print_saleae_status()

epochs = 10
for epoch in range (0, epochs):
    print(f"epoch {epoch}")
    
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S')
    name = prefix + time_stamp
    logic.export_binary_capture(2, base_path, name)

logic.kill_logic()

