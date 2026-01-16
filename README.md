# Acoustics V2
## One - Time SETUP (ONLY SUPPORTING MAC + WINDOWS rn)
Clone Repo
Open Saleae standalone
--> Click Options drop down in top right corner
--> Click Preferences
--> Click Developer Tab
--> Click Option to Enable Scripting Socket Server

https://support.saleae.com/product/logic-software/legacy-software/older-software-releases
./Logic-1.2.40-Linux.AppImage --appimage-extract

Create a new python enviornment and download requirements.txt

## Running Files
### Neural Network
#### Train Data
python src/nn_train.py --csv data/nn_sample_data.csv --epochs 100 --lr 1e-3

#### Predictions
python src/nn_predict.py --ckpt artifacts/hydrophone_model.pt --csv data/nn_sample_data.csv --out probs.csv

### Read in Binary Files
#### Create executable (C++)
Create .exe file
- g++ -std=c++17 -O2 src\read_bin_data.cpp -o read_bin.exe

Read a given analog file
- read_bin.exe data\analog_0.bin
- 
- TODO: Fix timing, currently time values are slightly off from true time