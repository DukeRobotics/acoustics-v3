# Acoustics V2

___

## One Time SETUP (ONLY SUPPORTING MAC + WINDOWS rn)
1. Clone Repo
2. Open Saleae standalone
3. Click Options drop down in top right corner
4. Click Preferences
5. Click Developer Tab
6. Click Option to Enable Scripting Socket Server
7. Download needed software (.AppImage or .dmg):
  - https://support.saleae.com/product/logic-software/legacy-software/older-software-releases
  - ./Logic-1.2.40-Linux.AppImage --appimage-extract
8. Create a new python enviornment and pip install requirements.txt

___

## Running Files

___

### Store Data
Data to be stored in the Duke Box, not in this repository:
https://duke.app.box.com/folder/361082889246?s=nggecd1enqpmz28wdk4jf7xjidw6crw0

### Neural Network
#### Train Data
python src/nn/nn_train.py --csv data/nn_sample_data.csv --epochs 100 --lr 1e-3

#### Predictions
python src/nn/nn_predict.py --ckpt artifacts/hydrophone_model.pt --csv data/nn_sample_data.csv --out probs.csv

___

### Read in Binary Files
#### Create executable (C++)
Create .exe file
- g++ -std=c++17 -O2 src\read_data\read_bin_data.cpp -o read_bin.exe

Read a given analog file
- read_bin.exe data\analog_0.bin

___

## TODO
- Fix timing, currently time values are slightly off from true time
- Organize files within repo