# Acoustics V3

___

## One Time SETUP (ONLY SUPPORTING WINDOWS rn)
1. Clone Repo
2. Open Logic standalone through command line @`Logic_Software\Logic-1.2.40_WINDOWS\Logic.exe`
3. Click Options drop down in top right corner
4. Click Preferences
5. Click Developer Tab
6. Click Option to Enable Scripting Socket Server
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
- `Scripts\Hydrophone_Array.py` Break out frequency spectrum graph from envelope plot. Make it its own independant feature. and something envelope and gcc can call. You can send it either raw signal, the processed envelope signal, or processed gcc signal. Right now it only shows the processed envelope signal. 
- `Scripts\Hydrophone_Array.py` FIX or Disable GCC. Too complicated right now with little understanding. Go back to basics. Check earlier version from acoustic v2 for reference. 
- `Scripts\Hydrophone_Array.py` Look into the effects of envelope whitening. May be a better way to normalize data and minimize per_hydrophone variation
- `Scripts\Hydrophone_Array.py` Large Fix. Re-organize script contents. Right now its very messy in terms of organization. A bit of AI-written code that makes following very difficult. It was made piecemail. No real final architecture in mind so just messy.  
- Make `nn_train.py` script more friendly for feature selection (what cols and # of cols to select for features from csv)
- Make `nn_train.py` easier to configure confidence thresholds 
- `read_data/` Fix timing, currently time values are slightly off from true time. Ahaan used [Imhex|https://github.com/WerWolv/ImHex] to manually parse the file. He can tell you exactly bit-wise file format. 
- Organize files within repo
- Introduce back Linux image and mac image. Make mac image easier to use (so don't have to initiallize every time). Linux worked at one point. Clear robot cache or temp and repo instantiation and try again. Be careful closing Saleae software in the middle of it being controlled by python. Can cause errors. troubleshooting: clear cache/temp and pray. (This is what happened to the image on the robot. and why it can no longer can be opened by the code)

___

## Script Usage Guide

### Controller.py
**Purpose:** Single data capture and analysis workflow

**Use Case:** Capture new data from Logic analyzer OR analyze historical data files

**Key Parameters:**
- `CAPTURE`: Set to `True` for new capture, `False` to use historical data
- `CAPTURE_TIME`: Duration of capture in seconds
- `CAPTURE_FORMAT`: Choose `.bin`, `.csv`, or `both`
- `HISTORICAL_PATH`: Path to existing data file (when `CAPTURE=False`)
- `PLOT_ENVELOPE` / `PLOT_GCC`: Enable/disable visualization


