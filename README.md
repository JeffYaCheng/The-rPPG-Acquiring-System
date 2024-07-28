# The rPPG Acquiring System
The rPPG acquiring system. 
- Using Trianswer to obtain ECG and PPG signals.
- Using a camera to get video.
# Operating Procedures
1.  Turn on the Trianswer ([Detailed process](https://hackmd.io/cn5WpB78Qp21ZtBuHi6e-Q?view))
2.  Run the code to collect ECG, PPG and images 
    - Run command：`python Trianswer.py` 
3. You will see the following UI
    - ![](https://hackmd.io/_uploads/SyGyH-zn2.png)
    - Press the following buttons to save signals 
        - Scan：Search Trianswer
        - Choose your Trianswer id
        - Connect to device
        - Adjust Light
        - Start REC
            - Files will save in the `./TriAnswer_Records` directory
        - Stop REC
4.  Run the code to synchronize the three data types
     - Run command: `python synchronize.py`
        - Files will be saved in the `./Normalize` directory
    - If you want to choose the input and output data files, you can run:
        - `python synchronize.py -i input_file_path -o output_file_path`
        - Default `input_file_path`: `./TriAnswer_Records`
        - Default `save_file_path`: `./Normalize`

# Saving Files Format
## TriAnswer_Records
 -----------------
    TriAnswer_Records/
    ├── 2023_08_05_22_37_07/
    │ ├── img/
    │ ├── ECG.csv
    │ ├── PPG_IR.csv
    │ ├── PPG_R.csv
    │ └── ECG_PPG_time_step.txt
    └── 2023_08_06_21_47_37/
 -----------------
 
- ECG_PPG_time_step.txt format
    - `time, ECG_signal_points, PPG_R_signal_points, PPG_IR_signal_points`
    - Example: `1720503173.519677,36,2,2` - We get 36 ECG, 2 PPG_R, and 2 PPG_IR points at time `1720503173.519677`
- ECG sample rate: `1000`
- PPGIR and PPGR sample rate: `100`
- Image sample rate: `30`
## Normalize
 -----------------
    Normalize/
    ├── 2023_08_05_22_37_07/
    │ ├── img/
    │ ├── raw_data/
    │ ├── syn_ECG.csv
    │ ├── syn_PPG_IR.csv
    │ ├── syn_PPG_R.csv
    │ ├── output.avi
    │ ├── syn_information.csv
    │ └── syn_time_step.txt
    └── 2023_08_06_21_47_37/
 -----------------

- This img folder will have some images deleted from the raw data:
    - Some images might not have corresponding signals.
- Raw data will copy the following files:
    - `ECG.csv`
    - `PPG_IR.csv`
    - `PPG_R.csv`
    - `ECG_PPG_time_step.txt`
- syn_time_step.txt format:
    - `time ECG_signal_points PPG_R_signal_points PPG_IR_signal_points number_of_images`
    - Example: `1720503037.2 60 3 3 3` - We get 60 ECG, 3 PPG_R, 3 PPG_IR points, and 3 images at time `1720503037.2`
    - Since synchronization is per 0.1 second, the time precision is 0.1s.
- syn_ECG sample rate: `600`
- syn_PPGIR and PPGR sample rate: `30`
- Image sample rate: `30`