import argparse
import h5py
import pandas as pd
import os
import numpy as np
import tqdm

def get_starts_durations(mask):
    starts = []
    durations = []
    current_start = -1
    current_duration = -1
    for i, ap in enumerate(mask):
        if ap == 1:
            if current_start == -1:
                current_start = i
                current_duration = 1
            else:
                current_duration += 1
        if ap == 0:
            if current_start != -1:
                starts.append(current_start)
                durations.append(current_duration)
                current_start = -1
                current_duration = -1
    return starts, durations

def single_h5_to_records(input_h5_path, output_folder, target_path=None):
    input_h5 = h5py.File(input_h5_path)
    if target_path is not None:
        input_labels = pd.read_csv(target_path, index_col="ID").values
        assert input_labels.shape[1] == 90
    
    signals = ['abdom_belt','airflow','PPG','thorac_belt','snore','SPO2','C4-A1','O2-A1']

    data = input_h5["data"]
    for i in tqdm.tqdm(range(data.shape[0]//200)):
        subject_id = input_h5["data"][200*i,1]
        with h5py.File(os.path.join(output_folder, "{}.h5".format(int(subject_id))), mode="w") as record:
            if target_path is not None:
                starts, durations = get_starts_durations(input_labels[200*i:200*(i+1),:].flatten())
                record.create_dataset('apnea/start', data=np.array(starts))
                record.create_dataset('apnea/duration', data=np.array(durations))
            
            for j, signal in enumerate(signals):
                idx_slice = slice(200*i, 200*(i+1))
                col_slice = slice(2+j*9000, (j+1)*9000)
                dset = record.create_dataset(signal, data=data[idx_slice, col_slice].flatten())
                dset.attrs["fs"] = 100

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5")
    parser.add_argument("--input-labels")
    parser.add_argument("--output-folder")
    args = parser.parse_args()
    single_h5_to_records(args.input_h5, args.output_folder, args.input_labels)