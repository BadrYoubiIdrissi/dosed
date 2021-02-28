python dosed/datasets/single_h5_to_records_format.py \
        --input-h5 ~/datasets/sleepapnea/raw/X_train.h5 \
        --input-labels ~/datasets/sleepapnea/raw/y_train_tX9Br0C.csv \
        --output-folder ~/datasets/sleepapnea/records/train/

python dosed/datasets/single_h5_to_records_format.py \
        --input-h5 ~/datasets/sleepapnea/raw/X_test.h5 \
        --output-folder ~/datasets/sleepapnea/records/test/