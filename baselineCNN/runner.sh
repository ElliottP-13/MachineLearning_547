for i in {1..10} ; do
    echo "FOLD ${i}"
    dataset_path=/mnt/data1/kwebst_data/data/GOOD_MEL_IMAGES/fold${i}
    output_path=/mnt/data1/kwebst_data/models/baseline_cnn/fold${i}
    python main.py --dataset $dataset_path --output_dir $output_path
done
