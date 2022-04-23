for i in {2..10} ; do
    echo "FOLD ${i}"
    dataset_path=/mnt/data1/kwebst_data/data/NSYNTH_MEL_IMAGES/
    output_path=/mnt/data1/kwebst_data/models/baseline_cnn/NSYNTH/fold${i}
    python main.py --dataset $dataset_path --output_dir $output_path --epochs 10
done
