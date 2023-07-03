
#--reinitialize

python train_svhn.py --val --gap 400 --num_gaps 4 --layer_wise 6 --train_mode_epoch 150 --reinitialize 1 --MetaStartEpoch 0 --times 1  --initialize_type zero --meta_loss CE --file_name svhn_test1
