## Use this for BIQA

1. Generate file list

```bash
# Converting data format
./gen_filelist_live.py \
    /data_partition/yang/new_exp_dbs/databaserelease2 \
    /data_partition/yang/new_exp_dbs/databaserelease2/filelist.csv 

# Dividing the dataset into training and testing set
./gen_dataset_live.py \
    /data_partition/yang/new_exp_dbs/databaserelease2/filelist.csv \
    /data_partition/yang/new_exp_dbs/databaserelease2
```

2. Train and test the model 

```bash
./Main.py --ckpt_path checkpoint-live-corr-n-03 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001
# If you want to train from scratch, add a parameter -f 
```

## Main idea

L2R out-performs conventional regression, because it is able to compare scores.  Then, why not compare more?  This time, the loss is like batched PLCC.  See `CorrelationLoss` in `model.py`.

The training and testing sets are divided using similar methods to UNIQUE, except that only LIVE is used. 

SRCC (of test set) goes beyond 0.95 for LIVE in several minutes. 

### Limitations

Need large GRAM

This model is not perfect, and has some potential for improvements, for example, it only accepts fixed size of images, and will randomly crop test images. 



## Adversarial Training

Batched PLCC used together with FGSM makes a model robust to both FGSM and the stronger search method (should be stronger than PGD methods) 

However, Vanilla L2R (as defined in `README.md`) used with FGSM does not enhance robustness at all. 

Batched SRCC with FGSM also helps robustness while batched L2R does not. 
