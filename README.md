_Please read `BRIEF.md` for brief information about using this for BIQA_.


To generate dataset: 

```bash
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur/training -v -m 30 -r 30
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur/training_128 -t 128 -v -m 30 -r 30
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur/training_128_s -t 128 -v -m 30 -r 1
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur/training_tmp -v -m 30 -r 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur/test128 -v -t 128 -r 4 -m 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur/test256 -v -t 256 -r 4 -m 30

```

```bash
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur_08/training -v -m 8 -r 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_08/test128 -v -t 128 -r 4 -m 8
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_08/test256 -v -t 256 -r 4 -m 8

./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur_10/training -v -m 10 -r 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_10/test128 -v -t 128 -r 4 -m 10
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_10/test256 -v -t 256 -r 4 -m 10

./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur_12/training -v -m 12 -r 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_12/test128 -v -t 128 -r 4 -m 12
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_12/test256 -v -t 256 -r 4 -m 12

./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur_13/training -v -m 13 -r 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_13/test128 -v -t 128 -r 4 -m 13
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_13/test256 -v -t 256 -r 4 -m 13

./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur_14/training -v -m 14 -r 30
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_14/test128 -v -t 128 -r 4 -m 14
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur_14/test256 -v -t 256 -r 4 -m 14
```

SSIM
```bash
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur/training_ssim -v -m 30 -r 100 -t 128 -s
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur/test_ssim -v -t 128 -r 4 -m 30 -s 
```

To train: 

```bash
./Main.py --ckpt_path checkpoint-n-01 -n
./Main.py --ckpt_path checkpoint-x-01 

./Main.py --ckpt_path checkpoint-n-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --ckpt_path checkpoint-x-30-01 --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --ckpt_path checkpoint-n-12-01 -n --trainset ../datasets/waterloo_blur_12/training/ --testset ../datasets/waterloo_blur_12/test128/ 
./Main.py --ckpt_path checkpoint-n-08-01 -n --trainset ../datasets/waterloo_blur_08/training/ --testset ../datasets/waterloo_blur_08/test128/ 
```

```bash
# Adding 2 layers temporarily
./Main.py --ckpt_path checkpoint-n-p2-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
```

To test

```bash
./Main.py --ckpt_path checkpoint-n-12-01 -n --trainset ../datasets/waterloo_blur_12/training/ --testset ../datasets/waterloo_blur_12/test128/ -e -v
```


Dependency
- pytorch, torchvisiohn
- scipy
- pandas



Some observation: 


### Baseline structure

- Repeat 3 times: 
    - Conv 5x5
    - ReLU
    - MaxPool
- Conv 3x3
- ReLU
- Spatial Pyramid Pooling
- Linear
- ReLU
- Linear

Loss:  $\left ( \frac{\bar y - y}{y + 1} \right ) ^ 2$ (`((yp - y) / (y + LOSS_N)) ** 2`)

- Radius = 30 -- Loss around 0.6
- Radius = 12 -- Loss around 0.4


For the one with radius 30: 
```
    Group    0 (length   145): 2.476473
    Group    1 (length   127): 0.235753
    Group    2 (length   138): 0.007818
    Group    3 (length   123): 0.038982
    Group    4 (length   128): 0.121743
    Group    5 (length   135): 0.202042
    Group    6 (length   139): 0.273547
    Group    7 (length   135): 0.336153
    Group    8 (length   127): 0.388704
    Group    9 (length   143): 0.435586
    Group   10 (length   130): 0.474906
    Group   11 (length   133): 0.509937
    Group   12 (length   120): 0.541125
    Group   13 (length   150): 0.568749
    Group   14 (length   134): 0.592465
    Group   15 (length   139): 0.613678
    Group   16 (length   129): 0.634188
    Group   17 (length   146): 0.651596
    Group   18 (length   162): 0.666993
    Group   19 (length   114): 0.682265
    Group   20 (length   152): 0.695345
    Group   21 (length   120): 0.707845
    Group   22 (length   141): 0.719688
    Group   23 (length   138): 0.729730
    Group   24 (length   131): 0.739950
    Group   25 (length   107): 0.748887
    Group   26 (length   126): 0.756903
    Group   27 (length   113): 0.765213
    Group   28 (length   127): 0.772737
    Group   29 (length   148): 0.779514
```

With radius 12
```
    Group    0 (length   347): 1.015711
    Group    1 (length   309): 0.035568
    Group    2 (length   322): 0.043231
    Group    3 (length   349): 0.147609
    Group    4 (length   317): 0.241021
    Group    5 (length   334): 0.325958
    Group    6 (length   341): 0.393264
    Group    7 (length   329): 0.450209
    Group    8 (length   308): 0.499618
    Group    9 (length   323): 0.538781
    Group   10 (length   369): 0.573523
    Group   11 (length   352): 0.603503
```


### Adding two layers

```python
    # Temporarily adding 2 layers
    layers += [
            nn.Conv2d(width, width, kernel_size=5, stride=1, padding=1, dilation=1, bias=True),
            normc(),
            nn.Conv2d(width, width, kernel_size=5, stride=1, padding=1, dilation=1, bias=True),
            normc(),
            ]

```

Does not help

```
    Group    0 (length   145): 2.435851
    Group    1 (length   127): 0.229026
    Group    2 (length   138): 0.007432
    Group    3 (length   123): 0.040317
    Group    4 (length   128): 0.123857
    Group    5 (length   135): 0.204146
    Group    6 (length   139): 0.275713
    Group    7 (length   135): 0.338163
    Group    8 (length   127): 0.390596
    Group    9 (length   143): 0.437431
    Group   10 (length   130): 0.476671
    Group   11 (length   133): 0.511535
    Group   12 (length   120): 0.542657
    Group   13 (length   150): 0.570201
    Group   14 (length   134): 0.593790
    Group   15 (length   139): 0.614967
    Group   16 (length   129): 0.635373
    Group   17 (length   146): 0.652721
    Group   18 (length   162): 0.668083
    Group   19 (length   114): 0.683357
    Group   20 (length   152): 0.696326
    Group   21 (length   120): 0.708785
    Group   22 (length   141): 0.720647
    Group   23 (length   138): 0.730542
    Group   24 (length   131): 0.740812
    Group   25 (length   107): 0.749732
    Group   26 (length   126): 0.757700
    Group   27 (length   113): 0.765983
    Group   28 (length   127): 0.773500
    Group   29 (length   148): 0.780242
```



### Change backbone

To 'BaseCNN': Overfitting with no mercy???  Still underfit

```bash
./Main.py --model basecnn --backbone resnet18 --ckpt_path checkpoint-basecnn-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --model basecnn --backbone resnet18 --ckpt_path checkpoint-basecnn-re-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --model basecnn --backbone resnet34 --ckpt_path checkpoint-basecnn34-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --model basecnn --phase1 1 --backbone resnet34 --ckpt_path checkpoint-basecnn34-re-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
```

To a very simple model: 

```bash
./Main.py --model simple --ckpt_path checkpoint-simple-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
```

To a Multi-scale model

```bash
./Main.py --model MS --ckpt_path checkpoint-ms-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --model MS --ckpt_path checkpoint-ms-re1-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --model MS --alternative_train_loss --ckpt_path checkpoint-ms-al-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --model MS --alternative_train_loss --ckpt_path checkpoint-ms-mae-30-01 -n --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/


# Let's try to let it overfit
./Main.py --model MS --ckpt_path checkpoint-ms-of-re-30-01 -n --trainset ../datasets/waterloo_blur/training_128/ --testset ../datasets/waterloo_blur/test128/ --decay_interval 20
./Main.py --model MS --ckpt_path checkpoint-ms-of-s-30-01 -n --trainset ../datasets/waterloo_blur/training_128_s/ --testset ../datasets/waterloo_blur/test128/ --decay_interval 10 --max_epochs 500 --epochs_per_eval 30 --epochs_per_save 30 --lr=1e-3 -f 
./Main.py --model MS --ckpt_path checkpoint-ms-of-s-30-01 -n --trainset ../datasets/waterloo_blur/training_128_s/ --testset ../datasets/waterloo_blur/test128/ --decay_interval 10 --max_epochs 500 --epochs_per_eval 30 --epochs_per_save 30 --lr=1e-3 -f 
```

### With SSIM

```bash
./gen_dataset.py ../datasets/waterloo_images/training/ ../datasets/waterloo_blur/training_ssim -v -m 30 -r 100 -t 128 -s
./gen_dataset.py ../datasets/waterloo_images/test/ ../datasets/waterloo_blur/test_ssim -v -t 128 -r 4 -m 30 -s 

```

```bash
./Main.py --ckpt_path checkpoint-ssim-n-30-01 -n --trainset ../datasets/waterloo_blur/training_ssim/ --testset ../datasets/waterloo_blur/test_ssim/
./Main.py --ckpt_path checkpoint-ssim-x-30-01 --trainset /data_partition/yang/training_ssim/ --testset /data_partition/yang/test_ssim/
# test error: 0.0001

./Main.py --ckpt_path checkpoint-ssim-mae-x-30-01 --lossfn MAE --eval_lossfn MAE --trainset /data_partition/yang/training_ssim/ --testset /data_partition/yang/test_ssim/

./Main.py --model VE2EUIQA+2+48 --ckpt_path checkpoint-ssim-x-30-02 --trainset /data_partition/yang/training_ssim/ --testset /data_partition/yang/test_ssim/
# test error: 0.0000? 

./Main.py --model VE2EUIQA+2+48 --ckpt_path checkpoint-ssim-x-30-02 --trainset /data_partition/yang/training_ssim/ --testset /data_partition/yang/test_ssim/

./Main.py --model VE2EUIQA+1+4 --ckpt_path checkpoint-ssim-x-30-03 --trainset /data_partition/yang/training_ssim/ --testset /data_partition/yang/test_ssim/


./Main.py --ckpt_path checkpoint-x-30-01 --trainset ../datasets/waterloo_blur/training/ --testset ../datasets/waterloo_blur/test128/
./Main.py --ckpt_path checkpoint-n-12-01 -n --trainset ../datasets/waterloo_blur_12/training/ --testset ../datasets/waterloo_blur_12/test128/ 
./Main.py --ckpt_path checkpoint-n-08-01 -n --trainset ../datasets/waterloo_blur_08/training/ --testset ../datasets/waterloo_blur_08/test128/ 
```



## Extending

This can be directly extended as IQA for LIVE

```bash
./gen_filelist_live.py \
    /data_partition/yang/new_exp_dbs/databaserelease2 \
    /data_partition/yang/new_exp_dbs/databaserelease2/filelist.csv 

./gen_dataset_live.py \
    /data_partition/yang/new_exp_dbs/databaserelease2/filelist.csv \
    /data_partition/yang/new_exp_dbs/databaserelease2
```


```bash
./Main.py --ckpt_path checkpoint-live-n-01 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --decay_interval 80 --max_epochs 400

./Main.py --ckpt_path checkpoint-live-mse-n-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MSE --eval_lossfn MSE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 

./Main.py --ckpt_path checkpoint-live-mae-n-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MAE --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 

./Main.py --ckpt_path checkpoint-live-n-01 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --eval --test_correlation


## Correlation loss
./Main.py --ckpt_path checkpoint-live-corr-n-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 

./Main.py --ckpt_path checkpoint-live-corr-n-03 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f


# after modifying CorrelationWithMeanLoss 
./Main.py --ckpt_path checkpoint-live-corr-n-03 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001
# SRCC 0.99, PLCC: 0.99

./Main.py --ckpt_path checkpoint-live-corr-n-04 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001

# Try another partition schema
./Main.py --ckpt_path checkpoint-live-corr-n-03-2 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining_2/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting_2/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001
# SRCC 0.982644, PLCC: 0.974200

```



