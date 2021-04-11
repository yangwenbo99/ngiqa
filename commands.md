## Generate datasets 

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


## Basics

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



## Initial testing 

Trained with LIVE (portion), test with CSIQ

```bash
./Main.py --ckpt_path checkpoint-live-corr-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001 -e --test_correlation
## SRCC 0.876355, PLCC: 0.874138
./Main.py --ckpt_path checkpoint-live-corr-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/IQA_database/CSIQ/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001 -e --test_correlation
## SRCC 0.709654, PLCC: 0.739426


./Main.py --ckpt_path checkpoint-live-corr-n-03 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/IQA_database/CSIQ/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001 -e --test_correlation

./Main.py --ckpt_path checkpoint-live-corr-n-04-empty -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/IQA_database/CSIQ/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001 -e --test_correlation -f 
./Main.py --ckpt_path ../checkpoint-empty -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/full_list/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001 -e --test_correlation -f 
```



For CLIVE: 

```bash
./gen_filelist_clive.py /data_partition/yang/new_exp_dbs/ChallengeDB_release
./gen_dataset_spl.py  /data_partition/yang/new_exp_dbs/ChallengeDB_release/full_list/file_list.tsv  /data_partition/yang/new_exp_dbs/ChallengeDB_release

./Main.py --ckpt_path checkpoint-clive-corr-n-01 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 -f --lr 0.0001

## Goes 

./Main.py --ckpt_path checkpoint-clive-corr-n-01 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 
./Main.py --ckpt_path checkpoint-clive-corr-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 -f 

./Main.py --model VE2EUIQA+2+16 --ckpt_path checkpoint-clive-corr-n-03 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --model VE2EUIQA+2+16 --ckpt_path checkpoint-clive-corr-n-03 -n --batch_size 64 --image_size 400 --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 
## 0.65, 0.72 (train 0.9)

./Main.py --model VE2EUIQA+2+8 --ckpt_path checkpoint-clive-corr-n-04 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 
## 0.65 0.72

./Main.py --model VE2EUIQA+2+8 --ckpt_path checkpoint-clive-corr-n-04a -n --batch_size 180 --image_size 500 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --model VE2EUIQA --ckpt_path checkpoint-clive-corr-n-04b -n --batch_size 180 --image_size 500 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --model VE2EUIQA+3+4 --ckpt_path checkpoint-clive-corr-n-05 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --model VE2EUIQA+1+24 --ckpt_path checkpoint-clive-corr-n-05a -n --batch_size 128 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --eval --train_correlation --test_batch_size=64
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --eval --test_batch_size=64
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/koniq-10k/full_list/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --eval --test_batch_size=64
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/BID/full_list/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --eval --test_batch_size=64
## Training batch 0.99, whole 0.4x, testing less than 0.4
## Because BATCH SIZE IS 1 and IT HAS NO BATCH NORM
## testing SRCC 0.947086, PLCC: 0.954963
## On KonIQ: SRCC 0.709912, PLCC: 0.733025
## On BID: SRCC 0.560673, PLCC: 0.541836

############################### Adversarial

## Test clean trained
## On test
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --eval --test_batch_size=64 --adversarial FGSM --adversarial_radius 0.035 --eval_adversarial
## Total: 26796.0
## Correct: 24613 (0.870125198)
## Inverted: 12259


## On training
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/ChallengeDB_release/ptraining/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --eval --test_batch_size=64 --adversarial FGSM --adversarial_radius 0.035 --eval_adversarial
#
Total: 431985.0
Correct: 399419
Inverted: 203536


## Train 
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-adv-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --test_batch_size=64 --adversarial FGSM --adversarial_radius 0.035

## On testing 
./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-adv-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --test_batch_size=64 --adversarial FGSM --adversarial_radius 0.035 --eval --eval_adversarial 
## Total: 26796.0
## Correct: 24762
## Inverted: 1650

./Main.py --ckpt_path checkpoint-clive-corr-n-basecnn-adv-01 --model BaseCNN -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_db_training/ChallengeDB_release/ptraining// --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 30 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001 --test_batch_size=64 --adversarial FGSM --adversarial_radius 0.035 --eval --eval_adversarial 



############### Live adv
## Clean
./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-clean-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --lr 1e-3 -f

## Interestingly, adversarial training on VEN... doe snot give good reesult

## Adversarial
./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-adv-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.035  --lr 1e-3 -f
SRCC 0.943443, PLCC: 0.948407
Epoch 94 Testing: 2.2654
Adam learning rate: 0.001000
./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-adv-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 200 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.035  --lr 1e-3 --eval --eval_adversarial 
Total: 13041.0
Correct: 11706
Inverted: 3059


## Lwt it be larger
./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-adv-03 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 -f
./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-adv-03 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 --eval --eval_adversarial 
## SRCC 0.876761, PLCC: 0.828767
## inverted: 5030

./Main.py --model E2EUIQA --ckpt_path checkpoint-live-corr-n-adv-0x -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 


## Test the clean one 
./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-clean-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --lr 1e-3  --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 --eval --eval_adversarial
## SRCC 0.961068, PLCC: 0.963520
## Total: 13041.0
## Correct: 11764
## Inverted: 10111

./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-clean-02 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --lr 1e-3  --adversarial FGSM --adversarial_radius 0.035  --lr 1e-3 --eval --eval_adversarial

Total: 13041.0
Correct: 11764
Inverted: 8661

./Main.py --model BaseCNN --ckpt_path checkpoint-live-corr-n-adv-03 -n --batch_size 128 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 --eval --eval_adversarial 
## 0.45
Total: 13041.0
Correct: 10953
Inverted: 5030


### 30 epochs shal lbe sufficient
/Main.py --model E2EUIQA --ckpt_path checkpoint-live-corr-n-adv-04 -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.035  --lr 1e-2 -f 


./Main.py --model BaseCNN --backbone resnet34 --ckpt_path checkpoint-live-corr-n-adv-05 -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 

Total: 13041.0
Correct: 11252
Inverted: 3860

time ./Main.py --model BaseCNN --backbone resnet34 --ckpt_path checkpoint-live-pl2r-n-clear-05-retrain -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5  --lr 1e-3 -f
## SRCC 0.968508, PLCC: 0.970246
## On CSIQ: 0.73, 0.80

./Main.py --model BaseCNN --backbone resnet34 --ckpt_path checkpoint-live-mse-n-adv-05 -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn mse --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 95 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-3 

./Main.py --model BaseCNN --backbone resnet34 --ckpt_path checkpoint-live-pl2r-n-clear-05 -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn PL2R --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 180 --epochs_per_eval 5 --epochs_per_save 5  --lr 1e-4 
./Main.py --model BaseCNN --backbone resnet34 --ckpt_path checkpoint-live-pl2r-n-adv-05 -n --batch_size 64 --image_size 256 --crop_test --trainset /data_partition/yang/new_exp_dbs/databaserelease2/ptraining/ --testset /data_partition/yang/new_exp_dbs/databaserelease2/ptesting/ --lossfn PL2R --eval_lossfn MAE --test_correlation --decay_interval 60 --max_epochs 180 --epochs_per_eval 5 --epochs_per_save 5 --adversarial FGSM --adversarial_radius 0.087  --lr 1e-4




./Main.py --model MS --ckpt_path checkpoint-clive-corr-n-06 -n --batch_size 256 --image_size 128 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 40 --max_epochs 200 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 


./Main.py --model VE2EUIQA --ckpt_path checkpoint-clive-corr-n-07tmp -n --batch_size 50 --image_size 300 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 40 --max_epochs 200 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --model E2EUIQA --ckpt_path checkpoint-clive-corr-n-07 -n --batch_size 50 --image_size 300 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 40 --max_epochs 200 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 
## Loss: 0.68
```


Change loss

```bash
./Main.py --model VE2EUIQA+2+8 --ckpt_path checkpoint-clive-scorr-n-01 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn SCORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 
./Main.py --model VE2EUIQA+2+8 --ckpt_path checkpoint-clive-scorr-n-01 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn SCORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 --eval --train_correlation
## Test: 0.68/0.73
## Train: 0.807/0.823

./Main.py --model BaseCNN --ckpt_path checkpoint-clive-scorr-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn SCORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 
./Main.py --model BaseCNN --ckpt_path checkpoint-clive-scorr-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn SCORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 --eval --train_correlation
## Interestingly, negative testing

./Main.py --model BaseCNN --ckpt_path checkpoint-clive-mcorr-x-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn MCORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.0001

./Main.py --model BaseCNN --ckpt_path checkpoint-clive-maecorr-n-01 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn MAECORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001

./Main.py --model VE2EUIQA --ckpt_path checkpoint-clive-maecorr-n-02 -n --batch_size 64 --image_size 400 --crop_test --trainset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptraining/ --testset /data_partition/yang/new_exp_dbs/ChallengeDB_release/ptesting/ --lossfn MAECORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001
```


Combined: 

```bash
./main_combined.py -c config_files/combined/basecnn_01.conf  -v 

```



Repeat dataset: 

```matlab
% Ratio of total number
num_selectionkadid = uint32(5 * num_selection);
num_selectionlive = uint32(2.5 * num_selection);
num_selectioncsiq = uint32(3.5 * num_selection);
num_selectiontid = uint32(3 * num_selection);
num_selectionbid = uint32(2 * num_selection);
num_selectionclive = uint32(4 * num_selection);
num_selectionkoniq = uint32(9 * num_selection);

```
DS Size: 
- LIVE: 779
- CSIQ: 866
- KADID: 10125
- CLIVE: 1162
- KonIQ: 10073
- BID: 585    

Total Size: 
```
	DB Size	Magnitier	Repeat	
LIVE	779	2.5	0.003209242618742	6.41848523748395
CSIQ	866	3.5	0.004041570438799	8.08314087759815
KADID	10125	5	0.000493827160494	0.987654320987654
CLIVE	1162	4	0.003442340791738	6.88468158347676
KonIQ	10073	9	0.000893477613422	1.78695522684404
BID	585	2	0.003418803418803	6.83760683760684

```


- LIVE 6
- CSIQ 8
- KDID 1
- CLIVE 7
- KonIQ 2
- BID 7

### ResNet34

```bash
time ./main_combined.py -c config_files/combined/basecnn_resnet34_rd_01.conf  -v  --eval --train_correlation
## live   Train: SRCC 0.985950, PLCC: 0.987201
## csiq   Train: SRCC 0.985780, PLCC: 0.991069
## kadid  Train: SRCC 0.975605, PLCC: 0.978068
## clive  Train: SRCC 0.987766, PLCC: 0.990316
## koniq  Train: SRCC 0.929869, PLCC: 0.943082
## bid    Train: SRCC 0.935415, PLCC: 0.942202

time ./main_combined.py -c config_files/combined/basecnn_resnet34_rd_01.conf  -v  --eval --test_correlation
live   Test: SRCC 0.942713, PLCC: 0.945508
csiq   Test: SRCC 0.883825, PLCC: 0.915668
kadid  Test: SRCC 0.902582, PLCC: 0.903919
clive  Test: SRCC 0.828987, PLCC: 0.855537
koniq  Test: SRCC 0.845036, PLCC: 0.860174
bid    Test: SRCC 0.723336, PLCC: 0.726696

### Interestingly, batch normalisation affects the performance on diffrerent DBs


```



### discoveries

Adding dataset does not help a lot......


<pre>live   Test: SRCC 0.944942, PLCC: 0.948001                                                               
csiq   Test: SRCC 0.896312, PLCC: 0.921174                                                               kadid  Test: SRCC 0.878834, PLCC: 0.876640                                                               
clive  Test: SRCC 0.828823, PLCC: 0.864057                                                               
koniq  Test: SRCC 0.839624, PLCC: 0.847759                                                               
bid    Test: SRCC 0.698317, PLCC: 0.694736                                                               
</pre>

Even if kadid is inverted




### What if only Koniq? 

```bash
./Main.py --model BaseCNN --ckpt_path checkpoint-koniq-corr-n-01 -n --batch_size 64 --test_batch_size 128 --image_size 400 --crop_test --trainset /data_partition/yang/new_db_training/koniq-10k/ptraining/ --testset /data_partition/yang/new_db_training/koniq-10k/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 80 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.001 -f 

./Main.py --model BaseCNN --ckpt_path checkpoint-koniq-corr-n-01 -n --batch_size 64 --test_batch_size 128 --image_size 400 --crop_test --trainset /data_partition/yang/new_db_training/koniq-10k/ptraining/ --testset /data_partition/yang/new_db_training/koniq-10k/ptesting/ --lossfn CORR --eval_lossfn MAE --test_correlation --decay_interval 20 --max_epochs 400 --epochs_per_eval 5 --epochs_per_save 5 --lr 0.01 -f 
```



## Cross-database setup


```bash
time ./main_combined.py -c config_files/combined/basecnn_cross_01.conf  -v 
time ./main_combined.py -c config_files/combined/basecnn_cross_02.conf  -v 

time ./main_combined.py -c config_files/combined/basecnn_cross_rank_01.conf  -v 
```



## Testing adversarial

```bash
time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf 
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial_radius 1.2e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial RFGSM --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial RFGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial RFGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial RFGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial RFGSM --adversarial_radius 1.2e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial SLINF --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32

time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf 

    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial_radius 1.2e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial RFGSM --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial RFGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial RFGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial RFGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial RFGSM --adversarial_radius 1.2e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32


time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf 

    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial FGSM --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial FGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial FGSM --adversarial_radius 1.2e-1 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial RFGSM --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial RFGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial RFGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial RFGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial RFGSM --adversarial_radius 1.2e-1 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32

time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf 
    time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf  --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf  --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf  --adversarial FGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf  --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf  --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/rfgsm/r1.conf  --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf 
    time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf  --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf  --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf  --adversarial FGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf  --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf  --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
    
    time ./main_combined.py -v -c config_files/adv/rfgsm/r2.conf  --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf 
    time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf  --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf  --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf  --adversarial FGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf  --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf  --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/rfgsm/r3.conf  --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --adversarial FGSM --adversarial_radius 1.5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/pl2rrfgsm/r2.conf
    time ./main_combined.py -v -c config_files/adv/pl2rrfgsm/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rrfgsm/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/pl2rrfgsm/r2.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rrfgsm/r2.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    # What about on the training set?  Well, not exactly 'training set', when training, we use pairs, but when testing, we use single images
    # 107011 / 170349
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1 --test_live /data_partition/yang/new_db_training/databaserelease2/ptraining/


time ./main_combined.py -v -c config_files/adv/clean.conf
    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1 --test_live /data_partition/yang/new_db_training/databaserelease2/ptraining/
    # 83855 / 177680

    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial FGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial RFGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial RFGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial RFGSM --adversarial_radius 1e-1 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/clean.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/pl2rclean.conf
    time ./main_combined.py -v -c config_files/adv/pl2rclean.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rclean.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/pl2rclean.conf --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rclean.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r1.conf
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r1.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r1.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1


time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf

time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r3.conf

# R2e3
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2-e3.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2-e3.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1



time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r2.conf
    time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1


time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r3.conf
    time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r3.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r3.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1


time ./main_combined.py -v -c config_files/adv/lossgradl1reg/corr-r2.conf
    time ./main_combined.py -v -c config_files/adv/lossgradl1reg/corr-r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/lossgradl1reg/corr-r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1


############## Pairwise


time ./main_combined.py -v -c config_files/adv/pl2rrandfgsm/r2.conf
    time ./main_combined.py -v -c config_files/adv/pl2rrandfgsm/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pl2rrandfgsm/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2.conf
    time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2-long-training.conf
    time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2-long-training.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2-long-training.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2-very-long-training.conf
    time ./main_combined.py -v -c config_files/adv/bl2rfgsm/r2-very-long-training.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1


############## Pairwise, hard fidelity

time ./main_combined.py -v -c config_files/adv/other_losses/pl2r_hf/r2.conf
    time ./main_combined.py -v -c config_files/adv/other_losses/pl2r_hf/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/other_losses/pl2r_hf/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

############### SRCC
time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/r2.conf 
    time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1 
    time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1 

time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/x1-r2.conf 
    time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/x1-r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1 
    time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/x1-r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1 

time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/x2-r2.conf 
    time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/x2-r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1 
    time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/x2-r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1 


time ./main_combined.py -v -c config_files/adv/batchedsrcc/fgsm/r3.conf 


time ./main_combined.py -v -c config_files/adv/clean.conf --eval --test_gradient_length --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --eval --test_gradient_length --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --eval --test_gradient_length --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --eval --test_gradient_length --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/pl2rclean.conf --eval --test_gradient_length --test_batch_size 1 
time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --eval --test_gradient_length --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r3.conf --eval --test_gradient_length --test_batch_size 1
time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r3.conf --eval --test_gradient_length --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/clean.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn CORR
time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn CORR
time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn CORR
time ./main_combined.py -v -c config_files/adv/fgsm/r3.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn CORR


time ./main_combined.py -v -c config_files/adv/pl2rclean.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn PL2R
time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r2.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn PL2R
time ./main_combined.py -v -c config_files/adv/pl2rfgsm/r3.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn PL2R

time ./main_combined.py -v -c config_files/adv/lossgradl1reg/r3.conf --eval --test_loss_gradient_length --test_batch_size 56 --eval_lossfn PL2R



```

default loss

```bash

time ./main_combined.py -v -c config_files/adv/other_losses/default_loss/r2.conf
```

cos loss

```bash

time ./main_combined.py -v -c config_files/adv/other_losses/cos/corr-r2.conf
    time ./main_combined.py -v -c config_files/adv/other_losses/cos/corr-r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/other_losses/cos/corr-r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/other_losses/cos/corr-r3.conf
    time ./main_combined.py -v -c config_files/adv/other_losses/cos/corr-r3.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/other_losses/cos/corr-r3.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
```

Reduced batch size 

```bash
time ./main_combined.py -v -c config_files/adv/fgsm/r2-b16.conf
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b16.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b16.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/fgsm/r2-b8.conf
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b8.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b8.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    # Test first 3 epoches
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b8.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1 --ckpt BaseCNN-00002.pt

time ./main_combined.py -v -c config_files/adv/fgsm/r2-b4.conf
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b4.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b4.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
```

Slow start, i.e. the first few epoches to train FC layers are not larger

```bash
time ./main_combined.py -v -c config_files/adv/fgsm/r2-slow-start.conf
```

PGD
```bash
time ./main_combined.py -v -c config_files/adv/pgd/r2.conf
    time ./main_combined.py -v -c config_files/adv/pgd/r2.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pgd/r2.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    time ./main_combined.py -v -c config_files/adv/pgd/r2.conf --adversarial FGSM --adversarial_radius 1e-2 --eval --eval_adversarial --test_batch_size 32

    time ./main_combined.py -v -c config_files/adv/pgd/r2.conf   --adversarial SLINF --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/pgd/r2.conf   --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

    time ./main_combined.py -v -c config_files/adv/pgd/r2.conf   --adversarial PGD --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 32
    
```


```bash
time ./main_combined.py -v -c config_files/adv/fgsm/r2-b16.conf
```


## Plot weight report

```bash
./compare_model.py -v -c config_files/adv/fgsm/r2-b16.conf -j tmp_json.json --md tmp_markdown


./compare_model.py -v -c config_files/adv/bl2rfgsm/r2.conf --md weight_reports/bl2rfgsm-r2 --cm
./compare_model.py -v -c config_files/adv/bl2rfgsm/r2-long-training.conf --md weight_reports/bl2rfgsm-r2-long-traininng --cm --step 8 
./compare_model.py -v -c config_files/adv/pl2rfgsm/r2.conf --md weight_reports/pl2rfgsm-r2 --cm
./compare_model.py -v -c config_files/adv/fgsm/r2.conf --md weight_reports/fgsm-r2 --cm
./compare_model.py -v -c config_files/adv/fgsm/r2-b16.conf --md weight_reports/fgsm-r2-b16     #pending

./compare_model.py -v -c config_files/adv/bl2rfgsm/r2.conf --md weight_reports/bl2rfgsm-r2 --sm
./compare_model.py -v -c config_files/adv/bl2rfgsm/r2-long-training.conf --md weight_reports/bl2rfgsm-r2-long-traininng --sm --smstep 8 
./compare_model.py -v -c config_files/adv/pl2rfgsm/r2.conf --md weight_reports/pl2rfgsm-r2 --sm
./compare_model.py -v -c config_files/adv/fgsm/r2.conf --md weight_reports/fgsm-r2 --sm

./compare_model.py -v -c config_files/adv/fgsm/r2-slow-start.conf --md weight_reports/fgsm-r2-slow-start --sm --cm

./compare_model.py -v -c config_files/adv/large_scale/resnet50-clean.conf --md weight_reports/large-scale-resnet50 --sm --batch_size 1 --test_batch_size 1
./compare_model.py -v -c config_files/adv/large_scale/r2.conf --md weight_reports/large-scale-r2 --sm 
```



## Testing other losses

```bash
time ./main_combined.py -v -c config_files/adv/losses/mse/r2.conf
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b16.conf --adversarial FGSM --adversarial_radius 2e-2 --eval --eval_adversarial --test_batch_size 1
    time ./main_combined.py -v -c config_files/adv/fgsm/r2-b16.conf --adversarial FGSM --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1

time ./main_combined.py -v -c config_files/adv/losses/mse/r2-kiniq.conf

time ./main_combined.py -v -c config_files/adv/losses/mse/r2-output-norm.conf


# Norm in norm
time ./main_combined.py -v -c config_files/adv/losses/nin/r2.conf
```



- output normalisation
- PGD
- cross-dataset

## Large scale

```bash
./main_combined.py -v -c config_files/adv/large_scale/r2.conf
./main_combined.py -v -c config_files/adv/large_scale/clean.conf
    ./main_combined.py -v -c config_files/adv/large_scale/clean.conf --eval 

./main_combined.py -v -c config_files/adv/large_scale/resnet50-clean.conf --eval --eval_adversarial  --adversarial FGSM --adversarial_radius 5e-2 
```


Larger model can get better result, but we don't have resource for training and testing such models 



## Demos


```bash
python demo.py -c config_files/adv/large_scale/clean.conf  -i demo_imgs/cat.jpg  -g demo_imgs/large-scale-clean-cat-high.tiff  -l demo_imgs/large-scale-clean-cat-low.tiff --adversarial FGSM --adversarial_radius 0.05


```
