_Please read `BRIEF.md` for brief information about using this for BIQA_.

The GitHub repository of this project is at <https://github.com/yangwenbo99/ngiqa>. 

Another part of the project (as a fork of UNIQUE) is at <https://github.com/yangwenbo99/UNIQUE>.



### Settings

Dependencies: 
- pytorch, torchvisiohn
- scipy
- pandas


### Technical details

#### Run

The project has two entry points.  Most of the experiments are conducted using `main_combined.py`. 

#### Training strategy for IQA (cross-dataset settings) 

Images are divided into test and training set; the portion is the same as UNIQUE.  Different `Dataset`s are combined into one `RepeatedDataLoader`, and smaller datasets are repeated for several times.  Only images from the same source dataset (LIVE, CSIQ, CLIVE, etc.) will be put into the same batch.  They are then trained with the selected loss function. 

#### File structures

- The base class of trainer and attackers are in `./trainer_common.py`
    - `model.py` and `model_combined.py` are subclass of trainer for different usages
- Network models: 
    - `ve2euiqa.py`: LFC with major modifications
        - `msve2euiqa.py`: another modification (multi-scale) 
    - `e2euiqa.py`: LFC with minor modifications
    - `BaseCNN.py`: Using ResNet as backbone (very tiny modification from UNIQUE) 
    - *The one with the best performance is `BaseCNN`*
    - `Trainer._get_model(...)` is used for resolve the model based on its name 
- Loss functions are stored in `losses.py`
    - `Trainer._get_loss_fn(...)` is used for resolve the loss function based on its name 


#### Loss Functions

- MAE
- MSE
- `CORR` (`CorrelationLoss`), `MCORR` (`CorrelationWithMeanLoss`), `MAECORR` (`CorrelationWithMAELoss`) 
    - Batched PLCC
- `L2RR` (`BatchedL2RLoss`) 
- `EL2R`
- `PL2R` (`PairwiseL2RLoss`) 
    - Pairwise-L2R
- `SSRCC` (`SSRCC`) 
    - Batched SRCC

##### Batched PLCC

The PLCC value between the label of each image and the predicted value is used as loss. 

![The formula](https://bit.ly/2PSY7J3)


##### Batched SRCC

Use the [soft rank function provided by google research](https://github.com/google-research/fast-soft-sort) to approximate the rank to calculate SRCC. 



##### Vanilla L2R

![The formula](https://bit.ly/3eBqBkT)

Suppose there are 56 images in a batch 

Pairwise-L2R: use the maximum likelihood estimator for each pair of images, i.e. the sum of 28 log-probabilities

Batched-L2R: use the maximum likelihood estimator for every pair of images, i.e. the sum of (56 * 55 / 2) log-probabilities


#### Attacking methods 

- FGSM
- RFGSM (repeated FGSM), repeat FGSM for 10 times, and set each step size as $\epsilon / 10$
- The search method (not for training): as defined in [the previous project](https://github.com/yangwenbo99/torch_niqe)



#### Configurations 

The project uses a config file to reduce the length of each command.  Some config files are in `./config_files/`.  For example, the following command uses FGSM to adversarially train a model with ResNet backbone. 

```bash
time ./main_combined.py -v -c config_files/adv/fgsm/r2.conf 
```

To test its adversarial performance: 

```bash
    time ./main_combined.py -v -c config_files/adv/fgsm/r1.conf --adversarial SLINF --adversarial_radius 5e-2 --eval --eval_adversarial --test_batch_size 1
```


### Datasets

They are available at <https://portland-my.sharepoint.com/:u:/g/personal/wenboyang3-c_my_cityu_edu_hk/EeCsecuIe_dEkEBSszywSu8BP1R1BzTRpsKUSxUP9MHCeQ?e=ndUuex>


### Trained weights

Some of them are available at <https://portland-my.sharepoint.com/:f:/g/personal/wenboyang3-c_my_cityu_edu_hk/EhOm7woyeqlAhKtKW01o70UBWjO2NyqnuN--APgRxwyHNw?e=GbDmI2>
