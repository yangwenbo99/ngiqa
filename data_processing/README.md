Manually get `filelist.tsv` for  CSIQ, koniq10k and kadid10k,

NOTE that `gen_filelist_clive.py` is unused. 

Use scripts to get filelist for  BID and CLIVE

```bash
./gen_filelist_bid.py -v /data_partition/yang/new_db_training/BID/
./gen_filelist_clive.py -v /data_partition/yang/new_db_training/ChallengeDB_release/
```

Then, split CSIQ, kadid10k, koniq, CLIVE and BID with script

```bash
./gen_dataset_fr_spl.py -v -n 24 \
    /data_partition/yang/new_db_training/CSIQ/full_list/file_list.tsv \
    /data_partition/yang/new_db_training/CSIQ/ 
    # lower, better
./gen_dataset_fr_spl.py -v -n 65 \
    /data_partition/yang/new_db_training/kadid10k/full_list/file_list.tsv \
    /data_partition/yang/new_db_training/kadid10k/
    # higher, better

./gen_dataset_spl.py  -v \
    /data_partition/yang/new_db_training/koniq-10k/full_list/file_list.tsv \
    /data_partition/yang/new_db_training/koniq-10k/
./gen_dataset_spl.py  -v \
    /data_partition/yang/new_db_training/ChallengeDB_release/full_list/file_list.tsv \
    /data_partition/yang/new_db_training/ChallengeDB_release
./gen_dataset_spl.py  -v \
    /data_partition/yang/new_db_training/BID/full_list/file_list.tsv \
    /data_partition/yang/new_db_training/BID/
```


Directly gen sets for LIVE

```bash
./gen_filelist_live.py \
    /data_partition/yang/new_db_training/databaserelease2 \
    /data_partition/yang/new_db_training//databaserelease2/filelist.csv 

./gen_dataset_live.py -n \
    /data_partition/yang/new_db_training/databaserelease2/filelist.csv \
    /data_partition/yang/new_db_training/databaserelease2
```
