### Activate environment
```
conda activate akruti
```

### Train, Test, Evaluation
```
python main.py  --train -model_id=1 -language=turkish -epochs=5
python main.py  --test  -model_id=1 -language=turkish
python evalm.py         -model_id=1 -language=turkish
```
Usage,
```
usage: main.py [-h] [--train] [--test] [--dont_save] [-model_id MODEL_ID]
               [-language LANGUAGE] [-device DEVICE] [-epochs EPOCHS]
               [-enc_h_dim ENC_H_DIM] [-dec_h_dim DEC_H_DIM]
               [-char_emb_dim CHAR_EMB_DIM] [-tag_emb_dim TAG_EMB_DIM]
               [-enc_dropout ENC_DROPOUT] [-dec_dropout DEC_DROPOUT]
               [-z_dim Z_DIM] [-batch_size BATCH_SIZE] [-kl_start KL_START]
               [-lambda_m LAMBDA_M] [-lr LR] [-kuma_msd KUMA_MSD] [-a0 A0]
               [-b0 B0]
```
