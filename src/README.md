### Activate environment
```
conda activate akruti
```

### Prepare data
```
python preprocess.py -language=turkish --rewrite
```

### Train, Test, Evaluation
```
python main.py  --train -model_id=1 -language=turkish -epochs=5
python main.py  --test  -model_id=1 -language=turkish
python evalm.py         -model_id=1 -language=turkish
```
