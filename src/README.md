### Activate environment
```
conda activate akruti
```

### Prepare data
```
python preprocess.py -language=turkish --rewrite
```

### Run

#### Train
```
python main.py --train -model_id=1 -language=turkish -epochs=5
```

#### Test
```
python main.py --test -model_id=1 -language=turkish
```

### Evaluation
```
python evalm.py --golden=../data/files/turkish-task3-test --guesses=../results/turkish-1-guesses
```
