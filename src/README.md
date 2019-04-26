### Activate environment
```
conda activate akruti
```

### Prepare data
```
python preprocess.py rewrite
```

### Run
```
python main.py --train --test -epochs=20
```

### Evaluation
```
python evalm.py --golden=../data/files/turkish-task3-test --guesses=../results/turkish-task3-guesses
```
