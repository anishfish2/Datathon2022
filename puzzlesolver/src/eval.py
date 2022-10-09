import os
import random
from pathlib import Path
from submission import Predictor

data_path = Path(__file__).parent.parent.joinpath('train')

loading = {}

for arrange in os.listdir(data_path):
    for img in os.listdir(data_path.joinpath(arrange))[:1000]:
        loading[data_path.joinpath(arrange).joinpath(img)] = arrange

model = Predictor()
keys = list(loading.keys())
random.shuffle(keys)

tests = dict()
for key in keys[:320]:
    tests[key] = loading[key]

success = 0
count = 0
total = len(tests.keys())
for img_path, answer in tests.items():
    pred = model.make_prediction(img_path)
    if pred == answer:
        success += 1
    count += 1
    print(answer, pred, 'SUCCESS' if answer == pred else 'FAIL')
    if count % 10 == 0:
        print(f'{count}/{total}')

print(success/count)