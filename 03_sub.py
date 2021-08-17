import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import mz_logger
from sklearn.model_selection import KFold
from torchvision.models import resnet34
from n01_resnet import *


if __name__ == '__main__':
    INPUT_data = 'data/'
    INPUT_photo = os.path.join(INPUT_data, 'photos/')

    OUTPUT = 'out_put/'
    os.makedirs(OUTPUT, exist_ok=True)

    photo_pathes = glob(os.path.join(INPUT_photo, '*.jpg'))
    train_df = pd.read_csv(os.path.join(INPUT_data, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_data, 'test.csv'))

    material_df = pd.read_csv(os.path.join(INPUT_data, 'materials.csv'))
    technique_df = pd.read_csv(os.path.join(INPUT_data, 'techniques.csv'))
    oof = np.zeros((len(train_df),), dtype=np.float32)
    fold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv = list(fold.split(X=train_df))
    OUTPUT = 'out_files/resnet32/sub03'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (train_idx, valid_idx) in enumerate(cv):
        print(f'-----------------{i: d} step-------------------')
        output_csv = get_output_dir(OUTPUT, i)
        model = resnet34(pretrained=False)
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

        model.to(device)

        oof_i = run_fold(
            model,
            train_df.iloc[train_idx],
            train_df.iloc[valid_idx],
            train_df['target'].values[valid_idx],
            output_csv,
            n_epochs=1
        )

        oof[valid_idx] = oof_i
        break

    print('finish')