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

INPUT_data = 'data/'
INPUT_photo = os.path.join(INPUT_data, 'photos/')

OUTPUT = 'out_put/'
os.makedirs(OUTPUT, exist_ok=True)

photo_pathes = glob(os.path.join(INPUT_photo, '*.jpg'))
train_df = pd.read_csv(os.path.join(INPUT_data, 'train.csv'))
test_df = pd.read_csv(os.path.join(INPUT_data, 'test.csv'))

material_df = pd.read_csv(os.path.join(INPUT_data, 'materials.csv'))
technique_df = pd.read_csv(os.path.join(INPUT_data, 'techniques.csv'))

##image functions
def to_img_path(object_id):
    return os.path.join(INPUT_photo, f'{object_id}.jpg')

def read_image(object_id):
    return Image.open(to_img_path(object_id))


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class art_Dataset(torch.utils.data.Dataset):
    object_path_key = 'object_path'
    label_key = 'target'

    def __init__(self, meta_data, is_train=True):
        self.meta_data = meta_data
        self.is_train = is_train

        self.train_data = meta_data.copy()
        self.train_data['object_path'] = self.train_data['object_id'].map(to_img_path)
        self.train_data = self.train_data.reset_index(drop=True)
        self.train_data = self.train_data.to_dict(orient='index')
        self.transform_train = transforms.ToTensor()
        self.transform_test = transforms.ToTensor()

    def __getitem__(self, idx):
        data = self.train_data[idx]
        obj_path, t_train = data.get(self.object_path_key), data.get(self.label_key, -1)
        x_data = Image.open(obj_path)

        if self.is_train: 
            x_data = self.transform_train(x_data)
        else:
            x_data = self.transform_test(x_data)

        return x_data, t_train

    def __len__(self):
        return len(self.meta_data)

#data argumentation
size = (224,224)
transform_train = transforms.Compose([
    # transforms.RandomCrop(size=size, padding=(4,4,4,4), padding_mode='constant'),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size),
    transforms.ToTensor()
])

transforms_test = transforms.Compose([
    transforms.RandomResizedCrop(size),
    transforms.ToTensor()
])

#make dataset
# train_data = art_Dataset(train_df)
test_data = art_Dataset(test_df, is_train=False)
test_data.transform_test = transforms_test

import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision.models import resnet34
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
from torch.optim.lr_scheduler import StepLR

#train
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: torch.utils.data.DataLoader
)-> pd.Series:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()
    loss_function = nn.MSELoss()

    metrics = defaultdict(float)
    n_iters = len(train_loader)

    for i, (x_i, y_i) in enumerate(train_loader):
        x_i = x_i.to(device)
        y_i = y_i.to(device).reshape(-1, 1).float()

        pred = model(x_i)
        loss = loss_function(pred, y_i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_i = {'loss': loss.item()}
        for k, v in metric_i.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] /= n_iters

    return pd.Series(metrics).add_prefix('train_')

def predict(model: nn.Module, loader: torch.utils.data.DataLoader) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    predicts = []
    for x_i, y_i in loader:
        with torch.no_grad():
            pred = model(x_i.to(device))

        predicts.extend(pred.data.cpu().numpy())

    predict = np.array(predicts).reshape(-1)
    return predict

def cal_loss(y_test, y_pred) -> dict:
    return {'rmse': mean_squared_error(y_test, y_pred) ** .5}

def valid(
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    y_test: np.ndarray
) -> pd.Series:

    pred = predict(model, valid_loader)
    loss = cal_loss(y_test, pred)

    valid_score = pd.Series(loss)
    return valid_score.add_prefix('valid_'), pred

    
#K fold
def run_fold(
    model: nn.Module,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    y_valid: np.ndarray,
    output_dir: str,
    n_epochs=30,
    batchsize=64) -> np.ndarray:

    os.makedirs(output_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=.2)

    train_data = art_Dataset(train_df)
    valid_data = art_Dataset(valid_df)
    train_data.transform_train = transform_train
    valid_data.transform_train = transform_train

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = batchsize,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batchsize,
        shuffle=True
    )

    loss_df = pd.DataFrame()
    best_loss = np.inf
    best_loss_key = 'valid_rmse'
    valid_best_pred = None

    for epoch in range(n_epochs):
        train_loss = train(model, optimizer, train_loader)
        valid_loss, valid_pred = valid(model, valid_loader, y_valid)

        row = pd.concat([train_loss, valid_loss])
        row['epoch'] = epoch
        row = pd.DataFrame([row])
        print(tabulate(row, headers=row.columns))
        loss_df = pd.concat([loss_df, row], ignore_index=True)

        current_loss = valid_loss[best_loss_key]

        if current_loss < best_loss:
            print(f'validation loss is improved {best_loss: .4f} -> {current_loss: .4f}')
            torch.save(
                model.state_dict(), os.path.join(output_dir, 'model_best.pth')
            )
            best_loss = current_loss
            valid_best_pred = valid_pred
        
        scheduler.step()

    loss_df.to_csv(os.path.join(output_dir, 'loss.csc'), index=False)
    return valid_best_pred

def get_output_dir(output: str, n_cv: int):
    return os.path.join(output, f'cv={n_cv}')

from sklearn.model_selection import KFold
oof = np.zeros((len(train_df),), dtype=np.float32)
fold = KFold(n_splits=5, shuffle=True, random_state=0)
cv = list(fold.split(X=train_df))
OUTPUT = 'out_files/resnet32/sub02'

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