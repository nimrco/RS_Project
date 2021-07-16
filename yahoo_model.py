import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader, accuracy, dump
from surprise.model_selection import train_test_split, GridSearchCV

file_path = "ydata-ymusic-user-artist-ratings-v1_0.txt"  # Yahoo data set


def train_model_with_gridsearch():
    """
    Train the optimal model after grid search with cross validation.
    Saves the model.
    """
    df = pd.read_table(file_path, names=['user', 'item', 'rating'])
    df = df[df['rating'] != 255]
    df['rating'] = df['rating'].apply(lambda x: x/20)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    train, test = train_test_split(data, test_size=.25)
    param_grid = {'n_epochs': [5, 10, 20, 30], 'n_factors': [100, 200],
                  'lr_all': [0.005, 0.002, 0.03], 'reg_all': [0.02, 0.4, 0.6]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    print("RMSE: {}".format(gs.best_score['rmse']))
    print(gs.best_params['rmse'])
    print("MAE: {}".format(gs.best_score['mae']))
    print(gs.best_params['mae'])
    svd = gs.best_estimator['rmse']
    svd.fit(train)
    preds = svd.test(test)
    rmse = accuracy.rmse(preds)
    mae = accuracy.mae(preds)
    print(rmse)
    print(mae)
    dump.dump("svd_model_grid", algo=svd, verbose=1)


def train_model():
    """
    Train SVD model on data.
    Saves the model.
    """
    df = pd.read_table(file_path, names=['user', 'item', 'rating'])
    df = df[df['rating'] != 255]
    df['rating'] = df['rating'].apply(lambda x: x / 20)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    train, test = train_test_split(data, test_size=.25)
    svd = SVD()
    svd.fit(train)
    preds = svd.test(test)
    rmse = accuracy.rmse(preds)
    mae = accuracy.mae(preds)
    print(rmse)
    print(mae)
    dump.dump("svd_model", algo=svd, verbose=1)


def load_model():
    """
    Creates Numpy arrays of user and artist latent factors (after training),
    for integration with the original model.
    """
    svd = dump.load("svd_model")
    np.save(file="user_factors", arr=svd[1].pu)
    np.save(file="item_factors", arr=svd[1].qi)


def date_exploration():
    """
    Data exploration and graph generation.
    """
    df = pd.read_table(file_path, names=['user', 'artist', 'rating'])
    df.info()
    print(df.describe())
    sns.histplot(data=df, x="rating", kde=True, bins=100)
    plt.savefig("hist_with_255.png")
    plt.clf()
    df = df[df['rating'] != 255]
    sns.histplot(data=df, x="rating", kde=True, bins=100)
    plt.savefig("hist_1.png")


def main():
    date_exploration()
    train_model()
    load_model()


if __name__ == '__main__':
    main()
