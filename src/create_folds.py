from sklearn import model_selection
import pandas as pd

def make_folds(splits):
    df = pd.read_csv("input/train.csv")

    df["fold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.label.values

    kf = model_selection.StratifiedKFold(n_splits=splits)

    for f, (_t, _v) in enumerate(kf.split(X=df, y=y)):
        df.loc[_v, "fold"] = f

    df.to_csv("train_fold.csv", index=False)

if __name__ == "__main__":
    make_folds(5)
