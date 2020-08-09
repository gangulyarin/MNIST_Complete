import pandas as pd
from sklearn import tree, metrics
import joblib
import os
import config
import argparse
import model_dispatcher

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df.fold == fold].reset_index(drop=True)
    df_test = df[df.fold != fold].reset_index(drop=True)

    X_train = df_train.drop("label", axis="columns").values
    y_train = df_train["label"].values

    X_test = df_test.drop("label", axis="columns").values
    y_test = df_test["label"].values

    #model = tree.DecisionTreeClassifier(max_depth=3)
    model = model_dispatcher.models[model]
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    score = metrics.accuracy_score(y_test, preds)

    print(f"Fold={fold} accuracy={score}")

    #joblib.dump(model, f"models/model_{fold}.bin")
    joblib.dump(model, os.path.join(config.MODEL_OUTPUT,f"model_{fold}.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    run(fold=args.fold, model=args.model)