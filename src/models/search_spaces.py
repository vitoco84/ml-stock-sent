def linreg_space(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        "max_iter": trial.suggest_int("max_iter", 1000, 5000, step=500)
    }

def xgboost_space(trial):
    space = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 2900, step=250),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "early_stopping_rounds": 100,
        "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"])
    }
    if space["grow_policy"] == "lossguide":
        space["max_leaves"] = trial.suggest_int("max_leaves", 64, 512, step=64)
    return space

SEARCH_SPACES = {
    "linreg": linreg_space,
    "xgboost": xgboost_space,
}
