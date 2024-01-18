import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    y = trial.suggest_uniform('y', -10, 10)
    print(f"x{x}")
    return (x - 2) ** 2 + (y + 3) ** 2  # Example objective function to minimize

# study = optuna.create_study(direction='minimize')
study = optuna.create_study(direction='minimize',
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                            sampler=optuna.samplers.TPESampler(seed=42))

study.optimize(objective, n_trials=100)

# 获取试验信息的 DataFrame
trial_df = study.trials_dataframe()

# 打印 DataFrame

print("Best trial:")
print(study.best_trial.params)
print("Best value:", study.best_trial.value)
print("Best value:", trial_df)
