from aide import Experiment

exp = Experiment(
    data_dir="/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/csvs/employee.csv",
    goal="Build a classification model for whether an employee will leave or not",
    eval="Accuracy"
)

best_solution = exp.run(steps=10)


print(f"Best solution has validation metric: {best_solution.valid_metric}")
print(f"Best solution code: {best_solution.code}")
