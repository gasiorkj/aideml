import aide

if __name__ == "__main__":
    exp = aide.Experiment(
        data_dir="/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/csvs/employee.csv",
        goal="Build a classification model for whether an employee will leave or not using the TabM model",
        
    )

    best_solution = exp.run(steps=10)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")

    with open("/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/examples/employee/best_solution.py", "w") as f:
        f.write(best_solution.code)