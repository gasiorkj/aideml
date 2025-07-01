import aide

if __name__ == "__main__":
    exp = aide.Experiment(
        data_dir="/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/csvs/GiveMeSomeCredit.csv",
        goal="Build a classification model and evaluate whether someone will be in financial distress in the next two years",
        
    )

    best_solution = exp.run(steps=10)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")

    with open("/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/examples/givemesomecredit/best_solution.py", "w") as f:
        f.write(best_solution.code)