import aide

if __name__ == "__main__":
    exp = aide.Experiment(
        data_dir="/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/regression/csvs/physiochemical_protein.csv",
        goal="Build a regression model that will predict the size of the protein residual size",
        
    )

    best_solution = exp.run(steps=10)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")

    with open("/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/regression/examples/physiochemical_protein/best_solution.py", "w") as f:
        f.write(best_solution.code)