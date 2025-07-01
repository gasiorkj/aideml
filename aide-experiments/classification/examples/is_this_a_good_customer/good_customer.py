import aide
import os
#print(os.getenv("RITS_API_KEY"))

if __name__ == "__main__":
    exp = aide.Experiment(
        data_dir="/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/csvs/Is-this-a-good-customer.csv",
        goal="Build a classification model and evaluate whether someone is a bad client or not",
        
    )

    best_solution = exp.run(steps=10)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")

    with open("/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/examples/is_this_a_good_customer/best_solution.py", "w") as f:
        f.write(best_solution.code)