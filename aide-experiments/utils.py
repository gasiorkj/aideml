import csv

def arff_to_csv(arff_filename, csv_filename):
    csv_filename = "aide-experiments/classification/csvs/"
    with open(arff_filename, 'r') as arff_file:
        lines = arff_file.readlines()

    data_started = False
    data = []
    header = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue  
        if not data_started:
            if line.lower().startswith('@attribute'):
                parts = line.split()
                attr_name = parts[1].strip("'\"")
                header.append(attr_name)
            elif line.lower().startswith('@data'):
                data_started = True
        else:
            if line:
                data.append([x.strip() for x in line.split(',')])

    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(data)



arff_to_csv('aide-experiments/classification/datasets/employee', '/Users/jogasior-kavishe/Desktop/aideml/aide-experiments/classification/csvs/employee.csv')