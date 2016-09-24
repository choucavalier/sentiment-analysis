import csv
import pandas as pd

def load_data(datasets):

    frames = []

    for dataset_path in datasets:
        with open(dataset_path, newline='', encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter=',')
            rows = [x[:1] + [','.join(x[1:-2])] + x[-2:] for x in reader]
            dataset = pd.DataFrame(rows)
            frames.append(dataset)

    # concatenate data frames into a single one
    data = pd.concat(frames)

    return data

def main():

    datasets = [
        'data/train/train_airlines.csv',
        'data/train/train_apple.csv',
        'data/train/train_products.csv',
        'data/test/test_airlines.csv',
        'data/test/test_apple.csv',
        'data/test/test_products.csv',
    ]

    data = load_data(datasets)

    print(data.shape)

    print(data.as_matrix())

if __name__ == '__main__':
    main()
