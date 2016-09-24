import pickle

def load_data(datasets):

    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

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

    print(len(data.keys()))

if __name__ == '__main__':
    main()
