import openml
dataset = openml.datasets.get_dataset(3892, download_data=True)
print("Target:", dataset.default_target_attribute)
X, y, cat, _ = dataset.get_data(target=dataset.default_target_attribute)
print("y unique:", y.unique())
