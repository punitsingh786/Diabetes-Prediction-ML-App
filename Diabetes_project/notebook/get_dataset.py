import pandas as pd

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

df = pd.read_csv(url)

df.to_csv("../data/diabetes.csv", index=False)

print("Dataset ready:", df.shape)