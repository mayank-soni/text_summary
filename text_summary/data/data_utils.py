import random
import pandas as pd
import datasets
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5, random_seed=36):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    random.seed(random_seed)
    picks = random.sample(range(len(dataset)), num_examples)
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
      if isinstance(typ, datasets.ClassLabel):
        df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

def convert_to_df(dataset):
    return pd.DataFrame(dataset)
