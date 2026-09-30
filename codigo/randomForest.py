import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

enlaces = [
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/audi.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/bmw.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/cclass.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/focus.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/ford.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/hyundi.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/merc.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/skoda.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/toyota.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/vauxhall.csv",
    "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/vw.csv"
]

dataframes = []
for url in enlaces:
    df_temp = pd.read_csv(url)
    if 'tax(£)' in df_temp.columns:
        df_temp.rename(columns={'tax(£)': 'tax'}, inplace=True)
    marca = url.split('/')[-1].replace('.csv', '').upper()
    df_temp['brand'] = marca 
    dataframes.append(df_temp)

dataset_crudo = pd.concat(dataframes, ignore_index=True)

dataset_limpio = dataset_crudo.dropna(axis=0)

df_numerico = pd.get_dummies(dataset_limpio, drop_first=True)

y = df_numerico['price']
X = df_numerico.drop('price', axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
