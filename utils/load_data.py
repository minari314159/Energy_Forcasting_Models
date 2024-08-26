import pandas as pd 

#Load and pred data

def load_PJME():
    df_East = pd.read_csv('DataSet/Regions/PJME_hourly.csv').set_index('Datetime').query('PJME_MW > 19_000').rename(
            {'PJME_MW': 'Energy Use (MW)'}, axis=1)
    df_East.index = pd.to_datetime(df_East.index)

    return df_East

def load_PJMW():
    df_West = pd.read_csv('DataSet/Regions/PJMW_hourly.csv').set_index(
        'Datetime').query('PJMW_MW > 2_300').rename({'PJMW_MW': 'Energy Use (MW)'}, axis=1)
    df_West.index = pd.to_datetime(df_West.index)

    return df_West