import pandas as pd
from pandas.api.types import CategoricalDtype


def create_timeseries_features(df: pd.DataFrame):
    """
    From DataTime column of the original DF create features for TIme Series

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with additional time series features.

    Raises:
        ValueError: If input DataFrame does not have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe index should be a datetime index")
    df = df.copy()

    features = {
        'hour': df.index.hour,
        'dayofweek': df.index.dayofweek,
        'quarter': df.index.quarter,
        'month': df.index.month,
        'year': df.index.year,
        'dayofyear': df.index.dayofyear,
        'dayofmonth': df.index.day,
        'weekofyear': df.index.isocalendar().week
    }

    for feature_name, feature_values in features.items():
        df[feature_name] = feature_values

    return df


def create_cat_features(df: pd.DataFrame):
    """
    From DataTime column of the original DF create categorical features for Time Series, adding weekday and season

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with additional categorical features.

    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe index should be a datetime index")
    df = df.copy()

    cat_type = CategoricalDtype(categories=['Monday', 'Tuesday',
                                            'Wednesday',
                                            'Thursday', 'Friday',
                                            'Saturday', 'Sunday'],
                                ordered=True)
    df['weekday'] = df.index.day_name().astype(cat_type)
    df['date_offset'] = (df.date.dt.month*100 + df.date.dt.day - 320) % 1300
    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300],
                          labels=['Spring', 'Summer', 'Autumn', 'Winter']
                          )
    df = df[['weekday', 'season']]

    return df
