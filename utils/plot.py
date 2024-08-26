import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

colour_pal = sns.color_palette('flare')


def plot_train_test_split(df: pd.DataFrame, split_date: str) -> None:
    """
    Plot the train and test split of the dataset

    Args:
        df (pd.DataFrame): The dataframe to plot
         split_date (str): The date to split the dataset on

    Returns:
        None
    """

    df_train = df.loc[df.index < split_date]
    df_test = df.loc[df.index >= split_date]

    fig, ax = plt.subplots(figsize=(15, 5))

    df_train.plot(style='.', y='PJME_MW', ax=ax, label='Training Set',
                  color=colour_pal[0], title='Data Train/Test Split')
    df_test.plot(style='.', y='PJME_MW', ax=ax,
                 label='Test Set', color=colour_pal[3])
    ax.axvline(split_date, color='grey', ls='--')
    ax.legend(['Training Set', 'Test Set'])
    ax.set_ylabel('Energy (MW)')
    plt.show()


def plot_time_range(df: pd.DataFrame, x: str) -> None:
    """
    Plot the hour, week, month or day of year of data 

    Args:
        df (pd.DataFrame): The dataframe to plot
        x (str): The time range to plot

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxenplot(data=df.dropna(),
                  x=x,
                  y='PJME_MW',
                  legend=False,
                  hue=x,
                  ax=ax)
    ax.set_title(f'Power Use MW by {x.capitalize()}')
    ax.set_xlabel(f'Day of {x.capitalize()}')
    ax.set_ylabel('Energy (MW)')
    plt.show()


def plot_mean_monthly(df: pd.DataFrame) -> None:
    """
    Plot the mean power use by month per year

    Args:
        df (pd.DataFrame): The dataframe to plot

    Returns:
        None
    """
    df_copy = df.copy()
    year_group = df_copy.groupby(['year', 'month']).mean().reset_index()

    years = df_copy['year'].unique()

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, y in enumerate(years):
        df_copy = year_group[year_group['year'] == y]
        plt.plot(df_copy['month'], df_copy['PJME_MW'], linewidth=1)
    ax.set_title('Mean Monthly Energy Consumption by Year')
    ax.set_xlabel('Month')
    ax.set_ylabel('Energy (MW)')
    plt.show()


def plot_feature_importance(model) -> None:
    """
    Plot the feature importance of a model

    Args:
       model (pd.DataFrame): The dataframe of feature model

    Returns:
        None
    """

    importances = pd.DataFrame(data=model.feature_importances_,
                               index=model.feature_names_in_, columns=['importance'])
    sorted_importances = importances.sort_values('importance')
    sorted_importances.plot(
        kind='barh', figsize=(10, 5), color=colour_pal[0], title='Feature Importance')
    plt.show()
