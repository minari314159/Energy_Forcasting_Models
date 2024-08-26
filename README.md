Time Series Forcasting of Energy Consumption Data

A time series taking historic data and uses machine learning models to predict the consumption in the future. Energy consumption tends to be considered a seasonal pattern.

Here the XGBoost machine learning model is used known to be a great out of the box model for tabular and timeseries data

About the Dataset
PJM Hourly Energy Consumption Data

PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. It is part of the Eastern Interconnection grid operating an electric transmission system serving all or parts of Delaware, Illinois, Indiana, Kentucky, Maryland, Michigan, New Jersey, North Carolina, Ohio, Pennsylvania, Tennessee, Virginia, West Virginia, and the District of Columbia.

The hourly power consumption data comes from PJM's website and are in megawatts (MW).

The regions have changed over the years so data may only appear for certain dates per region.

1998-2002 : LOAD Region (includes all regions no separations)
2002-2018 : PJME and PJMW (region is split between west and east- see map)
2004-2018 : Company State specific regions develop (AEP,COMED,DAYTON,DEOK,DOM,DUQ,EKPC - see map 2)

**This analysis and predictions with focus on PJME and PJMW as they are the most comprehensive data set, although the general regions are larger thus the results will be more generalized**
