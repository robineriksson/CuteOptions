import polars as pl

# 1) load csv
# 2) calc the price of flies
# 3) normalize
# 4) chart

path_raw = '../data/btc_eod_202312.txt'
path_clean = '../data/btc_eod_202312.csv'

# df = pl.read_csv(path_raw)

def clean_data(path_raw, path_clean):
    import csv
    # should probably add something to remove [ and ] from headings
    with open(path_raw) as f:
        reader = csv.reader(f, delimiter=",")
        # Adding newline='' to avoid extra newline characters in output
        with open(path_clean, "w", newline='') as fo:
            writer = csv.writer(fo)
            for rec in reader:
                writer.writerow(map(lambda x: x.strip(), rec))


import matplotlib.pyplot as plt

def plot_series(xcol:str, ycol:str, df:pl.DataFrame):
    x = df.select(pl.col(xcol)).to_numpy().flatten()
    y = df.select(pl.col(ycol)).to_numpy().flatten()

    plt.plot(x,y)
    plt.gcf().autofmt_xdate()
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.show()

    return None

#############################
# 1) load the option data.
#############################

df = pl.read_csv(path_clean)

# date, and datetime
df = df.with_columns(pl.col('^*.DATE.*$').cast(pl.Date),
                pl.col('QUOTE_READTIME').str.strptime(pl.Datetime,
                                                      format='%Y-%m-%d %H:%M'),
                     MID=0.5*(pl.col('BID_PRICE')+pl.col('ASK_PRICE'))
                )


# get the underlying price series.
spot = (df
 .group_by(pl.col('QUOTE_DATE'))
 .first()
 .select(pl.col('QUOTE_DATE'), pl.col('UNDERLYING_PRICE'))
 .sort('QUOTE_DATE')
 )

# look at this cute spot price ....
plot_series('QUOTE_DATE','UNDERLYING_PRICE',spot)

#############################
# 2) calculate the flies.
#############################

# probabiliy distribution can be computed by looking at the price of the flies.

# on the latest day
# per expiry date
# plot the vol smile



to_fly = (df
 .filter(pl.col('QUOTE_DATE')==pl.col('QUOTE_DATE').max())
 .select(pl.col('EXPIRY_DATE'), pl.col('OPTION_RIGHT'), pl.col('MID'),
         pl.col('MARK_IV'), pl.col('STRIKE'))
 )

import datetime

smile = to_fly.filter((pl.col('EXPIRY_DATE')==datetime.date(2023,12,31)) &
              (pl.col('OPTION_RIGHT')=='put')).sort(pl.col('STRIKE'))

plot_series('STRIKE','MARK_IV',smile)

# fly = +1x strike S  -2x strike S+1 + 1x strike S+2
smile_fly = smile.with_columns(FLY=pl.col('MID') - 2*pl.col('MID').shift(1) + pl.col('MID').shift(2))

plot_series('STRIKE','FLY',smile_fly)

# black scholes price that bitch ....

import numpy as n
from scipy.stats import norm


def bs_price(S, K, r, sigma, DTE, option_right:str):
    assert option_right in ['call','put'], 'option right must be put or call'

    d_plus = (np.log(S/K) + (r+sigma**2/2)*DTE) / (sigma*np.sqrt(DTE))
    d_minus = d_plus - sigma*np.sqrt(DTE)

    if option_right == 'call':
        price = norm.cdf(d_plus)*S - norm.cdf(d_minus)*K*np.exp(-r*DTE)
    else:
        price = norm.cdf(-d_minus)*K*np.exp(-r*DTE) - norm.cdf(-d_plus)*S

    return price

bs_price(42_219, 50_500, 0, 0.914, 0.12442, 'call')
