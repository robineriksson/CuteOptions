import polars as pl
import datetime
import numpy as np
from scipy.stats import norm

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
 .select(pl.col('UNDERLYING_PRICE'), pl.col('EXPIRY_DATE'), pl.col('OPTION_RIGHT'), pl.col('MID'),
         pl.col('MARK_IV'), pl.col('STRIKE'), pl.col('DTE'))
 )



smile = to_fly.filter((pl.col('EXPIRY_DATE')==datetime.date(2024,2,23)) &
              (pl.col('OPTION_RIGHT')=='put')).sort(pl.col('STRIKE'))

plot_series('STRIKE','MARK_IV',smile)

# fly = +1x strike S  -2x strike S+1 + 1x strike S+2
smile_fly = smile.with_columns(FLY=pl.col('MID') - 2*pl.col('MID').shift(1) + pl.col('MID').shift(2))

plot_series('STRIKE','FLY',smile_fly)

# black scholes price that bitch ....

import numpy as np
from scipy.stats import norm


def bs_price(F, K, sigma, DTE, option_right:str):
    assert option_right in ['call','put'], 'option right must be put or call'

    d1 = (np.log(F/K) + (sigma**2/2)*DTE) / (sigma*np.sqrt(DTE))
    d2 = d1 - sigma*np.sqrt(DTE)

    if option_right == 'call':
        price = norm.cdf(d1) - K/F*norm.cdf(d2)
    else:
        price = K/F*norm.cdf(-d2) - norm.cdf(-d1)

    return price

def bs_price_help(x):
    return bs_price(F=x['UNDERLYING_PRICE'],K=x['STRIKE'],
                    sigma=x['MARK_IV'],DTE=x['DTE'],
                    option_right=x['OPTION_RIGHT'])

# x = np.arange(1, 100, 1)
# y = bs_price(x, 50, 0.1, 90/365, 'call')
# plt.plot(x,y)
# plt.show()



expiry_date = datetime.date(2024,3,29)
end_date = datetime.date(2023,12,31)
fair = (df.filter((pl.col('EXPIRY_DATE')==expiry_date) &
                  (pl.col('QUOTE_DATE')==end_date))
        .with_columns(pl.struct(pl.col('UNDERLYING_PRICE'),
                                pl.col('STRIKE'),
                                #pl.lit(40).alias('MARK_IV')/10_000,
                                pl.col('MARK_IV')/100,
                                pl.col('DTE')/365,
                                pl.col('OPTION_RIGHT'))
                      .map_elements(bs_price_help)
                      .alias('FAIR'))
        #.with_columns(FAIR=pl.col('FAIR')/pl.col('UNDERLYING_PRICE'))
        .select(['STRIKE','FAIR','MID','UNDERLYING_PRICE','OPTION_RIGHT','MARK_IV','DTE'])
        .sort('STRIKE')
        )

# realized: spot.with_columns(((pl.col('UNDERLYING_PRICE').pct_change().add(1).log())).std()*np.sqrt(365))

fair_c = fair.filter(pl.col('OPTION_RIGHT')=='call')
fair_p = fair.filter(pl.col('OPTION_RIGHT')=='put')

plt.plot(fair_c['STRIKE'].to_numpy().flatten(),
         fair_c['FAIR'].to_numpy().flatten(),'-b',label='call')
plt.plot(fair_c['STRIKE'].to_numpy().flatten(),
         fair_c['MID'].to_numpy().flatten(),'--k')
plt.plot(fair_p['STRIKE'].to_numpy().flatten(),
         fair_p['FAIR'].to_numpy().flatten(),'-r',label='put')
plt.plot(fair_p['STRIKE'].to_numpy().flatten(),
         fair_p['MID'].to_numpy().flatten(),'--g')
plt.title('Option price as a function of the strike')
plt.legend()
plt.show()

## COMMENT:
# the fitted prices look nice! Next step is to calculate the prices of the flies.

# 1) let now interpolate prices: looking for an equidistant solution.

smin,smax,step = (fair
                  .with_columns(strike_min=pl.col('STRIKE').unique().min(),
                                strike_max=pl.col('STRIKE').unique().max(),
                                strike_step=pl.col('STRIKE').unique().diff().min(),
                                )
                  .select('strike_min','strike_max','strike_step')
                  ).row(0)

fair_i = pl.DataFrame({'STRIKE':np.arange(smin,smax,step)})

fair_i = (fair_i
          .join((fair
                 .filter(pl.col('OPTION_RIGHT')=='call')
                 .select(pl.col('STRIKE'),pl.col('FAIR'))),
                on='STRIKE', how='left')
          .with_columns(pl.col('FAIR').interpolate())
          )

#plot_series('STRIKE','FAIR',fair_i)


fly=(fair_i
     .with_columns(FLY=pl.col('FAIR')-2*pl.col('FAIR').shift(1) + pl.col('FAIR').shift(2))
     .with_columns(PROB=pl.col('FLY')/pl.col('FLY').sum())
     .with_columns(pl.when(pl.col('PROB')<0).then(None).otherwise(pl.col('PROB')))
     .drop_nulls()
     )

plot_series('STRIKE','PROB',fly)

# comment: finally some progress! the negative values are a bit concerning however, how to solve that???
