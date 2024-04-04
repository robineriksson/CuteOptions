import polars as pl
import datetime
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

# [X] load csv
# [X] calc the price of flies
# [X] normalize/smooth
# [ ] repeat per expiry
# [ ] chart

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




def plot_series(xcol:str, ycol_list:str, df:pl.DataFrame, style='-'):
    if not isinstance(ycol_list, list):
        ycol_list = [ycol_list]

    x = df.select(pl.col(xcol)).to_numpy().flatten()
    plt.figure()
    for ycol in ycol_list:
        y = df.select(pl.col(ycol)).to_numpy().flatten()
        plt.plot(x,y,style, label=ycol)
    plt.gcf().autofmt_xdate()
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend()
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


# black scholes price that bitch ....



def bs_price(F, K, sigma, DTE, option_right:str):
    assert option_right in ['call','put'], 'option right must be put or call'

    d1 = (jnp.log(F/K) + (sigma**2/2)*DTE) / (sigma*jnp.sqrt(DTE))
    d2 = d1 - sigma*jnp.sqrt(DTE)

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


(df.filter((pl.col('EXPIRY_DATE')==expiry_date) &
                  (pl.col('QUOTE_DATE')==end_date))
 .select(['STRIKE','MID','UNDERLYING_PRICE','OPTION_RIGHT','MARK_IV','DTE'])
 .sort('STRIKE')
 )

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

delta_strike_fn = jax.grad(bs_price, argnums=1)
gamma_strike_fn = jax.grad(delta_strike_fn, argnums=1)

def calc_rnp(F_range, K, sigma, DTE, option_right:str):
    """ the risk neutral pdf """
    gamma_strike = jax.vmap(gamma_strike_fn, (0,None,None,None,None))(F_range, K,
                                                                      sigma, DTE,
                                                                      option_right)
    return jnp.exp(DTE)*gamma_strike

F_range = jnp.linspace(5_000, 100_000, 100)
K = 50_000.
sigma=0.5
DTE=90/365
option_right='call'
rnp = calc_rnp(F_range, K, sigma, DTE, option_right)

plt.plot(F_range, rnp, label='Risk netural density')
plt.show()

# next steps:
# 1) get this line for all the strikes for one expiry
# - would loop over (strike, sigma) pairs.
# 2) do we have OI? thinking this could be used to weight in the average
# 3) for a specific contract, evolve this chart through time. Should be a nice graph
# 4) add the timeseries, and move the uncertainty as a cone like thingy.

def calc_rnp_help(x):
    F_range = jnp.linspace(1_000, 120_000, 1_00) #x['UNDERLYING_PRICE']
    return calc_rnp(F_range=F_range,K=x['STRIKE'],
                    sigma=x['MARK_IV'],DTE=x['DTE'],
                    option_right=x['OPTION_RIGHT'])

# run only every friday. Data is expensive to run more ...
rnp_exp = (df.filter((pl.col('EXPIRY_DATE')==expiry_date) & (pl.col('QUOTE_DATE').dt.weekday()==5))
        .with_columns(pl.struct(pl.col('STRIKE'),
                                pl.col('MARK_IV')/100,
                                pl.col('DTE')/365,
                                pl.col('OPTION_RIGHT')).map_elements(lambda x: calc_rnp_help(x),
                                                                     return_dtype=pl.Object).alias('STRIKE_GAMMA'))
           .select(['QUOTE_DATE','STRIKE_GAMMA'])
           )

rnp_exp.group_by('QUOTE_DATE').map_groups(lambda x: jnp.vstack(x['STRIKE_GAMMA']).mean(axis=0))


xxxxxxxxxxxxxxxxxxx
# dummy way of doing it seems the best way ...

j=0.5
for quote_date in rnp_exp['QUOTE_DATE'].unique():
    rnp_exp_avg = jnp.vstack(rnp_exp.filter(pl.col('QUOTE_DATE')==quote_date)['STRIKE_GAMMA']).mean(axis=0)
    plt.plot(F_range, rnp_exp_avg, label=quote_date, color='k', alpha=j)
    j *= 1.1

plt.legend()
plt.show()

xxx


xxxxxxx
rnp_exp_avg = jnp.vstack(rnp_exp['STRIKE_GAMMA']).mean(axis=0)

rnp_exp_avg /= rnp_exp_avg.sum()
F_range = jnp.linspace(1_000, 120_000, 1_00)

plt.plot(F_range, rnp_exp_avg, label='Risk netural probablility')
plt.axvline(44131, color='k', label='underlying')
plt.axvline(F_range @ rnp_exp_avg, color='r', linestyle='--', label='mean prob')
plt.legend()
plt.show()

xxxxxx
# realized: spot.with_columns(((pl.col('UNDERLYING_PRICE').pct_change().add(1).log())).std()*np.sqrt(365))

fair_c = fair.filter(pl.col('OPTION_RIGHT')=='call')
fair_p = fair.filter(pl.col('OPTION_RIGHT')=='put')

plot_series('STRIKE','MARK_IV',fair_c,'.')

from scipy.ndimage import gaussian_filter
plot_series('STRIKE',['MARK_IV','MARK_IVs'],
            fair_c.with_columns(MARK_IVs = gaussian_filter(fair_c['MARK_IV'], sigma=2)),
            '.')

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
                                strike_step=pl.col('STRIKE').unique().diff().min()/1,
                                )
                  .select('strike_min','strike_max','strike_step')
                  ).row(0)

fair_i = (pl
          .DataFrame({'STRIKE':jnp.arange(smin,smax,step)})
          .join((fair
                 .filter(pl.col('OPTION_RIGHT')=='call')
                 .select(pl.col('STRIKE'),pl.col('FAIR'))
                 ), on='STRIKE', how='left')
          .with_columns(pl.col('FAIR').interpolate())
          # .with_columns(DELTA=pl.col('FAIR').diff())
          )



plot_series('STRIKE',['FAIR'],fair_i, '.-')

plot_series('STRIKE',['DELTA'],fair_i)


fly=(fair_i
     .with_columns(FLY=pl.col('FAIR')-2*pl.col('FAIR').shift(1) + pl.col('FAIR').shift(2),
                   MAXPROFIT=pl.col('STRIKE').shift(-1) - pl.col('STRIKE'))
     .with_columns(FLY=pl.when(pl.col('FLY')<0).then(None).otherwise(pl.col('FLY')))
     .with_columns(PROB=pl.col('FLY')/pl.col('MAXPROFIT'))
     .drop_nulls()
     )

plot_series('STRIKE',['PROB'],fly,'.-')

plot_series('STRIKE',['PROB', 'PROBs'],
            fly.with_columns(PROBs = gaussian_filter(fly['PROB'], sigma=5)),
            '.')

xxxxxxxxxxxxxxxxxxx
fly_tight = fly.filter((pl.col('STRIKE')>40000) & (pl.col('STRIKE')<43000))
plot_series('STRIKE',['FAIR'],fly_tight,'.-')

plot_series('STRIKE',['DELTA'],fly_tight,'.-')


train = fair_i.drop_nulls()
test = fair_i.filter(pl.col('FAIR').is_null())
model.fit(train.select('STRIKE'), train.select('FAIR'))
pred = model.predict(test.select('STRIKE'))

fair_clean = (pl
              .concat([train, test.with_columns(FAIR=pred.ravel())]).sort('STRIKE')
              .with_columns(DELTA=pl.col('FAIR').diff())
              )

plot_series('STRIKE',['FAIR'],fair_clean,'.-')
plot_series('STRIKE',['DELTA'],fair_clean,'.-')

fly_clean=(fair_clean
     .with_columns(FLY=pl.col('FAIR')-2*pl.col('FAIR').shift(1) + pl.col('FAIR').shift(2))
     .with_columns(FLY=pl.when(pl.col('FLY')<0).then(None).otherwise(pl.col('FLY')))
     .with_columns(PROB=pl.col('FLY')/pl.col('FLY').sum())
     .with_columns(pl.col('PROB').interpolate())
     )


plot_series('STRIKE',['PROB'],fly_clean,'.-')
