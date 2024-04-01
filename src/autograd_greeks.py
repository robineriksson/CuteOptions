import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes_call_price(S, K, T, r, sigma):
    """Black-Scholes formula for European call option price using JAX."""
    d1 = (jnp.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    call_price = S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    return call_price

# Derivative of call price with respect to stock price S to calculate Delta
delta_fn = jax.grad(black_scholes_call_price, argnums=0)
gamma_fn = jax.grad(delta_fn, argnums=0)
vega_fn = jax.grad(black_scholes_call_price, argnums=4)

strike_fn = jax.grad(black_scholes_call_price, argnums=1)
strike2_fn = jax.grad(strike_fn, argnums=1)

def delta_curve(S_range, K, T, r, sigma):
    """Compute the Delta curve across a range of strike prices."""
    deltas = jax.vmap(delta_fn, in_axes=(0, None, None, None, None))(S_range, K, T, r, sigma)
    return deltas

def gamma_curve(S_range, K, T, r, sigma):
    """Compute the Gamma  curve across a range of strike prices."""
    gammas  = jax.vmap(gamma_fn, in_axes=(0, None, None, None, None))(S_range, K, T, r, sigma)
    return gammas

def vega_curve(S_range, K, T, r, sigma):
    """Compute the Gamma  curve across a range of strike prices."""
    vegas  = jax.vmap(vega_fn, in_axes=(0, None, None, None, None))(S_range, K, T, r, sigma)
    return vegas



def strike_curve(S_range, K, T, r, sigma):
    """Compute the Gamma  curve across a range of strike prices."""
    strikes  = jax.vmap(strike_fn, in_axes=(0, None, None, None, None))(S_range, K, T, r, sigma)
    return strikes


def strike2_curve(S_range, K, T, r, sigma):
    """Compute the Gamma  curve across a range of strike prices."""
    strike2s  = jnp.exp(r*T)*jax.vmap(strike2_fn, in_axes=(0, None, None, None, None))(S_range, K, T, r, sigma)
    return strike2s



def bs_delta(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * jnp.sqrt(T))
    call_price = norm.cdf(d1)
    return call_price



# Parameters
S_range = jnp.linspace(20, 180, 40)  # Range of underlying prices
K = 100.0  # strike price
T = 0.5    # Time to expiration in years
r = 0.05   # Risk-free interest rate
sigma = 0.2  # Volatility

# Calculate the Delta curve
deltas = delta_curve(S_range, K, T, r, sigma)
gammas = gamma_curve(S_range, K, T, r, sigma)
vegas = vega_curve(S_range, K, T, r, sigma)

strikes = strike_curve(S_range, K, T, r, sigma)
strike2s = strike2_curve(S_range, K, T, r, sigma)



#deltas_r = bs_delta(S_range, K, T, r, sigma)

plt.plot(S_range, deltas, label='JAX delta')
plt.plot(S_range, gammas, label='JAX gamma')
plt.plot(S_range, vegas, label='JAX vega')
#plt.plot(S_range, deltas_r, '--',label='raw')
plt.xlabel('strike')
plt.ylabel('derivative')
plt.legend()
plt.show()


#plt.plot(S_range, strikes, label='dCdK')
plt.plot(S_range, strike2s, label='dC2dK2')
#plt.plot(S_range, deltas_r, '--',label='raw')
plt.xlabel('underlying')
plt.ylabel('deriv')
plt.legend()
plt.show()


################
sigma0 = 0.3
sigma1 = 0.2
sigma2 = 0.25

prob0 = strike2_curve(S_range, K, T, r, sigma0)
prob1 = strike2_curve(S_range, K, T, r, sigma1)
prob2 = strike2_curve(S_range, K, T, r, sigma2)
prob_avg = (prob0 + prob1 + prob2)

prob_avg /= prob_avg.sum()
prob0 /= prob0.sum()
prob1 /= prob1.sum()
prob2 /= prob2.sum()

#plt.plot(S_range, strikes, label='dCdK')
plt.plot(S_range, prob0, label='prob0')
plt.plot(S_range, prob1, label='prob1')
plt.plot(S_range, prob2, label='prob2')
plt.plot(S_range, prob_avg, '--', label='prob_avg')

#plt.plot(S_range, deltas_r, '--',label='raw')
plt.xlabel('underlying')
plt.ylabel('deriv')
plt.legend()
plt.show()
