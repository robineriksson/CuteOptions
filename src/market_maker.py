import numpy as np
import matplotlib.pyplot as plt

def reservation_price(s, q, gamma, sigma, T, t):
    """Calculate the reservation price."""
    return s - q * gamma * sigma**2 * (T - t)

def spread(gamma, sigma, T, t, k):
    """Calculate the spread."""
    return gamma * sigma**2 * (T - t) + (2 / gamma) * np.log(1 + (gamma / k))

def optimal_quote(reservation_price, spread, side):
    """
    Calculate the optimal bid or ask quote.
    'side' should be -1 for bid and 1 for ask.
    """
    return reservation_price + (side * spread / 2)

def lambda_arrival(delta, k, A):
    """
    Calculate the arrival rate of orders based on the distance from the mid-price.

    :param delta: The absolute distance from the quote to the mid-price.
    :param k: A parameter that affects the decay rate of the intensity as a function of delta.
    :param A: A scaling factor for the intensity.
    :return: The intensity of order arrivals.
    """
    return A * np.exp(-k * delta)


def test_bid(q=0,gamma=0.5,t=0.5):
    # Assuming we have the following variables already defined:
    # s: mid-market price
    # q: current inventory
    # gamma: risk aversion
    # sigma: volatility
    # T: terminal time
    # t: current time
    # k: intensity parameter

    # Calculate reservation price
    r = reservation_price(s, q, gamma, sigma, T, t)

    # Calculate spread
    spread_value = spread(gamma, sigma, T, t, k)

    # Calculate optimal bid and ask quotes
    optimal_bid = optimal_quote(r, spread_value, -1)
    optimal_ask = optimal_quote(r, spread_value, 1)

    print((optimal_bid, optimal_ask))

def run_simulation(seed=42):
    np.random.seed(seed)
    # Initialize cash balance and a list to store transaction prices for buys and sells
    cash = 0
    buy_prices = []
    sell_prices = []

    # Initialize variables for the start of the simulation
    s = s0
    q = q0
    x = 0  # initial cash position

    # Initialize P&L tracking
    realized_pnl = 0
    unrealized_pnl = 0
    total_pnl = 0

    # Simulation loop for a single path
    for t in np.arange(0, T, dt):
        # Update mid-market price using a geometric Brownian motion (placeholder for actual implementation)
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        s += sigma * s * dW  # Update of mid-market price

        # Calculate reservation price
        r = reservation_price(s, q, gamma, sigma, T, t)

        # Calculate spread based on current market conditions and agent's risk preference
        current_spread = spread(gamma, sigma, T, t, k)

        # Calculate optimal bid and ask quotes
        optimal_bid = optimal_quote(r, current_spread, -1)
        optimal_ask = optimal_quote(r, current_spread, 1)

        # Simulate order arrivals
        if np.random.rand() < lambda_arrival(s - optimal_bid, k, A) * dt:
            # Update inventory and cash balance for a buy order
            q += 1
            cash -= optimal_bid
            buy_prices.append(optimal_bid)  # Track buy price

        if np.random.rand() < lambda_arrival(optimal_ask - s, k, A) * dt:
            # Update inventory and cash balance for a sell order
            q -= 1
            cash += optimal_ask
            sell_prices.append(optimal_ask)  # Track sell price
            # Calculate realized P&L for the sell order
            realized_pnl += optimal_ask - buy_prices.pop(0)  # Assume FIFO for the cost basis

        # Update unrealized P&L for the remaining inventory
        if q > 0:
            unrealized_pnl = q * (s - sum(buy_prices) / len(buy_prices)) if buy_prices else 0
        elif q < 0:
            unrealized_pnl = q * (sum(sell_prices) / len(sell_prices) - s) if sell_prices else 0

        # Update total P&L
        total_pnl = realized_pnl + unrealized_pnl

    # Final P&L will be the total P&L plus cash balance if the inventory is zero
    final_pnl = total_pnl + cash if q == 0 else None


    # Store the current state
    mid_prices.append(s)
    bid_prices.append(optimal_bid)
    ask_prices.append(optimal_ask)
    inventories.append(q)

    return np.array(mid_prices), np.array(bid_prices), np.array(ask_prices), np.array(inventories)

# At the end of the loop, we have lists of mid-prices, bid prices, ask prices, and inventories over time

# Define the parameters for the simulation
s0 = 100  # initial mid-market price
q0 = 0    # initial inventory
gamma = 0.1  # risk aversion parameter
sigma = 2    # volatility
T = 1       # terminal time
k = 1.5     # intensity parameter
A = 140     # arrival rate scale factor
dt = 0.005  # time step
seed = 42

mid, bid, ask, inv = run_simulation()

time = np.arange(0, T, dt)

plt.plot(time, mid, label='mid')
plt.plot(time, bid, label='bid')
plt.plot(time, ask, label='ask')
plt.legend()
plt.show()

plt.plot(time, inv)
plt.show()
