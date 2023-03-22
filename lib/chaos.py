def rossler(u, t, a, b, c):
    x, y, z = u
    dxdt = -y - z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return([dxdt, dydt, dzdt])


def lorenz(u, t, sigma, b, r):
    x, y, z = u
    dxdt = sigma*(y - x)
    dydt = -x*z + r*x - y
    dzdt = x*y - b*z
    return([dxdt, dydt, dzdt])