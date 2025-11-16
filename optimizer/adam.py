# ###
# Adam adaptive moment estimation
# parameter: alpha, t, b1,b2, m, v, theta, eps
# while theta not converged
# update time
# t += 1
# update bias
# m(t) = b1 * m(t - 1) + (1 - b1) * g
# v(t) = b2 * v(t - 1) + (1 - b2) * g**2
# bias correction
# m'(t) = m(t-1)/sqrt(1 - b1**t)
# v'(t) = v(t-1)/sqrt(1 - b2**t)
# update theta
# theta(t) = theta(t-1) - alpha * (m'(t)) / sqrt(v'(t) + eps)
# return theta
# ###

