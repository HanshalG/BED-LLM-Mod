# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Serial bounded root inversion using SciPy's compiled normal CDF."""

import numpy as np
from scipy.special import ndtr as python_ndtr
from libc.math cimport exp, fabs, isfinite, sqrt, INFINITY
from scipy.special.cython_special cimport ndtr, ndtri


cdef inline double normal_cdf(double z) noexcept nogil:
    # SciPy's double-precision CDF is already exactly saturated here. This
    # skips expensive special-function evaluation, never a mixture component.
    if z >= 9:
        return 1.0
    if z <= -40:
        return 0.0
    return ndtr(z)


cdef double cdf(const double[:, :] weights, Py_ssize_t row,
                const double[:] means, const double[:] sigmas, double y) noexcept nogil:
    return cdf_part(weights, row, means, sigmas, y, 0, means.shape[0])


cdef double cdf_part(const double[:, :] weights, Py_ssize_t row,
                     const double[:] means, const double[:] sigmas, double y,
                     Py_ssize_t start, Py_ssize_t count) noexcept nogil:
    # Match NumPy's pairwise reduction of a contiguous last axis (block 128).
    cdef Py_ssize_t p, j, split, index
    cdef double lanes[8]
    cdef double total = -0.0
    if count < 8:
        for p in range(start, start + count):
            total += weights[row, p] * normal_cdf((y - means[p]) / sigmas[p])
        return total
    if count > 128:
        split = count // 2
        split -= split % 8
        return cdf_part(weights, row, means, sigmas, y, start, split) + cdf_part(weights, row, means, sigmas, y, start + split, count - split)
    for j in range(8):
        p = start + j
        lanes[j] = weights[row, p] * normal_cdf((y - means[p]) / sigmas[p])
    p = 8
    while p < count - count % 8:
        for j in range(8):
            index = start + p + j
            lanes[j] += weights[row, index] * normal_cdf((y - means[index]) / sigmas[index])
        p += 8
    total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    for j in range(p, count):
        index = start + j
        total += weights[row, index] * normal_cdf((y - means[index]) / sigmas[index])
    return total


def envelope_starts(const double[:] mu, double variance, const double[:] logs):
    cdef Py_ssize_t p, index, length = 0
    cdef double slope, intercept, start
    cdef double[:] slopes = np.empty(mu.shape[0])
    cdef double[:] intercepts = np.empty(mu.shape[0])
    cdef double[:] starts = np.empty(mu.shape[0])
    cdef const long[:] order = np.argsort(np.asarray(mu))
    if mu.shape[0] != logs.shape[0] or variance <= 0 or not isfinite(variance):
        raise ValueError("invalid envelope dimensions or variance")
    for index in range(mu.shape[0]):
        p = order[index]
        slope = mu[p] / variance
        intercept = logs[p] - 0.5 * mu[p]**2 / variance
        if not isfinite(slope) or not isfinite(intercept):
            raise ValueError("unrepresentable density envelope")
        if length and slope == slopes[length - 1]:
            if intercept <= intercepts[length - 1]:
                continue
            length -= 1
        start = -INFINITY
        while length:
            start = (intercepts[length - 1] - intercept) / (slope - slopes[length - 1])
            if start > starts[length - 1]:
                break
            length -= 1
        slopes[length], intercepts[length] = slope, intercept
        starts[length] = start if length else -INFINITY
        length += 1
    return np.asarray(starts)[:length]


def rules(const double[:, :] logs, const double[:, :] weights,
          const double[:] means, double sigma, Py_ssize_t branches,
          const double[:, :] nodes, const double[:, :] masses):
    """Same hull, CDF cuts and Legendre arithmetic, batched without Python rows."""
    cdef Py_ssize_t r, p, j, length, intervals, order, output_index, extra
    cdef double u, low, high, variance = sigma * sigma
    cdef double[:, :] probabilities = np.empty((logs.shape[0], branches))
    cdef double[:, :] integration_weights = np.empty((logs.shape[0], branches))
    cdef double[:] cuts = np.empty(means.shape[0] + 1)
    if means.shape[0] == 0 or sigma <= 0 or not isfinite(sigma) or logs.shape[0] != weights.shape[0] or logs.shape[1] != means.shape[0] or weights.shape[1] != means.shape[0] or branches < 2 or nodes.shape[0] <= branches or masses.shape[0] <= branches or nodes.shape[1] < branches or masses.shape[1] < branches:
        raise ValueError("invalid quadrature dimensions")
    weight_array, mean_array, log_array = np.asarray(weights), np.asarray(means), np.asarray(logs)
    for r in range(logs.shape[0]):
        active = weight_array[r] > 0
        mu = mean_array[active]
        w = weight_array[r, active]
        starts = envelope_starts(mu, variance, log_array[r, active])
        length = 1
        cuts[0] = 0.0
        for x in starts[1:]:
            u = float(w @ python_ndtr((x - mu) / sigma))
            if u - cuts[length - 1] > 1e-14 and 1 - u > 1e-14:
                cuts[length] = u
                length += 1
        cuts[length] = 1.0
        intervals = length
        if 2 * intervals > branches:
            from .horizon import SearchLimitExceeded
            raise SearchLimitExceeded("density envelope exceeds quadrature budget")
        extra = branches % intervals
        output_index = 0
        for p in range(intervals):
            low, high = cuts[p], cuts[p + 1]
            order = branches // intervals + (1 if p < extra else 0)
            for j in range(order):
                probabilities[r, output_index] = low + (high - low) * (nodes[order, j] + 1) / 2
                integration_weights[r, output_index] = (high - low) * masses[order, j] / 2
                output_index += 1
    return np.asarray(probabilities), np.asarray(integration_weights)


def invert(const double[:, :] weights, const double[:] means,
           const double[:] sigmas, const double[:, :] quantiles):
    cdef Py_ssize_t r, k, p, index, step, count = means.shape[0]
    cdef const long[:] order = np.argsort(np.asarray(means))
    cdef double[:, :] output = np.empty((weights.shape[0], quantiles.shape[1]))
    cdef double q, z, component, low, high, middle, cumulative, local_q
    cdef double guess, f, density, correction, tolerance, proposal, left, right
    cdef bint solved
    if count == 0 or sigmas.shape[0] != count or weights.shape[1] != count or weights.shape[0] != quantiles.shape[0]:
        raise ValueError("invalid inversion dimensions")
    for r in range(weights.shape[0]):
        for k in range(quantiles.shape[1]):
            q = quantiles[r, k]
            z = ndtri(q)
            low, high = INFINITY, -INFINITY
            for p in range(count):
                if weights[r, p] > 0:
                    component = means[p] + sigmas[p] * z
                    low = min(low, component)
                    high = max(high, component)
            if not isfinite(low) or not isfinite(high):
                raise ValueError("quantiles exceed numerical range")
            guess = low / 2 + high / 2
            cumulative = 0.0
            for index in range(count):
                p = order[index]
                if weights[r, p] > 0 and cumulative + weights[r, p] >= q:
                    local_q = min(1 - 1e-15, max(1e-15, (q - cumulative) / weights[r, p]))
                    guess = means[p] + sigmas[p] * ndtri(local_q)
                    break
                cumulative += weights[r, p]
            if not isfinite(guess) or guess <= low or guess >= high:
                guess = low / 2 + high / 2
            solved = False
            if 1e-5 < q < 1 - 1e-5:
                for step in range(12):
                    f = cdf(weights, r, means, sigmas, guess)
                    density = 0.0
                    for p in range(count):
                        z = (guess - means[p]) / sigmas[p]
                        if fabs(z) < 40:
                            density += weights[r, p] * exp(-0.5 * z * z) / sigmas[p]
                    density /= sqrt(2 * 3.141592653589793)
                    if f < q:
                        low = guess
                    else:
                        high = guess
                    correction = (f - q) / density if density > 0 else INFINITY
                    tolerance = 1e-13 * max(1, fabs(guess))
                    if isfinite(correction) and fabs(correction) <= tolerance:
                        left, right = guess - tolerance, guess + tolerance
                        if cdf(weights, r, means, sigmas, left) < q and cdf(weights, r, means, sigmas, right) >= q:
                            low, high = left, right
                            solved = True
                            break
                    proposal = guess - correction
                    guess = proposal if isfinite(proposal) and low < proposal < high else low / 2 + high / 2
            if not solved:
                for step in range(64):
                    middle = low / 2 + high / 2
                    if middle <= low or middle >= high:
                        break
                    if cdf(weights, r, means, sigmas, middle) < q:
                        low = middle
                    else:
                        high = middle
            output[r, k] = low / 2 + high / 2
    return np.asarray(output)
