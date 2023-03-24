"""
Common definitions
"""

moment_definitions = r"""
Unidimensional moments
======================

rms: raw moments.
cms: central moments.
sms: scaled central moments.

rmss: plural, a collection of rms. The same goes for cmss and smss.
"""

generating_function_definitions = r"""
Moment-generating function M(z) := E[e^{z X}] = \sum_{n=0} z^n / n! E[X^n]

Central moment-generating function C(z) := E[e^{z (X - mu)}] = e^{-(z mu)} M(z) = \sum_{n=0} z^n / n! E[(X - mu)^n]

Scaled moment-generating function S(z) := E[exp(z (X - mu) / scale)] = e^{-(z mu) / sigma} M(z / scale) 
                                                                     = \sum_{n=0} z^n / n! E[((X - mu) / scale)^n]

Cumulant-generating function K(z) := log(M(z)) = log(e^{z mu} S(z scale)) = log(e^{z mu} C(z))
"""
