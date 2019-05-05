from __future__ import division, print_function, absolute_import

from scipy._lib.six import string_types, exec_, PY3
from scipy._lib._util import getargspec_no_self as _getargspec

import sys
import keyword
import re
import types
import warnings

from scipy.misc import doccer
# from scipy._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy._lib._util import _valarray as valarray

from scipy.special import (comb, chndtr, entr, rel_entr, xlogy, ive)

from numpy import (arange, putmask, ravel, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, NINF, empty)

import numpy as np

# from ._constants import _XMAX

if PY3:
    def instancemethod(func, obj, cls):
        return types.MethodType(func, obj)
else:
    instancemethod = types.MethodType


# These are the docstring parts used for substitution in specific
# distribution docstrings
"""
Sane parameters for stats.distributions.
"""

distcont = [
    ['alpha', (3.5704770516650459,)],
    ['anglit', ()],
    ['arcsine', ()],
    ['argus', (1.0,)],
    ['beta', (2.3098496451481823, 0.62687954300963677)],
    ['betaprime', (5, 6)],
    ['bradford', (0.29891359763170633,)],
    ['burr', (10.5, 4.3)],
    ['burr12', (10, 4)],
    ['cauchy', ()],
    ['chi', (78,)],
    ['chi2', (55,)],
    ['cosine', ()],
    ['crystalball', (2.0, 3.0)],
    ['dgamma', (1.1023326088288166,)],
    ['dweibull', (2.0685080649914673,)],
    ['erlang', (10,)],
    ['expon', ()],
    ['exponnorm', (1.5,)],
    ['exponpow', (2.697119160358469,)],
    ['exponweib', (2.8923945291034436, 1.9505288745913174)],
    ['f', (29, 18)],
    ['fatiguelife', (29,)],   # correction numargs = 1
    ['fisk', (3.0857548622253179,)],
    ['foldcauchy', (4.7164673455831894,)],
    ['foldnorm', (1.9521253373555869,)],
    ['frechet_l', (3.6279911255583239,)],
    ['frechet_r', (1.8928171603534227,)],
    ['gamma', (1.9932305483800778,)],
    ['gausshyper', (13.763771604130699, 3.1189636648681431,
                    2.5145980350183019, 5.1811649903971615)],  # veryslow
    ['genexpon', (9.1325976465418908, 16.231956600590632, 3.2819552690843983)],
    ['genextreme', (-0.1,)],
    ['gengamma', (4.4162385429431925, 3.1193091679242761)],
    ['gengamma', (4.4162385429431925, -3.1193091679242761)],
    ['genhalflogistic', (0.77274727809929322,)],
    ['genlogistic', (0.41192440799679475,)],
    ['gennorm', (1.2988442399460265,)],
    ['halfgennorm', (0.6748054997000371,)],
    ['genpareto', (0.1,)],   # use case with finite moments
    ['gilbrat', ()],
    ['gompertz', (0.94743713075105251,)],
    ['gumbel_l', ()],
    ['gumbel_r', ()],
    ['halfcauchy', ()],
    ['halflogistic', ()],
    ['halfnorm', ()],
    ['hypsecant', ()],
    ['invgamma', (4.0668996136993067,)],
    ['invgauss', (0.14546264555347513,)],
    ['invweibull', (10.58,)],
    ['johnsonsb', (4.3172675099141058, 3.1837781130785063)],
    ['johnsonsu', (2.554395574161155, 2.2482281679651965)],
    ['kappa4', (0.0, 0.0)],
    ['kappa4', (-0.1, 0.1)],
    ['kappa4', (0.0, 0.1)],
    ['kappa4', (0.1, 0.0)],
    ['kappa3', (1.0,)],
    ['ksone', (1000,)],  # replace 22 by 100 to avoid failing range, ticket 956
    ['kstwobign', ()],
    ['laplace', ()],
    ['levy', ()],
    ['levy_l', ()],
    ['levy_stable', (1.8, -0.5)],
    ['loggamma', (0.41411931826052117,)],
    ['logistic', ()],
    ['loglaplace', (3.2505926592051435,)],
    ['lognorm', (0.95368226960575331,)],
    ['lomax', (1.8771398388773268,)],
    ['maxwell', ()],
    ['mielke', (10.4, 3.6)],
    ['moyal', ()],
    ['nakagami', (4.9673794866666237,)],
    ['ncf', (27, 27, 0.41578441799226107)],
    ['nct', (14, 0.24045031331198066)],
    ['ncx2', (21, 1.0560465975116415)],
    ['norm', ()],
    ['norminvgauss', (1., 0.5)],
    ['pareto', (2.621716532144454,)],
    ['pearson3', (0.1,)],
    ['powerlaw', (1.6591133289905851,)],
    ['powerlognorm', (2.1413923530064087, 0.44639540782048337)],
    ['powernorm', (4.4453652254590779,)],
    ['rayleigh', ()],
    ['rdist', (0.9,)],   # feels also slow
    ['recipinvgauss', (0.63004267809369119,)],
    ['reciprocal', (0.0062309367010521255, 1.0062309367010522)],
    ['rice', (0.7749725210111873,)],
    ['semicircular', ()],
    ['skewnorm', (4.0,)],
    ['t', (2.7433514990818093,)],
    ['trapz', (0.2, 0.8)],
    ['triang', (0.15785029824528218,)],
    ['truncexpon', (4.6907725456810478,)],
    ['truncnorm', (-1.0978730080013919, 2.7306754109031979)],
    ['truncnorm', (0.1, 2.)],
    ['tukeylambda', (3.1321477856738267,)],
    ['uniform', ()],
    ['vonmises', (3.9939042581071398,)],
    ['vonmises_line', (3.9939042581071398,)],
    ['wald', ()],
    ['weibull_max', (2.8687961709100187,)],
    ['weibull_min', (1.7866166930421596,)],
    ['wrapcauchy', (0.031071279018614728,)]]


distdiscrete = [
    ['bernoulli',(0.3,)],
    ['binom', (5, 0.4)],
    ['boltzmann',(1.4, 19)],
    ['dlaplace', (0.8,)],  # 0.5
    ['geom', (0.5,)],
    ['hypergeom',(30, 12, 6)],
    ['hypergeom',(21,3,12)],  # numpy.random (3,18,12) numpy ticket:921
    ['hypergeom',(21,18,11)],  # numpy.random (18,3,11) numpy ticket:921
    ['logser', (0.6,)],  # re-enabled, numpy ticket:921
    ['nbinom', (5, 0.5)],
    ['nbinom', (0.4, 0.4)],  # from tickets: 583
    ['planck', (0.51,)],   # 4.1
    ['poisson', (0.6,)],
    ['randint', (7, 31)],
    ['skellam', (15, 8)],
    ['zipf', (6.5,)],
    ['yulesimon',(11.0,)]
]

docheaders = {'methods': """\nMethods\n-------\n""",
              'notes': """\nNotes\n-----\n""",
              'examples': """\nExamples\n--------\n"""}

_doc_rvs = """\
rvs(%(shapes)s, loc=0, scale=1, size=1, random_state=None)
    Random variates.
"""
_doc_pdf = """\
pdf(x, %(shapes)s, loc=0, scale=1)
    Probability density function.
"""
_doc_logpdf = """\
logpdf(x, %(shapes)s, loc=0, scale=1)
    Log of the probability density function.
"""
_doc_pmf = """\
pmf(k, %(shapes)s, loc=0, scale=1)
    Probability mass function.
"""
_doc_logpmf = """\
logpmf(k, %(shapes)s, loc=0, scale=1)
    Log of the probability mass function.
"""
_doc_cdf = """\
cdf(x, %(shapes)s, loc=0, scale=1)
    Cumulative distribution function.
"""
_doc_logcdf = """\
logcdf(x, %(shapes)s, loc=0, scale=1)
    Log of the cumulative distribution function.
"""
_doc_sf = """\
sf(x, %(shapes)s, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
"""
_doc_logsf = """\
logsf(x, %(shapes)s, loc=0, scale=1)
    Log of the survival function.
"""
_doc_ppf = """\
ppf(q, %(shapes)s, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
"""
_doc_isf = """\
isf(q, %(shapes)s, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
"""
_doc_moment = """\
moment(n, %(shapes)s, loc=0, scale=1)
    Non-central moment of order n
"""
_doc_stats = """\
stats(%(shapes)s, loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
"""
_doc_entropy = """\
entropy(%(shapes)s, loc=0, scale=1)
    (Differential) entropy of the RV.
"""
_doc_fit = """\
fit(data, %(shapes)s, loc=0, scale=1)
    Parameter estimates for generic data.
"""
_doc_expect = """\
expect(func, args=(%(shapes_)s), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
"""
_doc_expect_discrete = """\
expect(func, args=(%(shapes_)s), loc=0, lb=None, ub=None, conditional=False)
    Expected value of a function (of one argument) with respect to the distribution.
"""
_doc_median = """\
median(%(shapes)s, loc=0, scale=1)
    Median of the distribution.
"""
_doc_mean = """\
mean(%(shapes)s, loc=0, scale=1)
    Mean of the distribution.
"""
_doc_var = """\
var(%(shapes)s, loc=0, scale=1)
    Variance of the distribution.
"""
_doc_std = """\
std(%(shapes)s, loc=0, scale=1)
    Standard deviation of the distribution.
"""
_doc_interval = """\
interval(alpha, %(shapes)s, loc=0, scale=1)
    Endpoints of the range that contains alpha percent of the distribution
"""
_doc_allmethods = ''.join([docheaders['methods'], _doc_rvs, _doc_pdf,
                           _doc_logpdf, _doc_cdf, _doc_logcdf, _doc_sf,
                           _doc_logsf, _doc_ppf, _doc_isf, _doc_moment,
                           _doc_stats, _doc_entropy, _doc_fit,
                           _doc_expect, _doc_median,
                           _doc_mean, _doc_var, _doc_std, _doc_interval])

_doc_default_longsummary = """\
As an instance of the `rv_continuous` class, `%(name)s` object inherits from it
a collection of generic methods (see below for the full list),
and completes them with details specific for this particular distribution.
"""

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape,
location, and scale parameters returning a "frozen" continuous RV object:

rv = %(name)s(%(shapes)s, loc=0, scale=1)
    - Frozen RV object with the same methods but holding the given shape,
      location, and scale fixed.
"""
_doc_default_example = """\
Examples
--------
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate a few first moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability density function (``pdf``):

>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),
...                 %(name)s.ppf(0.99, %(shapes)s), 100)
>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),
...        'r-', lw=5, alpha=0.6, label='%(name)s pdf')

Alternatively, the distribution object can be called (as a function)
to fix the shape, location and scale parameters. This returns a "frozen"
RV object holding the given parameters fixed.

Freeze the distribution and display the frozen ``pdf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

Check accuracy of ``cdf`` and ``ppf``:

>>> vals = %(name)s.ppf([0.001, 0.5, 0.999], %(shapes)s)
>>> np.allclose([0.001, 0.5, 0.999], %(name)s.cdf(vals, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)

And compare the histogram:

>>> ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

"""

_doc_default_locscale = """\
The probability density above is defined in the "standardized" form. To shift
and/or scale the distribution use the ``loc`` and ``scale`` parameters.
Specifically, ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically
equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
``y = (x - loc) / scale``.
"""

_doc_default = ''.join([_doc_default_longsummary,
                        _doc_allmethods,
                        '\n',
                        _doc_default_example])

_doc_default_before_notes = ''.join([_doc_default_longsummary,
                                     _doc_allmethods])

docdict = {
    'rvs': _doc_rvs,
    'pdf': _doc_pdf,
    'logpdf': _doc_logpdf,
    'cdf': _doc_cdf,
    'logcdf': _doc_logcdf,
    'sf': _doc_sf,
    'logsf': _doc_logsf,
    'ppf': _doc_ppf,
    'isf': _doc_isf,
    'stats': _doc_stats,
    'entropy': _doc_entropy,
    'fit': _doc_fit,
    'moment': _doc_moment,
    'expect': _doc_expect,
    'interval': _doc_interval,
    'mean': _doc_mean,
    'std': _doc_std,
    'var': _doc_var,
    'median': _doc_median,
    'allmethods': _doc_allmethods,
    'longsummary': _doc_default_longsummary,
    'frozennote': _doc_default_frozen_note,
    'example': _doc_default_example,
    'default': _doc_default,
    'before_notes': _doc_default_before_notes,
    'after_notes': _doc_default_locscale
}

# Reuse common content between continuous and discrete docs, change some
# minor bits.
docdict_discrete = docdict.copy()

docdict_discrete['pmf'] = _doc_pmf
docdict_discrete['logpmf'] = _doc_logpmf
docdict_discrete['expect'] = _doc_expect_discrete
_doc_disc_methods = ['rvs', 'pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf',
                     'ppf', 'isf', 'stats', 'entropy', 'expect', 'median',
                     'mean', 'var', 'std', 'interval']
for obj in _doc_disc_methods:
    docdict_discrete[obj] = docdict_discrete[obj].replace(', scale=1', '')

_doc_disc_methods_err_varname = ['cdf', 'logcdf', 'sf', 'logsf']
for obj in _doc_disc_methods_err_varname:
    docdict_discrete[obj] = docdict_discrete[obj].replace('(x, ', '(k, ')

docdict_discrete.pop('pdf')
docdict_discrete.pop('logpdf')

_doc_allmethods = ''.join([docdict_discrete[obj] for obj in _doc_disc_methods])
docdict_discrete['allmethods'] = docheaders['methods'] + _doc_allmethods

docdict_discrete['longsummary'] = _doc_default_longsummary.replace(
    'rv_continuous', 'rv_discrete')

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape and
location parameters returning a "frozen" discrete RV object:

rv = %(name)s(%(shapes)s, loc=0)
    - Frozen RV object with the same methods but holding the given shape and
      location fixed.
"""
docdict_discrete['frozennote'] = _doc_default_frozen_note

_doc_default_discrete_example = """\
Examples
--------
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate a few first moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability mass function (``pmf``):

>>> x = np.arange(%(name)s.ppf(0.01, %(shapes)s),
...               %(name)s.ppf(0.99, %(shapes)s))
>>> ax.plot(x, %(name)s.pmf(x, %(shapes)s), 'bo', ms=8, label='%(name)s pmf')
>>> ax.vlines(x, 0, %(name)s.pmf(x, %(shapes)s), colors='b', lw=5, alpha=0.5)

Alternatively, the distribution object can be called (as a function)
to fix the shape and location. This returns a "frozen" RV object holding
the given parameters fixed.

Freeze the distribution and display the frozen ``pmf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
...         label='frozen pmf')
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

Check accuracy of ``cdf`` and ``ppf``:

>>> prob = %(name)s.cdf(x, %(shapes)s)
>>> np.allclose(x, %(name)s.ppf(prob, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)
"""


_doc_default_discrete_locscale = """\
The probability mass function above is defined in the "standardized" form.
To shift distribution use the ``loc`` parameter.
Specifically, ``%(name)s.pmf(k, %(shapes)s, loc)`` is identically
equivalent to ``%(name)s.pmf(k - loc, %(shapes)s)``.
"""

docdict_discrete['example'] = _doc_default_discrete_example
docdict_discrete['after_notes'] = _doc_default_discrete_locscale

_doc_default_before_notes = ''.join([docdict_discrete['longsummary'],
                                     docdict_discrete['allmethods']])
docdict_discrete['before_notes'] = _doc_default_before_notes

_doc_default_disc = ''.join([docdict_discrete['longsummary'],
                             docdict_discrete['allmethods'],
                             docdict_discrete['frozennote'],
                             docdict_discrete['example']])
docdict_discrete['default'] = _doc_default_disc

# clean up all the separate docstring elements, we do not need them anymore
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
del obj
try:
    del s
except NameError:
    # in Python 3, loop variables are not visible after the loop
    pass


def _moment(data, n, mu=None):
    if mu is None:
        mu = data.mean()
    return ((data - mu)**n).mean()


def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if (n == 0):
        return 1.0
    elif (n == 1):
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif (n == 2):
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu*mu
    elif (n == 3):
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)  # 3rd central moment
            val = mu3+3*mu*mu2+mu*mu*mu  # 3rd non-central moment
    elif (n == 4):
        if g1 is None or g2 is None or mu2 is None or mu is None:
            val = moment_func(4, *args)
        else:
            mu4 = (g2+3.0)*(mu2**2.0)  # 4th central moment
            mu3 = g1*np.power(mu2, 1.5)  # 3rd central moment
            val = mu4+4*mu*mu3+6*mu*mu*mu2+mu*mu*mu*mu
    else:
        val = moment_func(n, *args)

    return val


def _skew(data):
    """
    skew is third central moment / variance**(1.5)
    """
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m3 = ((data - mu)**3).mean()
    return m3 / np.power(m2, 1.5)


def _kurtosis(data):
    """
    kurtosis is fourth central moment / variance**2 - 3
    """
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m4 = ((data - mu)**4).mean()
    return m4 / m2**2 - 3


# Frozen RV class
class rv_frozen(object):

    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        # a, b may be set in _argcheck, depending on *args, **kwds. Ouch.
        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.dist._argcheck(*shapes)
        self.a, self.b = self.dist.a, self.dist.b

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def pdf(self, x):   # raises AttributeError in frozen discrete distribution
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, n):
        return self.dist.moment(n, *self.args, **self.kwds)

    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    def pmf(self, k):
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):
        return self.dist.logpmf(k, *self.args, **self.kwds)

    def interval(self, alpha):
        return self.dist.interval(alpha, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect method only accepts shape parameters as positional args
        # hence convert self.args, self.kwds, also loc/scale
        # See the .expect method docstrings for the meaning of
        # other parameters.
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)


# This should be rewritten
def argsreduce(cond, *args):
    """Return the sequence of ravel(args[i]) where ravel(condition) is
    True in 1D.

    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4, 5))
    >>> B = 2
    >>> C = rand((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> B1.shape
    (20,)
    >>> cond[2,:] = 0
    >>> [A2, B2, C2] = argsreduce(cond, A, B, C)
    >>> B2.shape
    (15,)

    """
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs, ]
    expand_arr = (cond == cond)
    return [np.extract(cond, arr1 * expand_arr) for arr1 in newargs]


parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s

def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)

def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""


# Both the continuous and discrete distributions depend on ncx2.
# The function name ncx2 is an abbreviation for noncentral chi squared.

def _ncx2_log_pdf(x, df, nc):
    # We use (xs**2 + ns**2)/2 = (xs - ns)**2/2  + xs*ns, and include the
    # factor of exp(-xs*ns) into the ive function to improve numerical
    # stability at large values of xs. See also `rice.pdf`.
    df2 = df/2.0 - 1.0
    xs, ns = np.sqrt(x), np.sqrt(nc)
    res = xlogy(df2/2.0, x/nc) - 0.5*(xs - ns)**2
    res += np.log(ive(df2, xs*ns) / 2.0)
    return res


def _ncx2_pdf(x, df, nc):
    return np.exp(_ncx2_log_pdf(x, df, nc))


def _ncx2_cdf(x, df, nc):
    return chndtr(x, df, nc)


class rv_generic(object):
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    """
    def __init__(self, seed=None):
        super(rv_generic, self).__init__()

        # figure out if _stats signature has 'moments' keyword
        sign = _getargspec(self._stats)
        self._stats_has_moments = ((sign[2] is not None) or
                                   ('moments' in sign[0]))
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the RandomState object for generating random variates.

        This can be either None or an existing RandomState object.

        If None (or np.random), use the RandomState singleton used by np.random.
        If already a RandomState instance, use it.
        If an int, use a new RandomState instance seeded with seed.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def __getstate__(self):
        return self._updated_ctor_param(), self._random_state

    def __setstate__(self, state):
        ctor_param, r = state
        self.__init__(**ctor_param)
        self._random_state = r
        return self

    def _construct_argparser(
            self, meths_to_inspect, locscale_in, locscale_out):
        """Construct the parser for the shape arguments.

        Generates the argument-parsing functions dynamically and attaches
        them to the instance.
        Is supposed to be called in __init__ of a class for each distribution.

        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.

        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """

        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, string_types):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getargspec(meth)   # NB: does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.keywords is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )
        ns = {}
        exec_(parse_arg_template % dct, ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name,
                    instancemethod(ns[name], self, self.__class__)
                    )

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''

        if shapes_vals is None:
            shapes_vals = ()
        vals = ', '.join('%.3g' % val for val in shapes_vals)
        tempdict['vals'] = vals

        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','

        if self.shapes:
            tempdict['set_vals_stmt'] = '>>> %s = %s' % (self.shapes, vals)
        else:
            tempdict['set_vals_stmt'] = ''

        if self.shapes is None:
            # remove shapes from call parameters if there are none
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace(
                    "\n%(shapes)s : array_like\n    shape parameters", "")
        for i in range(2):
            if self.shapes is None:
                # necessary because we use %(shapes)s in two forms (w w/o ", ")
                self.__doc__ = self.__doc__.replace("%(shapes)s, ", "")
            self.__doc__ = doccer.docformat(self.__doc__, tempdict)

        # correct for empty shapes
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def _construct_default_doc(self, longname=None, extradoc=None,
                               docdict=None, discrete='continuous'):
        """Construct instance docstring from the default template."""
        if longname is None:
            longname = 'A'
        if extradoc is None:
            extradoc = ''
        if extradoc.startswith('\n\n'):
            extradoc = extradoc[2:]
        self.__doc__ = ''.join(['%s %s random variable.' % (longname, discrete),
                                '\n\n%(before_notes)s\n', docheaders['notes'],
                                extradoc, '\n%(example)s'])
        self._construct_doc(docdict)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.

        """
        return rv_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    # The actual calculation functions (no basic checking need be done)
    # If these are defined, the others won't be looked at.
    # Otherwise, the other set can be defined.
    def _stats(self, *args, **kwds):
        return None, None, None, None

    #  Central moments
    def _munp(self, n, *args):
        # Silence floating point warnings from integration.
        olderr = np.seterr(all='ignore')
        vals = self.generic_moment(n, *args)
        np.seterr(**olderr)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        # Handle broadcasting and size validation of the rvs method.
        # Subclasses should not have to override this method.
        # The rule is that if `size` is not None, then `size` gives the
        # shape of the result (integer values of `size` are treated as
        # tuples with length 1; i.e. `size=3` is the same as `size=(3,)`.)
        #
        # `args` is expected to contain the shape parameters (if any), the
        # location and the scale in a flat tuple (e.g. if there are two
        # shape parameters `a` and `b`, `args` will be `(a, b, loc, scale)`).
        # The only keyword argument expected is 'size'.
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        # Eliminate trivial leading dimensions.  In the convention
        # used by numpy's random variate generators, trivial leading
        # dimensions are effectively ignored.  In other words, when `size`
        # is given, trivial leading dimensions of the broadcast parameters
        # in excess of the number of dimensions  in size are ignored, e.g.
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]], size=3)
        #   array([ 1.00104267,  3.00422496,  4.99799278])
        # If `size` is not given, the exact broadcast shape is preserved:
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]])
        #   array([[[[ 1.00862899,  3.00061431,  4.99867122]]]])
        #
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        # Check compatibility of size_ with the broadcast shape of all
        # the parameters.  This check is intended to be consistent with
        # how the numpy random variate generators (e.g. np.random.normal,
        # np.random.beta) handle their arguments.   The rule is that, if size
        # is given, it determines the shape of the output.  Broadcasting
        # can't change the output size.

        # This is the standard broadcasting convention of extending the
        # shape with fewer dimensions with enough dimensions of length 1
        # so that the two shapes have the same number of dimensions.
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,)*(-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,)*ndiff + size_

        # This compatibility test is not standard.  In "regular" broadcasting,
        # two shapes are compatible if for each dimension, the lengths are the
        # same or one of the lengths is 1.  Here, the length of a dimension in
        # size_ must not be less than the corresponding length in bcast_shape.
        ok = all([bcdim == 1 or bcdim == szdim
                  for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError("size does not match the broadcast shape of "
                             "the parameters.")

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]

        return param_bcast, loc_bcast, scale_bcast, size_

    ## These are the methods you must define (standard form functions)
    ## NB: generic _pdf, _logpdf, _cdf are different for
    ## rv_continuous and rv_discrete hence are defined in there
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond

    def _support_mask(self, x):
        return (self.a <= x) & (x <= self.b)

    def _open_support_mask(self, x):
        return (self.a < x) & (x < self.b)

    def _rvs(self, *args):
        # This method must handle self._size being a tuple, and it must
        # properly broadcast *args and self._size.  self._size might be
        # an empty tuple, which means a scalar random variate is to be
        # generated.

        ## Use basic inverse cdf algorithm for RV generation as default.
        U = self._random_state.random_sample(self._size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0-self._cdf(x, *args)

    def _logsf(self, x, *args):
        return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0-q, *args)  # use correct _ppf for subclasses

    # These are actually called, and should not be overwritten if you
    # want to keep error checking.
    def rvs(self, *args, **kwds):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : None or int or ``np.random.RandomState`` instance, optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            raise ValueError("Domain error in arguments.")

        if np.all(scale == 0):
            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            self._random_state = check_random_state(rndm)

        # `size` should just be an argument to _rvs(), but for, um,
        # historical reasons, it is made an attribute that is read
        # by _rvs().
        self._size = size
        vals = self._rvs(*args)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        # if discrete:
        #     if size == ():
        #         vals = int(vals)
        #     else:
        #         vals = vals.astype(int)

        return vals

    def stats(self, *args, **kwds):
        """
        Some statistics of the given RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional (continuous RVs only)
            scale parameter (default=1)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis.
            (default is 'mv')

        Returns
        -------
        stats : sequence
            of requested moments.

        """
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        # scale = 1 by construction for discrete RVs
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = valarray(shape(cond), self.badvalue)

        # Use only entries that are valid in calculation
        if np.any(cond):
            goodargs = argsreduce(cond, *(args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]

            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs,
                                              **{'moments': moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)
            if g1 is None:
                mu3 = None
            else:
                if mu2 is None:
                    mu2 = self._munp(2, *goodargs)
                if g2 is None:
                    # (mu2**1.5) breaks down for nan and inf
                    mu3 = g1 * np.power(mu2, 1.5)

            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)

            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    mu2 = mu2p - mu * mu
                    if np.isinf(mu):
                        # if mean is inf then var is also inf
                        mu2 = np.inf
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)

            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = mu3p - 3 * mu * mu2 - mu**3
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)

            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = mu3p - 3 * mu * mu2 - mu**3
                    with np.errstate(invalid='ignore'):
                        mu4 = mu4p - 4 * mu * mu3 - 6 * mu * mu * mu2 - mu**4
                        g2 = mu4 / mu2**2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:  # no valid args
            output = []
            for _ in moments:
                out0 = default.copy()
                output.append(out0)

        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        """
        Differential entropy of the RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional  (continuous distributions only).
            Scale parameter (default=1).

        Notes
        -----
        Entropy is defined base `e`:

        >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))
        >>> np.allclose(drv.entropy(), np.log(2.0))
        True

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        # NB: for discrete distributions scale=1 by construction in _parse_args
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        place(output, (1-cond0), self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output

    def moment(self, n, *args, **kwds):
        """
        n-th order non-central moment of distribution.

        Parameters
        ----------
        n : int, n >= 1
            Order of moment.
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        if not (self._argcheck(*args) and (scale > 0)):
            return nan
        if (floor(n) != n):
            raise ValueError("Moment must be an integer.")
        if (n < 0):
            raise ValueError("Moment must be positive.")
        mu, mu2, g1, g2 = None, None, None, None
        if (n > 0) and (n < 5):
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'vk'}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*args, **mdict)
        val = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, args)

        # Convert to transformed  X = L + S*Y
        # E[X^n] = E[(L+S*Y)^n] = L^n sum(comb(n, k)*(S/L)^k E[Y^k], k=0...n)
        if loc == 0:
            return scale**n * val
        else:
            result = 0
            fac = float(scale) / float(loc)
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, args)
                result += comb(n, k, exact=True)*(fac**k) * valk
            result += fac**n * val
            return result * loc**n

    def median(self, *args, **kwds):
        """
        Median of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter, Default is 0.
        scale : array_like, optional
            Scale parameter, Default is 1.

        Returns
        -------
        median : float
            The median of the distribution.

        See Also
        --------
        stats.distributions.rv_discrete.ppf
            Inverse of the CDF

        """
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """
        Mean of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        mean : float
            the mean of the distribution

        """
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """
        Variance of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        var : float
            the variance of the distribution

        """
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """
        Standard deviation of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        std : float
            standard deviation of the distribution

        """
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, alpha, *args, **kwds):
        """
        Confidence interval with equal areas around the median.

        Parameters
        ----------
        alpha : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        """
        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1 = (1.0-alpha)/2
        q2 = (1.0+alpha)/2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return a, b


# Helpers for the discrete distributions
def _drv2_moment(self, n, *args):
    """Non-central moment of discrete distribution."""
    def fun(x):
        return np.power(x, n) * self._pmf(x, *args)
    return _expect(fun, self.a, self.b, self.ppf(0.5, *args), self.inc)


def _drv2_ppfsingle(self, q, *args):  # Use basic bisection algorithm
    b = self.b
    a = self.a
    if isinf(b):            # Be sure ending point is > q
        b = int(max(100*q, 10))
        while 1:
            if b >= self.b:
                qb = 1.0
                break
            qb = self._cdf(b, *args)
            if (qb < q):
                b += 10
            else:
                break
    else:
        qb = 1.0
    if isinf(a):    # be sure starting point < q
        a = int(min(-100*q, -10))
        while 1:
            if a <= self.a:
                qb = 0.0
                break
            qa = self._cdf(a, *args)
            if (qa > q):
                a -= 10
            else:
                break
    else:
        qa = self._cdf(a, *args)

    while 1:
        if (qa == q):
            return a
        if (qb == q):
            return b
        if b <= a+1:
            if qa > q:
                return a
            else:
                return b
        c = int((a+b)/2.0)
        qc = self._cdf(c, *args)
        if (qc < q):
            if a != c:
                a = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qa = qc
        elif (qc > q):
            if b != c:
                b = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qb = qc
        else:
            return c


def entropy(pk, qk=None, base=None):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=0)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=0)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).

    Returns
    -------
    S : float
        The calculated entropy.

    """
    pk = asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=0)
    if qk is None:
        vec = entr(pk)
    else:
        qk = asarray(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0*qk / np.sum(qk, axis=0)
        vec = rel_entr(pk, qk)
    S = np.sum(vec, axis=0)
    if base is not None:
        S /= log(base)
    return S


# Must over-ride one of _pmf or _cdf or pass in
#  x_k, p(x_k) lists in initialization

class rv_discrete(rv_generic):

    def __new__(cls, a=0, b=inf, name=None, badvalue=None,
                moment_tol=1e-8, values=None, inc=1, longname=None,
                shapes=None, extradoc=None, seed=None):

        if values is not None:
            # dispatch to a subclass
            return super(rv_discrete, cls).__new__(rv_sample)
        else:
            # business as usual
            return super(rv_discrete, cls).__new__(cls)

    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super(rv_discrete, self).__init__(seed)

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self.vecentropy = vectorize(self._entropy)
        self.shapes = shapes

        if values is not None:
            raise ValueError("rv_discrete.__init__(..., values != None, ...)")

        self._construct_argparser(meths_to_inspect=[self._pmf, self._cdf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')

        # nin correction needs to be after we know numargs
        # correct nin for generic moment vectorization
        _vec_generic_moment = vectorize(_drv2_moment, otypes='d')
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = instancemethod(_vec_generic_moment,
                                             self, rv_discrete)

        # correct nin for ppf vectorization
        _vppf = vectorize(_drv2_ppfsingle, otypes='d')
        _vppf.nin = self.numargs + 2
        self._ppfvec = instancemethod(_vppf,
                                      self, rv_discrete)

        # now that self.numargs is defined, we can adjust nin
        self._cdfvec.nin = self.numargs + 1

        self._construct_docstrings(name, longname, extradoc)

    def _construct_docstrings(self, name, longname, extradoc):
        if name is None:
            name = 'Distribution'
        self.name = name
        self.extradoc = extradoc

        # generate docstring for subclass instances
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # Skip adding docstrings if interpreter is run with -OO
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            extradoc=extradoc,
                                            docdict=docdict_discrete,
                                            discrete='discrete')
            else:
                dct = dict(distdiscrete)
                self._construct_doc(docdict_discrete, dct.get(self.name))

            # discrete RV do not have the scale parameter, remove it
            self.__doc__ = self.__doc__.replace(
                '\n    scale : array_like, '
                'optional\n        scale parameter (default=1)', '')

    def _updated_ctor_param(self):
        """ Return the current version of _ctor_param, possibly updated by user.

            Used by freezing and pickling.
            Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['badvalue'] = self.badvalue
        dct['moment_tol'] = self.moment_tol
        dct['inc'] = self.inc
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        dct['extradoc'] = self.extradoc
        return dct

    def _nonzero(self, k, *args):
        return floor(k) == k

    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k-1, *args)

    def _logpmf(self, k, *args):
        return log(self._pmf(k, *args))

    def _cdf_single(self, k, *args):
        m = arange(int(self.a), k+1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        k = floor(x)
        return self._cdfvec(k, *args)

    # generic _logcdf, _sf, _logsf, _ppf, _isf, _rvs defined in rv_generic

    def rvs(self, *args, **kwargs):
        """
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        size : int or tuple of ints, optional
            Defining number of random variates (Default is 1).  Note that `size`
            has to be given as keyword, not as positional argument.
        random_state : None or int or ``np.random.RandomState`` instance, optional
            If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        kwargs['discrete'] = True
        return super(rv_discrete, self).rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        """
        Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k <= self.b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """
        Log of the probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter. Default is 0.

        Returns
        -------
        logpmf : array_like
            Log of the probability mass function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k <= self.b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(NINF)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k < self.b)
        cond2 = (k >= self.b)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2*(cond0 == cond0), 1.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        """
        Log of the cumulative distribution function at k of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k < self.b)
        cond2 = (k >= self.b)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(NINF)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2*(cond0 == cond0), 0.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        """
        Survival function (1 - `cdf`) at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        sf : array_like
            Survival function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k < self.b)
        cond2 = (k < self.a) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        """
        Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= self.a) & (k < self.b)
        cond2 = (k < self.a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(NINF)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """
        Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Lower tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1
        output = valarray(shape(cond), value=self.badvalue, typecode='d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), self.a-1)
        place(output, cond2, self.b)
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._ppf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """
        Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1

        # same problem as with ppf; copied from ppf and changed
        output = valarray(shape(cond), value=self.badvalue, typecode='d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), self.b)
        place(output, cond2, self.a-1)

        # call place only if at least 1 valid argument
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            # PB same as ticket 766
            place(output, cond, self._isf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def _entropy(self, *args):
        if hasattr(self, 'pk'):
            return entropy(self.pk)
        else:
            return _expect(lambda x: entr(self.pmf(x, *args)),
                           self.a, self.b, self.ppf(0.5, *args), self.inc)

    def expect(self, func=None, args=(), loc=0, lb=None, ub=None,
               conditional=False, maxcount=1000, tolerance=1e-10, chunksize=32):
        """
        Calculate expected value of a function with respect to the distribution
        for discrete distribution by numerical summation.

        Parameters
        ----------
        func : callable, optional
            Function for which the expectation value is calculated.
            Takes only one argument.
            The default is the identity mapping f(k) = k.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter.
            Default is 0.
        lb, ub : int, optional
            Lower and upper bound for the summation, default is set to the
            support of the distribution, inclusive (``ul <= k <= ub``).
        conditional : bool, optional
            If true then the expectation is corrected by the conditional
            probability of the summation interval. The return value is the
            expectation of the function, `func`, conditional on being in
            the given interval (k such that ``ul <= k <= ub``).
            Default is False.
        maxcount : int, optional
            Maximal number of terms to evaluate (to avoid an endless loop for
            an infinite sum). Default is 1000.
        tolerance : float, optional
            Absolute tolerance for the summation. Default is 1e-10.
        chunksize : int, optional
            Iterate over the support of a distributions in chunks of this size.
            Default is 32.

        Returns
        -------
        expect : float
            Expected value.

        Notes
        -----
        For heavy-tailed distributions, the expected value may or may not exist,
        depending on the function, `func`. If it does exist, but the sum converges
        slowly, the accuracy of the result may be rather low. For instance, for
        ``zipf(4)``, accuracy for mean, variance in example is only 1e-5.
        increasing `maxcount` and/or `chunksize` may improve the result, but may
        also make zipf very slow.

        The function is not vectorized.

        """
        if func is None:
            def fun(x):
                # loc and args from outer scope
                return (x+loc)*self._pmf(x, *args)
        else:
            def fun(x):
                # loc and args from outer scope
                return func(x+loc)*self._pmf(x, *args)
        # used pmf because _pmf does not check support in randint and there
        # might be problems(?) with correct self.a, self.b at this stage maybe
        # not anymore, seems to work now with _pmf

        self._argcheck(*args)  # (re)generate scalar self.a and self.b
        if lb is None:
            lb = self.a
        else:
            lb = lb - loc   # convert bound for standardized distribution
        if ub is None:
            ub = self.b
        else:
            ub = ub - loc   # convert bound for standardized distribution
        if conditional:
            invfac = self.sf(lb-1, *args) - self.sf(ub, *args)
        else:
            invfac = 1.0

        # iterate over the support, starting from the median
        x0 = self.ppf(0.5, *args)
        res = _expect(fun, lb, ub, x0, self.inc, maxcount, tolerance, chunksize)
        return res / invfac


def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10,
            chunksize=32):
    """Helper for computing the expectation value of `fun`."""

    # short-circuit if the support size is small enough
    if (ub - lb) <= chunksize:
        supp = np.arange(lb, ub+1, inc)
        vals = fun(supp)
        return np.sum(vals)

    # otherwise, iterate starting from x0
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub

    count, tot = 0, 0.
    # iterate over [x0, ub] inclusive
    for x in _iter_chunked(x0, ub+1, chunksize=chunksize, inc=inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge', RuntimeWarning)
            return tot

    # iterate over [lb, x0)
    for x in _iter_chunked(x0-1, lb-1, chunksize=chunksize, inc=-inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge', RuntimeWarning)
            break

    return tot


def _iter_chunked(x0, x1, chunksize=4, inc=1):
    """Iterate from x0 to x1 in chunks of chunksize and steps inc.

    x0 must be finite, x1 need not be. In the latter case, the iterator is
    infinite.
    Handles both x0 < x1 and x0 > x1. In the latter case, iterates downwards
    (make sure to set inc < 0.)

    >>> [x for x in _iter_chunked(2, 5, inc=2)]
    [array([2, 4])]
    >>> [x for x in _iter_chunked(2, 11, inc=2)]
    [array([2, 4, 6, 8]), array([10])]
    >>> [x for x in _iter_chunked(2, -5, inc=-2)]
    [array([ 2,  0, -2, -4])]
    >>> [x for x in _iter_chunked(2, -9, inc=-2)]
    [array([ 2,  0, -2, -4]), array([-6, -8])]

    """
    if inc == 0:
        raise ValueError('Cannot increment by zero.')
    if chunksize <= 0:
        raise ValueError('Chunk size must be positive; got %s.' % chunksize)

    s = 1 if inc > 0 else -1
    stepsize = abs(chunksize * inc)

    x = x0
    while (x - x1) * inc < 0:
        delta = min(stepsize, abs(x - x1))
        step = delta * s
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp


class rv_sample(rv_discrete):
    """A 'sample' discrete distribution defined by the support and values.

       The ctor ignores most of the arguments, only needs the `values` argument.
    """
    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super(rv_discrete, self).__init__(seed)

        if values is None:
            raise ValueError("rv_sample.__init__(..., values=None,...)")

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy

        xk, pk = values

        if len(xk) != len(pk):
            raise ValueError("xk and pk need to have the same length.")
        if not np.allclose(np.sum(pk), 1):
            raise ValueError("The sum of provided pk is not 1.")

        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]
        self.qvals = np.cumsum(self.pk, axis=0)

        self.shapes = ' '   # bypass inspection
        self._construct_argparser(meths_to_inspect=[self._pmf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')

        self._construct_docstrings(name, longname, extradoc)

    def _pmf(self, x):
        return np.select([x == k for k in self.xk],
                         [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)

    def _cdf(self, x):
        xx, xxk = np.broadcast_arrays(x[:, None], self.xk)
        indx = np.argmax(xxk > xx, axis=-1) - 1
        return self.qvals[indx]

    def _ppf(self, q):
        qq, sqq = np.broadcast_arrays(q[..., None], self.qvals)
        indx = argmax(sqq >= qq, axis=-1)
        return self.xk[indx]

    def _rvs(self):
        # Need to define it explicitly, otherwise .rvs() with size=None
        # fails due to explicit broadcasting in _ppf
        U = self._random_state.random_sample(self._size)
        if self._size is None:
            U = np.array(U, ndmin=1)
            Y = self._ppf(U)[0]
        else:
            Y = self._ppf(U)
        return Y

    def _entropy(self):
        return entropy(self.pk)

    def generic_moment(self, n):
        n = asarray(n)
        return np.sum(self.xk**n[np.newaxis, ...] * self.pk, axis=0)


def get_distribution_names(namespace_pairs, rv_base_class):
    """
    Collect names of statistical distributions and their generators.

    Parameters
    ----------
    namespace_pairs : sequence
        A snapshot of (name, value) pairs in the namespace of a module.
    rv_base_class : class
        The base class of random variable generator classes in a module.

    Returns
    -------
    distn_names : list of strings
        Names of the statistical distributions.
    distn_gen_names : list of strings
        Names of the generators of the statistical distributions.
        Note that these are not simply the names of the statistical
        distributions, with a _gen suffix added.

    """
    distn_names = []
    distn_gen_names = []
    for name, value in namespace_pairs:
        if name.startswith('_'):
            continue
        if name.endswith('_gen') and issubclass(value, rv_base_class):
            distn_gen_names.append(name)
        if isinstance(value, rv_base_class):
            distn_names.append(name)
    return distn_names, distn_gen_names



if __name__ == "__main__":
    pass
