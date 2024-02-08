import os
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.2.2'

import numpy
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import IntVector

pandas2ri.activate()
R_STATS = importr('stats')
R_DESCTOOLS = importr("pscl")

def calculate_fisher_exact_test_R(flatten_contintingency_table):
    v = robjects.IntVector(flatten_contintingency_table)
    m = robjects.r['matrix'](v, nrow = 2)
    assert m.shape == (2, 2)

    result = R_STATS.fisher_test(m)
    pvalue = result[0][0]
    odds_ci = result[1]
    odds = result[2][0]

    return pvalue, odds, odds_ci

def calculate_simulated_chisq(contingency_table, simulate_p_value=True):
    v1 = robjects.IntVector( contingency_table[0] )
    v2 = robjects.IntVector( contingency_table[1] )
    m = robjects.r.matrix(v1+v2, ncol=2).transpose()

    result = R_STATS.chisq_test(m, simulate_p_value=simulate_p_value, B=10000)
    chi2 = result[0][0]
    df = result[1][0]
    pvalue = result[2][0]

    k = len(contingency_table[0]) - 1
    A = ",".join( [ str(n) for n in contingency_table[0] ] )
    B = ",".join( [ str(n) for n in contingency_table[1] ] )
    code = f'pR2(glm(matrix(c({A},{B}),ncol=2)~c(0:{k}), family = binomial()))["McFadden"]'
    mcfadden = robjects.r(code)[0]

    return chi2, df, pvalue, mcfadden

def calculate_simulated_chisq_two(count_a, count_b, simulate_p_value=True):
    v1 = robjects.IntVector( [count_a, count_b] )
    result = R_STATS.chisq_test(v1, simulate_p_value=simulate_p_value, B=10000)
    chi2 = result[0][0]
    df = result[1][0]
    pvalue = result[2][0]
    phi = numpy.sqrt(chi2 / (count_a+count_b) )

    return count_a, count_b, chi2, df, pvalue, phi



def test():
    print( calculate_fisher_exact_test_R([60, 10, 30, 25]) )
    print( calculate_simulated_chisq( [[60, 10, 30, 25], [50, 11, 32, 24]] ) )
    print( calculate_simulated_chisq_two(1000, 900))

#test()
