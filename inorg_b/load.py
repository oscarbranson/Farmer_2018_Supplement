import pandas as pd
from .helpers import sol_B_iso_Rae2018
from .phreeqpy_fns import calc_cb_rows

import re
import uncertainties.unumpy as un


def raw_data():
    """
    Load raw data from google sheet.
    """
    raw_data_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRo_UMyhlIlOpdYffTQMFlySTs8v1lnr1EZpsQBHATWrRBrNJG8CnCnGKJJbcrC6Vj7L9k3_Fy4WmT8/pub?gid=0&single=true&output=csv'
    rd = pd.read_csv(raw_data_url, header=[0, 1], index_col=[0, 1, 2])

    rd.sort_index(0, inplace=True)
    rd.sort_index(1, inplace=True)

    return rd

def package_errors(rd):
    """
    Pack measurements with their associated uncertainties as uarrays.
    """
    # solid
    errcs = [c for c in rd.Solid.columns if 'std' in c]

    for ecol in errcs:
        pcol = re.sub('_2?std', '', ecol)
        ncol = re.sub('_2?std', '_eprop', ecol)

        rd.loc[:, ('Solid', ncol)] = un.uarray(rd.loc[:, ('Solid', pcol)],
                                               rd.loc[:, ('Solid', ecol)])

    # solution
    errcs = [c for c in rd.Solution.columns if 'std' in c]

    for ecol in errcs:
        pcol = re.sub('_2?std', '', ecol)
        ncol = re.sub('_2?std', '_eprop', ecol)

        rd.loc[:, ('Solution', ncol)] = un.uarray(rd.loc[:, ('Solution', pcol)],
                                               rd.loc[:, ('Solution', ecol)])
# Implement package_errors in all the above.

def calc_phreeqc(rd, database='pitzer'):
    """
    Calculate C and B speciation using phreeqc

    Parameters
    ----------
    rd : pandas.DataFrame
        Produced by raw_data() function.

    Returns
    -------
    pandas.DataFrame
    """
    sol = calc_cb_rows(rd.Solution)
    sol.columns = pd.MultiIndex.from_product([[database], sol.columns])

    rd = rd.join(sol)
    return rd

def calc_lambda(rd, numerator='B', denom='C', database='pitzer'):
    """
    Calculate lambdaB (B/Ca) / ([B]/[DIC]).

    Parameters
    ----------
    rd : pandas.DataFrame
        Produced by raw_data() function
    numerator, denominator : str
        The numerator and denominator used for soluton C and B species when
        calculating the lamda: lambdaB = [B/Ca] / (numerator / denom)
        Where each must specify a valid column name produced after phreeqc solution
        chemistry calculation.
        For numerator, 'B'=[B], 'BOH3'=[B(OH)3], 'BOH4'=[B(OH)4-]_{total}, 'BOH4_free'=[B(OH)4-]_{free}
        For denom, 'C'=[DIC], 'HCO3'=[HCO3-] and 'CO3'=[CO32-]
    database : str
        The name of the database used to calculate solution C and B speciation.

    Returns
    -------
        None
    """
    rd.loc[:, ('Solid', 'LambdaB')] = ((1e-3 * rd.loc[:, ('Solid', 'B/Ca (umol/mol)')]) /
                                       (rd.loc[:, (database, numerator)] /
                                        rd.loc[:, (database, denom)]))
    if 'B/Ca_eprop (umol/mol)' in rd.loc[:, 'Solid']:
        rd.loc[:, ('Solid', 'LambdaB_eprop')] = ((1e-3 * rd.loc[:, ('Solid', 'B/Ca_eprop (umol/mol)')]) /
                                                 (rd.loc[:, (database, numerator)] /
                                                  rd.loc[:, (database, denom)]))

def calc_sol_iso(rd, database='pitzer', borate_mode='total', alpha=1.026):
    """
    Calculate isotopic content of solution BO3 and BO4.

    Must be run after calc_phreeqc.

    Parameters
    ----------
    rd : pandas.DataFrame
        created by raw_data()
    database : str
        The name of the database used to calculate solution C and B speciation.
    borate_mode : str
        Whether to use 'free' or 'total' borate when calculating B isotope fractionation.
        Should probably be 'total', but it can be interesting to see what 'free' does.

    Returns
    -------
    """
    if borate_mode == 'free':
        BO4_mode = 'BOH4_free'
    else:
        BO4_mode = 'BOH4'

    d11BO4, d11BO3 = sol_B_iso_Rae2018(pH=rd.loc[:, (database, 'pH')],
                                       BT=rd.loc[:, (database, 'B')],
                                       BO4=rd.loc[:, (database, BO4_mode)],
                                       d11BT=rd.loc[:, ('Solution', 'd11B (permil vs NIST951)')], alpha=alpha)
    rd.loc[:, ('Solution', 'd11BO3 (permil vs NIST951)')] = d11BO3
    rd.loc[:, ('Solution', 'd11BO4 (permil vs NIST951)')] = d11BO4

    if 'd11B_eprop (permil vs NIST951)' in rd.loc[:, 'Solution']:
        d11BO4, d11BO3 = sol_B_iso_Rae2018(pH=rd.loc[:, (database, 'pH')],
                                           BT=rd.loc[:, (database, 'B')],
                                           BO4=rd.loc[:, (database, BO4_mode)],
                                           d11BT=rd.loc[:, ('Solution', 'd11B_eprop (permil vs NIST951)')], alpha=alpha)
    rd.loc[:, ('Solution', 'd11BO3_eprop (permil vs NIST951)')] = d11BO3
    rd.loc[:, ('Solution', 'd11BO4_eprop (permil vs NIST951)')] = d11BO4

def calc_epsilon(rd):
    """
    Calcilate epsilon (d11B - d11BO4).

    Must be preceded by calc_sol_iso.

    Parameters
    ----------
    rd : pandas.DataFrame
        created by raw_data()
    Returns
    -------
    """
    rd.loc[:, ('Solid', 'EpsilonB')] = (rd.loc[:, ('Solid', 'd11B (permil vs NIST951)')] -
                                        rd.loc[:, ('Solution', 'd11BO4 (permil vs NIST951)')])
    if 'd11B_eprop (permil vs NIST951)' in rd.loc[:, 'Solid'] and 'd11BO4_eprop (permil vs NIST951)' in rd.loc[:, 'Solution']:
        rd.loc[:, ('Solid', 'EpsilonB_eprop')] = (rd.loc[:, ('Solid', 'd11B_eprop (permil vs NIST951)')] -
                                                  rd.loc[:, ('Solution', 'd11BO4_eprop (permil vs NIST951)')])

def processed(database='pitzer', lambda_num='B', lambda_denom='C', borate_mode='total', alpha_sol=1.026):
    """
    Load and process all data.

    Parameters
    ----------
    database : str
        Which database phreeqc should use.
    lambda_num, lambda_denom : str
        The numerator and denominator used for soluton C and B species when
        calculating the lamda: lambdaB = [B/Ca] / (lambda_num / lambda_denom)
        Where each must specify a valid column name produced after phreeqc solution
        chemistry calculation.
        For lambda_num, 'B'=[B], 'BOH3'=[B(OH)3], 'BOH4'=[B(OH)4-]_{total}, 'BOH4_free'=[B(OH)4-]_{free}
        For lambda denom, 'C'=[DIC], 'HCO3'=[HCO3-] and 'CO3'=[CO32-]
    borate_mode : str
        Whether to use 'free' or 'total' borate when calculating B isotope fractionation.
        Should probably be 'total', but it can be interesting to see what 'free' does.

    Returns
    -------
    pandas.DataFrame
    """
    rd = raw_data()
    package_errors(rd)
    rd = calc_phreeqc(rd, database)
    calc_lambda(rd, database=database, numerator=lambda_num, denom=lambda_denom)
    calc_sol_iso(rd, database, borate_mode=borate_mode, alpha=alpha_sol)
    calc_epsilon(rd)

    rd.sort_index(0, inplace=True)
    rd.sort_index(1, inplace=True)

    return rd