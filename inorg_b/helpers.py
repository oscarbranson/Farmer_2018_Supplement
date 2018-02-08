from pandas import IndexSlice as idx
import uncertainties.unumpy as up
import numpy as np

def err(x):
    return up.std_devs(x)

def nom(x):
    return up.nominal_values(x)

def sol_B_iso(BT, BO4, d11B_total, alpha=1.026):
    """
    Calculate d11B of aqueous species.

    Returns
    -------
    d11BO4, d11BO3
    """
    eps = 1000 * (alpha - 1)
    BO3 = BT - BO4

    d11BO4 = (d11B_total * BT - eps * BO3) / (BO4 + alpha * BO3)
    d11BO3 = d11BO4 + eps

    return d11BO4, d11BO3

def sol_B_iso_Rae2018(pH, BO4, BT, alpha, d11BT):
    """
    Calculate d11B of aqueous species using full mass-balance approach of Rae, 2018.
    
    Returns
    -------
    d11BO4, d11BO3
    """

    R_BT = d11_2_R11(d11BT)
    Hval = 10**-pH
    Kbval = BO4 * Hval / (BT - BO4)
    RB4 = (((Hval**2*R_BT**2 + 2*Hval**2*R_BT*alpha + Hval**2*alpha**2 + 2*Hval*Kbval*R_BT**2*alpha - 
             2*Hval*Kbval*R_BT*alpha**2 + 8*Hval*Kbval*R_BT*alpha - 2*Hval*Kbval*R_BT + 2*Hval*Kbval*alpha + 
             Kbval**2*R_BT**2*alpha**2 + 2*Kbval**2*R_BT*alpha + Kbval**2)**(1/2) - 
            Hval*alpha - Kbval + Hval*R_BT + Kbval*R_BT*alpha)/(2*alpha*(Hval + Kbval)))
    RB3 = RB4 * alpha

    # return RB4, RB3
    return R11_2_d11(RB4), R11_2_d11(RB3)

# Unit Converters
def d11_2_A11(d11, SRM_ratio=4.04367):
    """
    Convert Delta to Abundance notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return SRM_ratio * (d11 / 1000 + 1) / (SRM_ratio * (d11 / 1000 + 1) + 1)

def A11_2_d11(A11, SRM_ratio=4.04367):
    """
    Convert Abundance to Delta notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return ((A11 / (1 - A11)) / SRM_ratio - 1) * 1000

def A11_2_R11(A11):
    """
    Convert Abundance to isotope ratio notation.
    """
    return A11 / (1 - A11)

def d11_2_R11(d11, SRM_ratio=4.04367):
    """
    Convert Delta to isotope ratio notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return (d11/1000 + 1) * SRM_ratio


def R11_2_d11(R11, SRM_ratio=4.04367):
    """
    Convert isotope ratio to Delta notation.

    Default SRM_ratio is NIST951 11B/10B
    """
    return (R11 / SRM_ratio - 1) * 1000


def R11_2_A11(R11):
    """
    Convert isotope ratio to abundance notation.
    """
    return R11 / (1 + R11)



def extract_model_vars(rd, exp='Uchikawa', Rvar=('Solid', 'logR'), phase='Calcite'):
    """
    Returns
    -------
    (logRp, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4, LambdaB, EpsilonB, 
     LambdaB_err, EpsilonB_err, LambdaB_err_norm, EpsilonB_err_norm)
    """
    # prepare fitting variables
    cind = idx[:, exp, phase]

    # Precipitation Rate
    if 'log' in Rvar[-1]:
        logRp = (rd.loc[cind, Rvar]).values
        Rp = 10**logRp
    else:
        Rp = (rd.loc[cind, Rvar]).values
        logRp = np.log10(Rp)

    # Solution BO3/C and BO4/CO3 ratios
    rL3 = (rd.loc[cind, ('pitzer', 'BOH3')] / rd.loc[cind, ('pitzer', 'C')]).astype(float).values
    rL4 = (rd.loc[cind, ('pitzer', 'BOH4_free')] / rd.loc[cind, ('pitzer', 'CO3')]).astype(float).values
    # B/DIC for LambdaB calculation
    B_DIC = (rd.loc[cind, ('pitzer', 'B')] / rd.loc[cind, ('pitzer', 'C')]).astype(float).values
    # Isotopic content of each B species
    ABO3 = d11_2_A11(rd.loc[cind, ('Solution', 'd11BO3 (permil vs NIST951)')].astype(float).values)
    ABO4 = d11_2_A11(rd.loc[cind, ('Solution', 'd11BO4 (permil vs NIST951)')].astype(float).values)
    # Borate d11B for EpsilonB calculation
    dBO4 = rd.loc[cind, ('Solution', 'd11BO4 (permil vs NIST951)')].astype(float).values

    # Measured LambdaB and EpsilonB fo residual calculation
    LambdaB = rd.loc[cind, ('Solid', 'LambdaB')].astype(float).values
    EpsilonB = rd.loc[cind, ('Solid', 'EpsilonB')].astype(float).values

    # Uncertainties on the measured variables
    LambdaB_err = err(rd.loc[cind, ('Solid', 'LambdaB_eprop')].values)
    EpsilonB_err = err(rd.loc[cind, ('Solid', 'EpsilonB_eprop')].values)

    # normalised to their mean, to make them comparable
    LambdaB_err_norm = (LambdaB_err / LambdaB_err.mean())**0.5
    EpsilonB_err_norm = (EpsilonB_err / EpsilonB_err.mean())**0.5

    return (logRp, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4, LambdaB, EpsilonB, 
            LambdaB_err, EpsilonB_err, LambdaB_err_norm, EpsilonB_err_norm)
