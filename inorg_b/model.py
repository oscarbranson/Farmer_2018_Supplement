import numpy as np
from .helpers import A11_2_d11

def rS_calc(Rb, Rp, Kf, rL, Kb):
    """
    Calculate rS from input parameters.
    """
    Rf = Rp + Rb
    
    return (Kf * rL * Rf) / (Rb * Kb + Rp)

def predfn(Kb3, Kf3, Kb4, Kf4, logRb, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4):
    """
    Predict B partitioning and offset from d11B BOH4.

    Returns
    -------
    LambdaB, EpsilonB
    """
    Rb = 10**logRb
    
    # Partitioning Calculations
    rS4 = rS_calc(Rb, Rp, Kf4, rL4, Kb4)
    rS3 = rS_calc(Rb, Rp, Kf3, rL3, Kb3)

    rSB = rS3 + rS4  # total B/Ca of solid
    
    KB = rSB / B_DIC  # calculate lambda partitioning

    # Isotope Calculations
    # Abundance of 11B in calcite via mixing calculation
    ABcal = (ABO3 * rS3 + ABO4 * rS4) / rSB
    DdBcal = A11_2_d11(ABcal) - dBO4  # convert back to d11B
    
    return KB, DdBcal

def fitfn(p, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4, LambdaB, EpsilonB,
          LambdaB_err=1, EpsilonB_err=1, LambdaB_bias=None):
    """
    Function to minimise when fitting the model.

    Parameters
    ----------
    p : tuple
        Parameters to fit, in the order (Kb3, Kf3, Kb4, Kf4, logRb).
    """
    if LambdaB_bias is None:
        LambdaB_bias = np.ptp(EpsilonB) / np.ptp(LambdaB)

    LambdaB_calc, EpsilonB_calc = predfn(*p, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4)

    Lam_err = LambdaB_bias * np.sum((LambdaB_calc - LambdaB)**2 / (LambdaB_err**2))
    Eps_err = np.sum((EpsilonB_calc - EpsilonB)**2 / (EpsilonB_err**2))
    return Lam_err / 2 + Eps_err / 2

    # Lam_err = -0.5 * np.sum((LambdaB_calc - LambdaB)**2 / (LambdaB_err**2) + np.log(2 * np.pi * LambdaB_err**2))
    # Eps_err = -0.5 * np.sum((EpsilonB_calc - EpsilonB)**2 / (EpsilonB_err**2) + np.log(2 * np.pi * LambdaB_err**2))

    # return -(LambdaB_bias * Lam_err + Eps_err)