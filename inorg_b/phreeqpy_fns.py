"""
Functions for calculating the speciation of C and B in precipitation solutions.
"""

import os
import pandas as pd
import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod


def input_str(temp=25, pH=8.1, Na=0, Cl=0, K=0, B=0, Ca=0, DIC=0, Mg=0, SO4=0, units='mol/L'):
    """
    Generate phreeqc input string for calculating C and B chemistry of solution.
    """
    template = """SOLUTION 1
        temp      {temp:.3f}
        pH        {pH:.3f}
        pe        4
        redox     pe
        units     {units:s}
        density   1
        Cl        {Cl:.6f}
        Na        {Na:.6f}
        Mg        {Mg:.6f}
        B         {B:.6f}
        Ca        {Ca:.6f}
        C         {DIC:.6f}
        K         {K:.6f}
        S(6)      {SO4:.6f}
        -water    1 # kg

    SELECTED_OUTPUT
        -pH
        -temperature
        -alkalinity
        -ionic_strength
        -totals Cl Na Mg K B Ca C S(6)
        -m OH- H+
        # minteq4 outputs
        -m H3BO3 H2BO3- NaH2BO3 CaH2BO3+ H5(BO3)2- H8(BO3)3-  # boron
        -m HCO3- NaHCO3 NaCO3- CO3-2 H2CO3 CaCO3 CaHCO3+  # carbon
        # pitzer outputs
        -m B(OH)4- B(OH)3 CaB(OH)4+ B3O3(OH)4- B4O5(OH)4-2  # boron
        -m HCO3- CO3-2 CO2  # carbon
        -si Calcite Aragonite
    END
    """

    return template.format(temp=temp,
                           pH=pH,
                           Na=Na,
                           Cl=Cl,
                           B=B,
                           Ca=Ca,
                           DIC=DIC,
                           Mg=Mg,
                           SO4=SO4,
                           units=units,
                           K=K)

def run_phreeqc(input_string, dbase_path, phreeq_path='/usr/local/lib/libiphreeqc.so'):
    """
    Run input string in phreeqc with specified database.

    Parameters
    ----------
    input_string : str
        Valid phreeqc input string with SELECTED_OUTPUT.
    dbase_path : str
        Path to valid phreeqc database (e.g. minteq.v4.dat)
    phreeq_path : str
        Path to iphreeqc shared library.

    Returns
    -------
    pandas.Series of calculated species
    """
    phreeqc = phreeqc_mod.IPhreeqc(phreeq_path)
    phreeqc.load_database(dbase_path)
    phreeqc.run_string(input_string)
    out = phreeqc.get_selected_output_array()
    return pd.Series(out[1], out[0])

# function to calculate solution C and B speciation
def calc_cb(temp=25, pH=8.1, Na=0, Cl=0, K=0, B=0, Ca=0, DIC=0, Mg=0, SO4=0, dbase='pitzer', database_path=None, summ=True, phreeq_path='/usr/local/lib/libiphreeqc.so'):  
    """
    Calculate carbon and boron chemistry of solution.

    Parameters
    ----------
    temp, pH, Na, Cl, B, Ca, DIC, Mg : float
        Solution characteristics. All concentrations in mol/L.
    dbase : str
        Name of phreeqc database to use.
    summ : boolean
        If true, returns summary data, with column names tweaked to accommodate
        different B species returned by pitzer / minteq.v4 databases.

    Returns
    -------
    pandas.Series of results.
    """
    # path to phreeqc database files
    if database_path is None:
        database_path = "/home/oscar/phreeqc/iphreeqc-3.3.9-11951/database/"
    # create input string
    inp = input_str(temp, pH, Na, Cl, K, B, Ca, DIC, Mg, SO4)
    # run phreeqc
    dat = run_phreeqc(inp, os.path.join(database_path, dbase + '.dat'), phreeq_path=phreeq_path)

    if summ:
        # return C and B chemistry (simple ions only)
        out = pd.Series(index=['C', 'CO2', 'HCO3', 'CO3', 'B', 'BOH3', 'BOH4'])
        # carbon species
        out['C'] = dat['C(mol/kgw)']
        if dat['m_CO2(mol/kgw)'] != 0:
            out['CO2'] = dat['m_CO2(mol/kgw)']
        else:
            out['CO2'] = dat['m_H2CO3(mol/kgw)']

        out['HCO3'] = dat['m_HCO3-(mol/kgw)']
        out['CO3'] = dat['m_CO3-2(mol/kgw)']
        # boron species
        out['B'] = dat['B(mol/kgw)']
        if dat['m_B(OH)3(mol/kgw)'] != 0:
            out['BOH3'] = dat['m_B(OH)3(mol/kgw)']
        else:
            out['BOH3'] = dat['m_H3BO3(mol/kgw)']
        if dat['m_B(OH)4-(mol/kgw)'] != 0:
            out['BOH4'] = dat['m_B(OH)4-(mol/kgw)'] + dat['m_CaB(OH)4+(mol/kgw)']
            out['BOH4_free'] = dat['m_B(OH)4-(mol/kgw)']
        else:
            out['BOH4'] = dat['m_H2BO3-(mol/kgw)'] + dat['m_NaH2BO3(mol/kgw)']  + dat['m_CaH2BO3+(mol/kgw)']
            out['BOH4_free'] = dat['m_H2BO3-(mol/kgw)']

        # auxillary data
        out['pH'] = dat['pH']
        out['temp'] = dat['temp(C)']
        out['alk'] = dat['Alk(eq/kgw)']
        out['SIc'] = dat['si_Calcite']
        out['SIa'] = dat['si_Aragonite']
        out['ion_str'] = dat['mu']

        out['Ca'] = dat['Ca(mol/kgw)']
        out['Na'] = dat['Na(mol/kgw)']
        out['Cl'] = dat['Cl(mol/kgw)']
        out['Mg'] = dat['Mg(mol/kgw)']
        out['K'] = dat['K(mol/kgw)']
        out['SO4'] = dat['S(6)(mol/kgw)']

        return out
    else:
        return dat


def calc_cb_rows(df, dbase='pitzer', phreeq_path='/usr/local/lib/libiphreeqc.so', database_path=None):
    """
    Calculate solution conditions for each row of solution data.

    Parameters
    ----------
    df : pandas.DataFrame
        Each row must contain ['Temp (°C)', 'pH (NBS)', '[Na M)',
        '[Cl M)', '[Ca M)', '[B M)', '[DIC M)', '[Mg M)]
    
    Returns
    -------
    pandas.Dataframe with same index as input, with calculated C and B chemistry.
    """
    # determine column names
    r = df.iloc[0,:]
    cols = calc_cb(temp=r['Temp (°C)'],
                   pH=r['pH (NBS)'],
                   Na=r['[Na] (M)'], 
                   Cl=r['[Cl] (M)'], 
                   Ca=r['[Ca] (M)'], 
                   B=r['[B] (M)'], 
                   DIC=r['[DIC] (M)'], 
                   Mg=r['[Mg] (M)'],
                   dbase=dbase,
                   phreeq_path=phreeq_path,
                   database_path=database_path).index

    # create empty output dataframe
    out = pd.DataFrame(index=df.index, columns=cols)
    out.sort_index(1, inplace=True)
    out.sort_index(0, inplace=True)
    
    for i, r in df.iterrows():
        out.loc[i, :] = calc_cb(temp=r['Temp (°C)'],
                                pH=r['pH (NBS)'],
                                Na=r['[Na] (M)'], 
                                Cl=r['[Cl] (M)'], 
                                Ca=r['[Ca] (M)'], 
                                B=r['[B] (M)'], 
                                DIC=r['[DIC] (M)'], 
                                Mg=r['[Mg] (M)'],
                                dbase=dbase,
                                phreeq_path=phreeq_path,
                                database_path=database_path)
    return out
