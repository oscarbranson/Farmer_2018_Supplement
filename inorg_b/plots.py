import numpy as np
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
from .model import predfn
from .helpers import extract_model_vars, err, nom

def model_vs_data(params, rd, param_CIs=None, exp='Uchikawa'):

    (logRp, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4, LambdaB, EpsilonB, 
     LambdaB_err, EpsilonB_err, LambdaB_err_norm, EpsilonB_err_norm) = extract_model_vars(rd, exp)
    
    if param_CIs is None:
        param_CIs = np.full(params.shape, None)

    cind = idx[:, exp, :]

    LambdaB_pred, EpsilonB_pred = predfn(*params, Rp, rL3, rL4, B_DIC, ABO3, ABO4, dBO4)
    
    fig, axs = plt.subplots(2, 2, figsize=[6, 4], sharex=True)
    ((ax1, rx1), (ax2, rx2)) = axs

    # plot options for all scatter points
    uni_opts = {'lw': 0.3,
                'edgecolor': (0,0,0,0.7),
                's': 20}

    ax1.scatter(rd.loc[cind, ('Solid', 'logR')],
                LambdaB, **uni_opts, label='Data')
    sc = ax1.scatter(rd.loc[cind, ('Solid', 'logR')],
                     LambdaB_pred, **uni_opts, label='Model')
    ax1.legend()

    rx1.scatter(rd.loc[cind, ('Solid', 'logR')],
                LambdaB_pred - LambdaB, c=sc.get_facecolor(), **uni_opts)
    rx1.errorbar(rd.loc[cind, ('Solid', 'logR')],
                 LambdaB_pred - rd.loc[cind, ('Solid', 'LambdaB')], 
                 yerr=err(rd.loc[cind, ('Solid', 'LambdaB_eprop')]),
                 xerr=err(rd.loc[cind, ('Solid', 'logR_eprop')]),
                 color=(0,0,0,0.4), lw=0, elinewidth=1, label='_')

    ax2.scatter(rd.loc[cind, ('Solid', 'logR')],
                EpsilonB, **uni_opts)
    ax2.scatter(rd.loc[cind, ('Solid', 'logR')],
                EpsilonB_pred, **uni_opts)

    rx2.scatter(rd.loc[cind, ('Solid', 'logR')],
                EpsilonB_pred - EpsilonB, c=sc.get_facecolor(), **uni_opts)
    rx2.errorbar(rd.loc[cind, ('Solid', 'logR')],
                 EpsilonB_pred - rd.loc[cind, ('Solid', 'EpsilonB')], 
                 yerr=err(rd.loc[cind, ('Solid', 'EpsilonB_eprop')]),
                 xerr=err(rd.loc[cind, ('Solid', 'logR_eprop')]),
                 color=(0,0,0,0.4), lw=0, elinewidth=1)

    for rx in [rx1, rx2]:
        rx.axhline(0, color=(0,0,0,0.4), ls='dashed', zorder=-1)
        rx.yaxis.tick_right()
        rx.yaxis.set_label_position('right')

    # axis labels
    ax2.set_xlabel('$log_{10}R\ (mol\ m2\ s^{-1})$')
    rx2.set_xlabel('$log_{10}R\ (mol\ m2\ s^{-1})$')

    rx1.set_ylabel('Model - Data')
    rx2.set_ylabel('Model - Data')

    ax1.set_ylabel('$\lambda_B \\times 1000$', fontsize=12)
    ax2.set_ylabel('$\epsilon_B\ (\u2030_{NIST915})$', fontsize=12)

    parlab = ('$log_{10}R_b$: ' + fmt(params[-1], 1, 1, param_CIs[-1]) + '\n' + 
              '$^3K_f$: ' + fmt(params[1],2,8,param_CIs[1]) + '   $^3K_b$: ' + fmt(params[0],1,7,param_CIs[0]) + '\n' + 
              '$^4K_f$: ' + fmt(params[3],2,8,param_CIs[3]) + '  $^4K_b$: ' + fmt(params[2],1,7,param_CIs[2]))

    # parlab = ('$log_{10}' + 'R_b$: {logRb:1.2f}\n'.format(logRb=params[-1]) + 
    #           '$^3K_f$: {Kf3:6.2f}  $^3K_b$: {Kb3:.2f}\n'.format(Kf3=params[1],
    #                                                              Kb3=params[0]) + 
    #           '$^4K_f$: {Kf4:6.2f}  $^4K_b$: {Kb4:6.2f}'.format(Kf4=params[3],
    #                                                             Kb4=params[2]))

    ax2.text(.02, .02, parlab,
             transform=ax2.transAxes, ha='left', va='bottom', fontsize=7)

    fig.tight_layout()
    
    return fig, axs


def fmt(a, decimals=2, spc=0, ci=None):
    fmt = '{:' + '{:.0f}'.format(spc) + '.' + '{:.0f}'.format(decimals) + 'f}'
    if ci is None:
        return fmt.format(a)
    else:
        return '$' + fmt.format(a) + '_{' + fmt.format(ci[0]) + '}^{' + fmt.format(ci[1]) + '}$'