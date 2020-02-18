import numpy as np
from iisr_old.antenna_pattern.sun_utils import get_smoothed_elliptic_sun, \
    OPTICAL_DIAMETER
from matplotlib import rcParams
from matplotlib import pyplot as plt
from iisr_old.config import RESULTS_DIR


LANGUAGES = ['ru', 'en']

# Plot global parameters
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['axes.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'medium'
rcParams['xtick.labelsize'] = 'medium'
rcParams['ytick.labelsize'] = 'medium'
rcParams['figure.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'medium'
rcParams['legend.framealpha'] = 0.75
rcParams['axes.formatter.useoffset'] = False
rcParams['axes.formatter.use_mathtext'] = True


def plot_2d():
    theta, phi, vals = get_smoothed_elliptic_sun()

    fig = plt.figure()
    plt.subplot(121, projection='polar')
    plt.pcolormesh(phi, theta, vals)
    plt.ylim(0, theta.max())

    plt.subplot(122, projection='polar')
    plt.scatter(phi, theta, s=5.)
    plt.ylim(0, theta.max())


def plot_slices(language, save_dir, figsize=(4.5, 2.7), display_legend=False):
    theta, phi, vals = get_smoothed_elliptic_sun()

    fig = plt.figure(figsize=figsize)
    plt.subplot(111)
    unique_phi = np.unique(phi)
    theta_angle = {}
    vals_angle = {}
    for angle in [0, 90, 180, 270]:
        argmin_phi = np.argmin(np.abs(unique_phi - np.deg2rad(angle)))
        phi_angle = unique_phi[argmin_phi]
        mask = (phi == phi_angle)
        theta_angle[angle] = theta[mask].ravel()
        vals_angle[angle] = vals[mask].ravel()

    theta_north_south = np.concatenate([theta_angle[0], -theta_angle[180]])
    vals_north_south = np.concatenate([vals_angle[0], vals_angle[180]])
    argsort = np.argsort(theta_north_south)
    theta_north_south = theta_north_south[argsort]
    vals_north_south = vals_north_south[argsort]

    theta_east_west = np.concatenate([theta_angle[90], -theta_angle[270]])
    vals_east_west = np.concatenate([vals_angle[90], vals_angle[270]])
    argsort = np.argsort(theta_east_west)
    theta_east_west = theta_east_west[argsort]
    vals_east_west = vals_east_west[argsort]

    plt.plot(np.rad2deg(theta_north_south), vals_north_south, '--',
             label='N-S')
    plt.plot(np.rad2deg(theta_east_west), vals_east_west, label='E-W')
    xlabel = {'ru': r'$\theta$, град', 'en': r'$\theta$, deg'}
    plt.xlabel(xlabel[language])
    plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1.1)
    ylabel = {'ru': 'Нормированная яркость',
              'en': 'Normalized brightness'}
    plt.ylabel(ylabel[language])
    radii = np.rad2deg(OPTICAL_DIAMETER) / 2
    plt.xticks([-0.5, -radii, 0., radii, 0.5],
               ['-0.5', '$R_\odot$', '0', '$R_\odot$', '0.5'])
    yticks = np.arange(0., 1.1, 0.25)
    plt.yticks(yticks, ['{:.2f}'.format(tick) for tick in yticks])
    plt.grid()
    if display_legend:
        plt.legend(loc='upper right')

    fig.tight_layout()

    save_name = 'elliptic_sun_' + language + '.png'
    fig.savefig(str(save_dir / save_name))
    plt.close(fig)


if __name__ == '__main__':
    # plot_2d()
    # plt.show()
    save_dir = RESULTS_DIR / 'misc' / 'sun'
    for lang in LANGUAGES:
        plot_slices(lang, save_dir)