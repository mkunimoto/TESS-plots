import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from astroquery.mast import Catalogs
from brokenaxes import brokenaxes
from scipy.stats import binned_statistic

import tessDiffImage

def phase_diagram(time, flux, epo, per, frac):
    t = (time - epo) % per
    t[(t > frac*per)] -= per
    temp = np.column_stack((t, flux))
    sort = temp[np.argsort(temp[:,0])]
    phase_time = sort[:,0]
    phase_flux = sort[:,1]
    return phase_time, phase_flux

def binned_data(time, flux, nbins):
    bin_centre, _, _ = binned_statistic(time, time, statistic='mean', bins=nbins)
    bin_average, _, _ = binned_statistic(time, flux, statistic='mean', bins=nbins)
    bin_std, _, _ = binned_statistic(time, flux, statistic='std', bins=nbins)
    bin_count, _, _ = binned_statistic(time, flux, statistic='count', bins=nbins)
    bin_sterr = bin_std/np.sqrt(bin_count)
    return bin_centre, bin_average, bin_sterr

def plot_vs_phase(ax, time, flux, epo, per, dur, sigma, fs, ms, scale=24, phs=0, nbins=30, frac=0.5, colour='0.7', closeup=True):
    phase_time, phase_flux = phase_diagram(time, flux, epo + per*phs, per, frac)
    phase_time *= scale
    if closeup:
        ax.set_xlim([-2*dur*scale, 2*dur*scale])
        in_plot = (abs(phase_time) < 2*dur*scale)
    else:
        ax.set_xlim([min(phase_time), max(phase_time)])
        in_plot = (abs(phase_time) < max(phase_time))
    if nbins == 0:
        ax.plot(phase_time[in_plot], phase_flux[in_plot], 'k.', ms=ms)
    else:
        bin_centre, bin_average, bin_sterr = binned_data(phase_time[in_plot], phase_flux[in_plot], nbins)
        ax.plot(phase_time[in_plot], phase_flux[in_plot], color=colour, ls='none', marker='.', ms=ms)
        if closeup:
            ax.errorbar(bin_centre, bin_average, bin_sterr, fmt='k.', capsize=2, ms=ms)
        else:
            ax.plot(bin_centre, bin_average,'k.',ms=ms)
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    ax.tick_params(axis='both', which='minor', labelsize=fs-2)
    ax.set_ylim(top=ax.get_ylim()[1]+sigma, bottom=ax.get_ylim()[0]-sigma)

def plot_text(ax, star, planet, indent, spacing, start, ts):
    ax.annotate('Planet Properties:', [0, start+spacing], fontsize=ts, fontweight='bold')
    ax.annotate('$R_{p}$ = %.2f $R_{\oplus}$' % (planet['radius']), [indent, start], fontsize=ts)
    ax.annotate('$P$ = %.4f days' % (planet['period']), [indent, start-spacing], fontsize=ts)
    ax.annotate('$T_{0}$ = %.4f BJTD' % (planet['epoch']), [indent, start-spacing*2], fontsize=ts)
    ax.annotate('$T_{dur}$ = %.3f hours' % (planet['durationHours']), [indent, start-spacing*3], fontsize=ts)
    ax.annotate('Star Properties:', [0,start-spacing*5], fontsize=ts, fontweight='bold')
    catalogdata = Catalogs.query_object('TIC %i' % star['id'], radius=0.001, catalog='TIC')[:1]
    ax.annotate('$R_{s}$ = %.2f [%.2f] $R_{\odot}$' % (catalogdata['rad'], catalogdata['e_rad']), [indent, start-spacing*6], fontsize=ts)
    ax.annotate('$M_{s}$ = %.2f [%.2f] $M_{\odot}$' % (catalogdata['mass'], catalogdata['e_mass']), [indent, start-spacing*7], fontsize=ts)
    ax.annotate('$\\rho$ = %.2f [%.2f] $\\rho_{\odot}$' % (catalogdata['rho'], catalogdata['e_rho']), [indent, start-spacing*8], fontsize=ts)
    ax.annotate('$T_{eff}$ = %.0f [%.0f] K' % (catalogdata['Teff'], catalogdata['e_Teff']), [indent, start-spacing*9], fontsize=ts)
    ax.annotate('log$g$ = %.3f [%.3f]' % (catalogdata['logg'], catalogdata['e_logg']), [indent, start-spacing*10], fontsize=ts)
    ax.annotate('$T_{mag}$ = %.2f' % (catalogdata['Tmag']), [indent, start-spacing*11], fontsize=ts)
    ax.annotate('$Gaia_{mag}$ = %.2f' % (catalogdata['GAIAmag']), [indent, start-spacing*12], fontsize=ts)
    ax.annotate('RA = %.5f$^{o}$' % (catalogdata['ra']), [indent, start-spacing*13], fontsize=ts)
    ax.annotate('DEC = %.5f$^{o}$' % (catalogdata['dec']), [indent, start-spacing*14], fontsize=ts)
    ax.annotate('$\mu_{RA}$ = %.2f mas/yr' % (catalogdata['pmRA']), [indent, start-spacing*15], fontsize=ts)
    ax.annotate('$\mu_{DEC}$ = %.2f mas/yr' % (catalogdata['pmDEC']), [indent, start-spacing*16], fontsize=ts)
    ax.annotate('Parallax = %.3f' % (catalogdata['plx']), [indent, start-spacing*17], fontsize=ts)
    ax.axis('off')

def plot_odd(ax, time, flux, per, epo, dur, fs, ms, colour):
    phase = (time - epo) % (2*per)
    phase[phase > per] -= 2*per
    in_odd = (abs(phase) < 2*dur)
    phase_time, phase_flux = phase_diagram(time[in_odd], flux[in_odd], epo, per, 0.5)
    phase_time *= 24
    ax.set_xlim([-2*dur*24, 2*dur*24])
    in_plot = (abs(phase_time) < 2*dur*24)
    bin_centre, bin_average, bin_sterr = binned_data(phase_time[in_plot], phase_flux[in_plot], 30)
    ax.plot(phase_time[in_plot], phase_flux[in_plot], color='0.7', ls='none', marker='.', ms=ms)
    ax.errorbar(bin_centre, bin_average, bin_sterr, fmt='k.', capsize=2, ms=ms)
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    ax.tick_params(axis='both', which='minor', labelsize=fs-2)
    ax.text(0.02, 0.02, 'Odd', fontsize=fs, transform=ax.transAxes, color=colour)
    ax.set_xlabel('Hours from Midtransit',fontsize=fs)

def plot_even(ax, time, flux, per, epo, dur, fs, ms, colour):
    phase = (time - epo) % (2*per)
    phase[phase > per] -= 2*per
    in_even = (abs(phase) > per-2*dur)
    phase_time, phase_flux = phase_diagram(time[in_even], flux[in_even], epo, per, 0.5)
    phase_time *= 24
    ax.set_xlim([-2*dur*24, 2*dur*24])
    in_plot = (abs(phase_time) < 2*dur*24)
    bin_centre, bin_average, bin_sterr = binned_data(phase_time[in_plot], phase_flux[in_plot], 30)
    ax.plot(phase_time[in_plot], phase_flux[in_plot], color='0.7', ls='none', marker='.', ms=ms)
    ax.errorbar(bin_centre, bin_average, bin_sterr, fmt='k.', capsize=2, ms=ms)
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    ax.tick_params(axis='both', which='minor', labelsize=fs-2)
    ax.set_yticks([])
    ax.text(0.02, 0.02, 'Even', fontsize=fs, transform=ax.transAxes, color=colour)
    ax.set_xlabel('Hours from Midtransit',fontsize=fs)

def plot_report(star, planet, imageData):
    fsize, fs, ms = (13, 9), 8, 5
    time, raw, flux = star['time'], star['raw'], star['flux']
    per = planet['period']
    epo = planet['epoch']
    dur = planet['durationHours']/24.
    fig = plt.figure(figsize=fsize)
    gs = gridspec.GridSpec(nrows=4, ncols=6, hspace=0.5, wspace=0.3)
    # Set up subplots
    axPhase = fig.add_subplot(gs[2, :2])
    axClose = fig.add_subplot(gs[3, :2])
    oddeven_gs = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs[1, 2:4], wspace=0)
    axOdd = fig.add_subplot(oddeven_gs[:, 0])
    axEven = fig.add_subplot(oddeven_gs[:, 1])
    axSec = fig.add_subplot(gs[2, 2:4])
    axHalf = fig.add_subplot(gs[3, 2:4])
    axText = fig.add_subplot(gs[:,5])
    pix_gs = gridspec.GridSpecFromSubplotSpec(nrows=4,ncols=1, subplot_spec=gs[:,4], hspace=0.5)
    axDiff1 = fig.add_subplot(pix_gs[0, :])
    axDiff2 = fig.add_subplot(pix_gs[2, :])
    axDir1 = fig.add_subplot(pix_gs[1, :])
    axDir2 = fig.add_subplot(pix_gs[3, :])
    # Plot full lightcurve
    gap = 20
    end = np.nonzero(np.diff(time) > gap)[0]
    ngaps = len(end)
    start = np.append(0,end+1)
    end = np.append(end,-1)
    bnds = []
    for i in range(ngaps+1):
        if i == 0:
            bnds.append([time[0], time[end[0]]+1])
        elif i == ngaps:
            bnds.append([time[start[i]]-1, time[-1]])
        else:
            bnds.append([time[start[i]]-1, time[end[i]]+1])
    axRaw = brokenaxes(xlims=bnds,subplot_spec=gs[0,:4],wspace=0.05,despine=False,d=0)
    axDet = brokenaxes(xlims=bnds,subplot_spec=gs[1,:2],wspace=0.05,despine=False,d=0)
    axRaw.set_title('TIC %i' % (star['id']), fontsize=fs+2)
    axRaw.plot(time,raw,"k.",ms=ms)
    axDet.plot(time,flux,"k.",ms=ms)
    if ngaps > 0:
        for x in start[1:]:
            axRaw.axvline(x=time[x]-1,ls='--',color='k')
            axDet.axvline(x=time[x]-1,ls='--',color='k')
        for x in end[:-1]:
            axRaw.axvline(x=time[x]+1,ls='--',color='k')
            axDet.axvline(x=time[x]+1,ls='--',color='k')
    axRaw.tick_params(axis='both', which='major', labelsize=fs-2)
    axRaw.tick_params(axis='both', which='minor', labelsize=fs-2)
    axDet.tick_params(axis='both', which='major', labelsize=fs-2)
    axDet.tick_params(axis='both', which='minor', labelsize=fs-2)
    axRaw.set_xlabel('BJD - 2457000', fontsize=fs)
    axDet.set_xlabel('BJD - 2457000', fontsize=fs)
    phase = (time - epo) % per
    phase[phase > 0.5*per] -= per
    out_transit = (abs(phase) > 2*dur)
    sigma = 1.4826*np.nanmedian(np.abs(flux - np.nanmedian(flux)))
    axRaw.set_ylim(top=axRaw.get_ylim()[0][1]+sigma,bottom=axRaw.get_ylim()[0][0]-sigma)
    axDet.set_ylim(top=axDet.get_ylim()[0][1]+sigma,bottom=axDet.get_ylim()[0][0]-sigma)
    # Mark all transits on lightcurves by odd-even
    odd_colour = 'C0'
    even_colour = 'C1'
    cent = epo
    nt = 1
    while cent < time[-1]:
        if nt % 2 == 1:
            colour = odd_colour
        else:
            colour = even_colour
        for bnd in bnds:
            if (cent > bnd[0]) & (cent < bnd[1]):
                axRaw.axvline(x=cent, ymax=0.1, color=colour)
                axDet.axvline(x=cent, ymax=0.1, color=colour)
        nt += 1
        cent += per
    cent = epo - per
    nt = 0
    while cent > time[0]:
        if nt % 2 == 1:
            colour = odd_colour
        else:
            colour = even_colour
        for bnd in bnds:
            if (cent > bnd[0]) & (cent < bnd[1]):
                axRaw.axvline(x=cent, ymax=0.1, color=colour)
                axDet.axvline(x=cent, ymax=0.1, color=colour)
        nt -= 1
        cent -= per
    # Plot full phase diagram
    plot_vs_phase(axPhase, time, flux, epo, per, dur, sigma, fs, ms, scale=1, nbins=round(per*24*60/30), frac=0.7, closeup=False)
    axPhase.set_xlabel('Days from Midtransit',fontsize=fs)
    axPhase.axvline(x=0, ymax=0.1, color='r')
    # Close up phase diagram
    plot_vs_phase(axClose, time, flux, epo, per, dur, sigma, fs, ms)
    axClose.set_xlabel('Hours from Midtransit', fontsize=fs)
    # Odd transits
    plot_odd(axOdd, time, flux, per, epo, dur, fs, ms, odd_colour)
    axOdd.set_ylim(axClose.get_ylim())
    # Even transits
    plot_even(axEven, time, flux, per, epo, dur, fs, ms, even_colour)
    axEven.set_ylim(axClose.get_ylim())
    # Sig secondary
    try:
        planet['phs_sec']
    except KeyError:
        planet['phs_sec'] = 0.5
    plot_vs_phase(axSec, time, flux, epo, per, dur, sigma, fs, ms, phs=planet['phs_sec'])
    axSec.set_xlabel('Hours from Secondary (Phase %.2f)' % (planet['phs_sec']),fontsize=fs)
    # Half secondary
    plot_vs_phase(axHalf, time, flux, epo, per, dur, sigma, fs, ms, phs=0.5)
    axHalf.set_xlabel('Hours from Secondary (Phase 0.50)',fontsize=fs)
    # Difference Images
    tdi = tessDiffImage.tessDiffImage(star,spiceFileLocation = "..")
    tdi.draw_pix_catalog(imageData[0]['diffSNRImage'], imageData[1], imageData[1]["extent"], ax=axDiff1, filterStars=True, dMagThreshold=4, fs=fs, ss=10)
    axDiff1.set_title('Diff Image SNR', fontsize=fs)
    tdi.draw_pix_catalog(imageData[0]['diffSNRImage'], imageData[1], imageData[1]["extentClose"], ax=axDiff2, close=True, filterStars=True, dMagThreshold=4, fs=fs, ss=40)
    axDiff2.set_title('Diff Image SNR (Close)', fontsize=fs)
    tdi.draw_pix_catalog(imageData[0]['meanOutTransit'], imageData[1], imageData[1]["extent"], ax=axDir1, filterStars=True, dMagThreshold=4, fs=fs, ss=10)
    axDir1.set_title('Direct Image', fontsize=fs)
    tdi.draw_pix_catalog(imageData[0]['meanOutTransit'], imageData[1], imageData[1]["extentClose"], ax=axDir2, close=True, filterStars=True, dMagThreshold=4, fs=fs, ss=40)
    axDir2.set_title('Direct Image (Close)', fontsize=fs)
    # Text
    plot_text(axText, star, planet, 0.15, 0.035, 0.95, 8)
    # Save figure
    plt.savefig('tic%i/report.png' % (star['id']),dpi=150)
