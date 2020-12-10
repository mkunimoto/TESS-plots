import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np

from brokenaxes import brokenaxes
from scipy.stats import binned_statistic
from tessDiffImage import plot_pix_catalog

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

def plot_text(ax, tic, res, star_params, indent, spacing, start, ts):
    Rp = 109.*res['rprs']*star_params['rad']
    Rp_err = 109.*np.sqrt((res['rprs']*star_params['e_rad'])**2 + (star_params['rad']*res['rprs_err'])**2)
    rho = (365.25/res['per'])**2.*(res['ars']/215.)**3.
    rho_err = rho*np.sqrt((3*res['ars_err']/res['ars'])**2 + (2*res['per_err']/res['per'])**2)
    ax.annotate('Fitted Properties:', [0, start+spacing], fontsize=ts, fontweight='bold')
    ax.annotate('$P$ = %.4f [%.4f] days' % (res['per'], res['per_err']), [indent, start], fontsize=ts)
    ax.annotate('$T_{0}$ = %.4f [%.4f] BJTD' % (res['epo'], res['epo_err']), [indent, start-spacing], fontsize=ts)
    ax.annotate('$R_{p}/R_{s}$ = %.3f [%.3f]' % (res['rprs'], res['rprs_err']), [indent, start-spacing*2], fontsize=ts)
    ax.annotate('$a/R_{s}$ = %.3f [%.3f]' % (res['ars'], res['ars_err']), [indent, start-spacing*3], fontsize=ts)
    ax.annotate('$b$ = %.3f [%.3f]' % (res['b'], res['b_err']), [indent, start-spacing*4], fontsize=ts)
    ax.annotate('BLS S/N = %.2f' % (res['SN']), [indent, start-spacing*5], fontsize=ts)
    ax.annotate('Derived Properties:', [0, start-spacing*6], fontsize=ts, fontweight='bold')
    ax.annotate('$R_{p}$ = %.2f [%.2f] $R_{\oplus}$' % (Rp, Rp_err), [indent, start-spacing*7], fontsize=ts)
    ax.annotate('$\\rho_{s}$ = %.2f [%.2f] $\\rho_{\odot}$' % (rho, rho_err), [indent, start-spacing*8], fontsize=ts) 
    ax.annotate('$T_{dur}$ = %.3f hours' % (res['dur']*24), [indent, start-spacing*9], fontsize=ts)
    ax.annotate('$T_{flat}/T_{dur}$ = %.3f' % (1 - 2*res['din']/res['dur']), [indent, start-spacing*10], fontsize=ts)
    ax.annotate('$\delta$ = %i ppm' % (int(res['dep']*1e6)), [indent, start-spacing*11], fontsize=ts)
    ax.annotate('Stellar Properties:', [0,start-spacing*12], fontsize=ts, fontweight='bold')
    ax.annotate('$R_{s}$ = %.2f [%.2f] $R_{\odot}$' % (star_params['rad'], star_params['e_rad']), [indent, start-spacing*13], fontsize=ts)
    ax.annotate('$M_{s}$ = %.2f [%.2f] $M_{\odot}$' % (star_params['mass'], star_params['e_mass']), [indent, start-spacing*14], fontsize=ts)
    ax.annotate('$\\rho$ = %.2f [%.2f] $\\rho_{\odot}$' % (star_params['rho'], star_params['e_rho']), [indent, start-spacing*15], fontsize=ts)
    ax.annotate('$T_{eff}$ = %.0f [%.0f] K' % (star_params['teff'], star_params['e_teff']), [indent, start-spacing*16], fontsize=ts)
    ax.annotate('log$g$ = %.3f [%.3f]' % (star_params["logg"], star_params['e_logg']), [indent, start-spacing*17], fontsize=ts)
    ax.annotate('$T_{mag}$ = %.2f' % (star_params["tmag"]), [indent, start-spacing*18], fontsize=ts)
    ax.annotate('$Gaia_{mag}$ = %.2f' % (star_params["gaiamag"]), [indent, start-spacing*19], fontsize=ts)
    ax.annotate('$Gaia_{rp} - Gaia_{bp}$ = %.2f' % (star_params["gaiarp"] - star_params["gaiabp"]), [indent, start-spacing*20], fontsize=ts)
    ax.annotate('RA = %.5f$^{o}$' % (star_params['ra']), [indent, start-spacing*21], fontsize=ts)
    ax.annotate('DEC = %.5f$^{o}$' % (star_params['dec']), [indent, start-spacing*22], fontsize=ts)
    ax.annotate('$\mu_{RA}$ = %.2f mas/yr' % (star_params["pmra"]), [indent, start-spacing*23], fontsize=ts)
    ax.annotate('$\mu_{DEC}$ = %.2f mas/yr' % (star_params["pmdec"]), [indent, start-spacing*24], fontsize=ts)
    ax.annotate('Parallax = %.3f' % (star_params["plx"]), [indent, start-spacing*25], fontsize=ts)
    ax.annotate('TOI Cross-Match:', [0,start-spacing*26], fontsize=ts, fontweight='bold')
    toi = res['toi']
    ntois = len(toi)
    if ntois == 0:
        ax.annotate('No TOIs', [indent, start-spacing*27], fontsize=ts)
    else:
        no_match = True
        i = 0
        while no_match & (i < ntois):
            toi_id = toi.iloc[i]['TOI']
            toi_per = toi.iloc[i]['Period (days)']
            toi_epo = toi.iloc[i]['Epoch (BJD)'] - 2457000
            toi_disp = toi.iloc[i]['TFOPWG Disposition']
            if not isinstance(toi_disp, str):
                toi_disp = toi.iloc[i]['TESS Disposition']
            else:
                toi_disp = toi_disp + '*'
            if abs(toi_per - res['per']) < 10*res['per_err']:
                comp_epo = toi_epo - round((toi_epo - res['epo'])/toi_per)*toi_per
                if abs(comp_epo - res['epo']) < 10*res['epo_err']:
                    ax.annotate('Match with TOI %s (%s)' % (toi_id, toi_disp),[indent,start-spacing*27],fontsize=ts,color='r')
                    #ax.annotate('$P$ = %.3f days' % toi_per,[indent,start-spacing*28],fontsize=ts,color='r')
                    #ax.annotate('$T_{0}$ = %.3f days' % toi_epo,[indent,start-spacing*29],fontsize=ts,color='r')
                    no_match = False
            i += 1
        if no_match:
            ax.annotate('No match with %i TOIs' % ntois,[indent,start-spacing*27],fontsize=ts,color='k')
    ax.axis('off')

def plot_page1(star, planet, imageData):
    fsize, fs, ms = (13, 9), 8, 5
    time, raw, flux = star['time'], star['raw'], star['flux']
    flux1 = star['flux1']
    flux2 = star['flux2']
    flux3 = star['flux3']
    per = planet['period']
    epo = planet['epoch']
    dur = planet['durationHours']/24.
    fig = plt.figure(figsize=fsize)
    gs = gridspec.GridSpec(nrows=5, ncols=6, hspace=0.5, wspace=0.3)
    # Set up subplots
    axPhase = fig.add_subplot(gs[2, :2])
    axClose = fig.add_subplot(gs[3, :2])
    oddeven_gs = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs[1, 2:4], wspace=0)
    axOdd = fig.add_subplot(oddeven_gs[:, 0])
    axEven = fig.add_subplot(oddeven_gs[:, 1])
    axSec = fig.add_subplot(gs[2, 2:4])
    axHalf = fig.add_subplot(gs[3, 2:4])
    axText = fig.add_subplot(gs[1:,5])
    pix_gs = gridspec.GridSpecFromSubplotSpec(nrows=4,ncols=1, subplot_spec=gs[1:,4], hspace=0.5)
    axDiff1 = fig.add_subplot(pix_gs[0, :])
    axDiff2 = fig.add_subplot(pix_gs[2, :])
    axDir1 = fig.add_subplot(pix_gs[1, :])
    axDir2 = fig.add_subplot(pix_gs[3, :])
    aps_gs = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs[4, :4], wspace=0)
    axAp1 = fig.add_subplot(aps_gs[:, 0])
    axAp2 = fig.add_subplot(aps_gs[:, 1])
    axAp3 = fig.add_subplot(aps_gs[:, 2])
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
    axRaw = brokenaxes(xlims=bnds,subplot_spec=gs[0,:],wspace=0.05,despine=False,d=0)
    axDet = brokenaxes(xlims=bnds,subplot_spec=gs[1,:2],wspace=0.05,despine=False,d=0)
    axRaw.plot(time,raw,"k.",ms=ms)
    axDet.plot(time,flux,"k.",ms=ms)
    #axDet.plot(time,model,"r")
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
    axRaw.set_title('TIC %i' % (star['id']), fontsize=fs+2)
    axRaw.set_xlabel('BJD - 2457000', fontsize=fs)
    axDet.set_xlabel('BJD - 2457000', fontsize=fs)
    phase = (time - epo) % per
    phase[np.where(phase > 0.5*per)] -= per
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
    #model_phase, model_flux = phase_diagram(time, model, epo, per, 0.7)
    #axPhase.plot(model_phase, model_flux, "r")
    axPhase.set_xlabel('Days from Midtransit',fontsize=fs)
    axPhase.axvline(x=0, ymax=0.1, color='r')
    #axPhase.axvline(x=per*res['phs_sec'], ymax=0.1, color='r')
    # Close up phase diagram
    plot_vs_phase(axClose, time, flux, epo, per, dur, sigma, fs, ms)
    #model_phase, model_flux = phase_diagram(time, model, epo, per, 0.5)
    #axClose.plot(model_phase*24, model_flux, "r")
    #axClose.text(0.01, 0.02, '%ippm (%.1f$\sigma$)' % (int(dep*1e6), res['sig_pri']), fontsize=fs, transform=axClose.transAxes)
    axClose.set_xlabel('Hours from Midtransit', fontsize=fs)
    # Odd transits
    phase = (time - epo) % (2*per)
    phase[np.where(phase > per)] -= 2*per
    in_odd = (abs(phase) < 2*dur)
    in_even = (abs(phase) > per-2*dur)
    phase_time, phase_flux = phase_diagram(time[in_odd], flux[in_odd], epo, per, 0.5)
    phase_time *= 24
    axOdd.set_xlim([-2*dur*24, 2*dur*24])
    in_plot = (abs(phase_time) < 2*dur*24)
    bin_centre, bin_average, bin_sterr = binned_data(phase_time[in_plot], phase_flux[in_plot], 30)
    axOdd.plot(phase_time[in_plot], phase_flux[in_plot], color='0.7', ls='none', marker='.', ms=ms)
    axOdd.errorbar(bin_centre, bin_average, bin_sterr, fmt='k.', capsize=2, ms=ms)
    #odd_model = transit_model(time[in_odd], res['epo'], res['per'], res['rprs_odd'], res['b'], res['ars'], res['u1'], res['u2'])
    #model_phase, model_flux = phase_diagram(time[in_odd], odd_model, epo, per, 0.5)
    #axOdd.plot(model_phase*24, model_flux, odd_colour)
    #axOdd.plot(bin_centre,bin_average,'k.',ms=ms)
    axOdd.tick_params(axis='both', which='major', labelsize=fs-2)
    axOdd.tick_params(axis='both', which='minor', labelsize=fs-2)
    axOdd.set_ylim(axClose.get_ylim())
    axOdd.text(0.02, 0.02, 'Odd', fontsize=fs, transform=axOdd.transAxes, color=odd_colour)
    axOdd.set_xlabel('Hours from Midtransit',fontsize=fs)
    # Even transits
    phase_time, phase_flux = phase_diagram(time[in_even], flux[in_even], epo, per, 0.5)
    phase_time *= 24
    axEven.set_xlim([-2*dur*24, 2*dur*24])
    in_plot = (abs(phase_time) < 2*dur*24)
    bin_centre, bin_average, bin_sterr = binned_data(phase_time[in_plot], phase_flux[in_plot], 30)
    axEven.plot(phase_time[in_plot], phase_flux[in_plot], color='0.7', ls='none', marker='.', ms=ms)
    axEven.errorbar(bin_centre, bin_average, bin_sterr, fmt='k.', capsize=2, ms=ms)
    #even_model = transit_model(time[in_even], res['epo'], res['per'], res['rprs_even'], res['b'], res['ars'], res['u1'], res['u2'])
    #model_phase, model_flux = phase_diagram(time[in_even], even_model, epo, per, 0.5)
    #axEven.plot(model_phase*24, model_flux, even_colour)
    #axEven.plot(bin_centre,bin_average,'k.',ms=ms)
    axEven.tick_params(axis='both', which='major', labelsize=fs-2)
    axEven.tick_params(axis='both', which='minor', labelsize=fs-2)
    axEven.set_ylim(axClose.get_ylim())
    axEven.set_yticks([])
    axEven.text(0.02, 0.02, 'Even', fontsize=fs, transform=axEven.transAxes, color=even_colour)
    axEven.set_xlabel('Hours from Midtransit',fontsize=fs)
    #axOdd.set_title('Odd-Even:', loc='right',fontsize=fs)
    #axEven.set_title(' %.2f$\sigma$' % (res['sig_oe']), loc='left', fontsize=fs)
    # Sig secondary
    #plot_vs_phase(axSec, time, flux, epo, per, dur, sigma, fs, ms, phs=res['phs_sec'])
    #axSec.text(0.01,0.02,'%ippm (%.1f$\sigma$)' % (int(res['dep_sec']*1e6), res['sig_sec']), fontsize=fs, transform=axSec.transAxes)
    #axSec.set_xlabel('Hours from Secondary (Phase %.4f)' % (res['phs_sec']),fontsize=fs)
    # Half secondary
    plot_vs_phase(axHalf, time, flux, epo, per, dur, sigma, fs, ms, phs=0.5)
    axHalf.set_xlabel('Hours from Secondary (Phase 0.5)',fontsize=fs)
    # Depth-apertures
    plot_vs_phase(axAp1, time, flux1, epo, per, dur, sigma, fs, ms)
    plot_vs_phase(axAp2, time, flux2, epo, per, dur, sigma, fs, ms)
    plot_vs_phase(axAp3, time, flux3, epo, per, dur, sigma, fs, ms)
    axAp1.set_ylim(top=axAp3.get_ylim()[1], bottom=axAp3.get_ylim()[0])
    axAp2.set_ylim(top=axAp3.get_ylim()[1], bottom=axAp3.get_ylim()[0])
    #axAp1.axhline(y=1-dep,ls='--',color='C2')
    #axAp2.axhline(y=1-dep,ls='--',color='C2')
    #axAp3.axhline(y=1-dep,ls='--',color='C2')
    #if lc.read_attr('bestap') == 1:
    #    fw1, fw2, fw3 = 'bold', 'normal', 'normal'
    #elif lc.read_attr('bestap') == 2:
    #    fw1, fw2, fw3 = 'normal', 'bold', 'normal'
    #elif lc.read_attr('bestap') == 3:
    #    fw1, fw2, fw3 = 'normal', 'normal', 'bold'
    #else:
    #    fw1, fw2, fw3 = 'normal', 'normal', 'normal'
    axAp1.text(0.02, 0.02, 'Ap1', fontsize=fs, transform=axAp1.transAxes, color='C2')
    axAp2.text(0.02, 0.02, 'Ap2', fontsize=fs, transform=axAp2.transAxes, color='C2')
    axAp3.text(0.02, 0.02, 'Ap3', fontsize=fs, transform=axAp3.transAxes, color='C2')
    axAp2.set_yticks([])
    axAp3.set_yticks([])
    axAp2.set_xlabel('Hours from Midtransit', fontsize=fs)
    # Difference Images
    plot_pix_catalog(imageData[0]['diffImageSigma'], imageData[1], ax=axDiff1, filterStars=True, dMagThreshold=4, fs=fs, ss=10)
    axDiff1.set_title('Diff Image SNR', fontsize=fs)
    plot_pix_catalog(imageData[0]['diffImageSigma'], imageData[1], ax=axDiff2, close=True, filterStars=True, dMagThreshold=4, fs=fs, ss=40)
    axDiff2.set_title('Diff Image SNR (Close)', fontsize=fs)
    plot_pix_catalog(imageData[0]['meanOutTransit'], imageData[1], ax=axDir1, filterStars=True, dMagThreshold=4, fs=fs, ss=10)
    axDir1.set_title('Direct Image', fontsize=fs)
    plot_pix_catalog(imageData[0]['meanOutTransit'], imageData[1], ax=axDir2, close=True, filterStars=True, dMagThreshold=4, fs=fs, ss=40)
    axDir2.set_title('Direct Image (Close)', fontsize=fs)
    # Text
    #plot_text(axText, tic, res, star_params, 0.15, 0.035, 0.95, 8)
    # Save figure
    plt.savefig('tic%i/page1.png' % (star['id']),dpi=150)
    #plt.close()

def plot_page2(star, imageData):
    fs = 12
    indent, spacing, start, ts = 0.3, 0.035, 1., 10
    fsize = (13,6)
    fig = plt.figure(figsize=fsize)
    gs = gridspec.GridSpec(nrows=2, ncols=3, hspace=0.2, wspace=0.2)
    axDiff1 = fig.add_subplot(gs[0,0])
    axDiff2 = fig.add_subplot(gs[0,1])
    axDir1 = fig.add_subplot(gs[1,0])
    axDir2 = fig.add_subplot(gs[1,1])
    axText = fig.add_subplot(gs[0:,2])
    plot_pix_catalog(imageData[0]['diffImageSigma'], imageData[1], ax=axDiff1, annotate=True, magColorBar=True, fs=fs-2, ss=50)
    axDiff1.set_title('Diff Image SNR', fontsize=fs-2)
    plot_pix_catalog(imageData[0]['diffImageSigma'], imageData[1], ax=axDiff2, annotate=True, magColorBar=True, close=True, fs=fs-2, ss=200)
    axDiff2.set_title('Diff Image SNR (Close)', fontsize=fs-2)
    plot_pix_catalog(imageData[0]['meanOutTransit'], imageData[1], ax=axDir1, annotate=True, magColorBar=True, fs=fs-2, ss=50)
    axDir1.set_title('Direct Image', fontsize=fs-2)
    plot_pix_catalog(imageData[0]['meanOutTransit'], imageData[1], ax=axDir2, annotate=True, magColorBar=True, close=True, fs=fs-2, ss=200)
    axDir2.set_title('Direct Image (Close)', fontsize=fs-2)
    # List of tics
    axText.annotate('ID', [0,start], fontsize=ts, fontweight='bold')
    axText.annotate('TIC', [indent*0.5,start], fontsize=ts, fontweight='bold')
    axText.annotate('Sep (")', [indent*1.8,start], fontsize=ts, fontweight='bold')
    axText.annotate('TESS Mag',[indent*3,start], fontsize=ts, fontweight='bold')
    catalogData = imageData[1]
    targetMag = catalogData["ticMag"][0]
    cnt = 0
    for s in range(len(catalogData['ticID'])):
        px = catalogData["ticColPix"][s] - catalogData["dCol"]
        py = catalogData["ticRowPix"][s] - catalogData["dRow"]
        ticMag = catalogData["ticMag"][s]
        if ((ticMag-targetMag < 4) & (px >= catalogData['extent'][0]) & (px <= catalogData['extent'][1]) & (py > catalogData['extent'][2]) & (py < catalogData['extent'][3])):
            ticid = catalogData['ticID'][s]
            sep = catalogData['separation'][s]
            mag = catalogData['ticMag'][s]
            axText.annotate(str(s), [0,start-spacing*(cnt+1)], fontsize=ts)
            axText.annotate(str(ticid), [indent*0.5,start-spacing*(cnt+1)], fontsize=ts)
            axText.annotate('%.1f' % sep, [indent*1.8,start-spacing*(cnt+1)], fontsize=ts)
            axText.annotate('%.3f' % mag, [indent*3,start-spacing*(cnt+1)], fontsize=ts)
            cnt += 1
    # Save figure
    axText.axis('off')
    plt.savefig('tic%i/page2.png' % (star['id']),dpi=100)
    #plt.close()

