import os
import glob
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
import pickle

import tess_stars2px as trdp
import barycentricCorrection as bc

def make_ffi_difference_image(ticData, thisPlanet=None, nPixOnSide = 20, dMagThreshold = 4, allowedBadCadences = 2):
    fitsList = get_tess_cut(ticData["id"], ticData["raDegrees"], ticData["decDegrees"], ticData["sector"], nPixOnSide = nPixOnSide)
    ticName = "tic" + str(ticData["id"])
    for fitsFile in fitsList:
        pixelData = get_cadence_data(fitsFile)
        # check that this is the camera and sector that we want here
        if (ticData["cam"] != None) & ((ticData["sector"] != pixelData["sector"]) | (ticData["cam"] != pixelData["camera"])):
            continue

        sectorQflags = np.array([])
        if ticData["qualityFiles"] != None:
            for sectorList in ticData["qualityFiles"]:
                if sectorList[0] == pixelData["sector"]:
                    for fname in sectorList[1]:
                        sectorQflags = np.append(sectorQflags, np.loadtxt(fname, usecols=(1)))
        elif ticData["qualityFlags"] != None:
            for sectorList in ticData["qualityFlags"]:
                if sectorList[0] == pixelData["sector"]:
                    sectorQflags = sectorList[1]
            
        if len(sectorQflags) == len(pixelData["quality"]):
            pixelData["quality"] = pixelData["quality"] + sectorQflags
        else:
            print("no sector quality flags of the same length as FFI quality flags")

        catalogData = make_stellar_scene(pixelData, ticData, ticName, dMagThreshold)
        
        if thisPlanet == None:
            for p, planetData in enumerate(ticData["planetData"]):
                planetData["planetIndex"] = p
                planetData["planetID"] = p
                make_planet_difference_image(ticData, planetData, pixelData, catalogData, ticName, dMagThreshold, allowedBadCadences = allowedBadCadences)
        else:
            planetData = ticData["planetData"][thisPlanet]
            planetData["planetIndex"] = thisPlanet
            planetData["planetID"] = thisPlanet
            make_planet_difference_image(ticData, planetData, pixelData, catalogData, ticName, dMagThreshold, allowedBadCadences = allowedBadCadences)

    #       print('rm ' + ticName + '/*.fits')
    os.system('rm ' + ticName + '/*.fits')

def make_planet_difference_image(ticData, planetData, pixelData, catalogData, ticName, dMagThreshold = 4, allowedBadCadences = 0):
    # for each planet, make an array of cadences that are in the transits of other planets
    inOtherTransitIndices = [];
    for pi, otherPlanet in enumerate(ticData["planetData"]):
        if pi == planetData["planetIndex"]:
            continue
        else:
            transitTimes, transitIndex = find_transit_times(pixelData, otherPlanet)
            
            durationDays = otherPlanet["durationHours"]/24;
            transitAverageDurationDays = durationDays/2;
            for i in transitIndex:
                thisTransitInIndices = np.argwhere(
                    np.abs(pixelData["time"][i] - pixelData["time"]) < transitAverageDurationDays)
                inOtherTransitIndices = np.append(inOtherTransitIndices, thisTransitInIndices)
                
    inOtherTransitIndices = np.array(inOtherTransitIndices).astype(int)
    pixelData["inOtherTransit"] = np.zeros(pixelData["quality"].shape)
    pixelData["inOtherTransit"][inOtherTransitIndices] = 1

    inTransitIndices, outTransitIndices, transitIndex = find_transits(pixelData, planetData, allowedBadCadences = allowedBadCadences)
    diffImageData = make_difference_image(pixelData, inTransitIndices, outTransitIndices)
    draw_difference_image(diffImageData, pixelData, ticData, planetData, catalogData, ticName, dMagThreshold)
    draw_lc_transits(pixelData, planetData, inTransitIndices, outTransitIndices, transitIndex, ticName)

    f = open(ticName + "/imageData_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pickle", 'wb')
    pickle.dump([diffImageData, catalogData, pixelData, inTransitIndices, outTransitIndices, transitIndex], f, pickle.HIGHEST_PROTOCOL)
    f.close()


def get_tess_cut(ticNumber, ra, dec, sector=None, fitsNum = 0, nPixOnSide = 20):
    ticName = "tic" + str(ticNumber)
    if sector == None:
        curlStr = 'curl "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=' \
                    + str(ra) + '&dec=' + str(dec) + '&y=' + str(nPixOnSide) + '&x=' + str(nPixOnSide) \
                    + '" --output ' + ticName + '.zip'
    else:
        curlStr = 'curl "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=' \
                    + str(ra) + '&dec=' + str(dec) + '&y=' + str(nPixOnSide) + '&x=' + str(nPixOnSide) + '&sector=' + str(sector) \
                    + '" --output ' + ticName + '.zip'
    print(curlStr)
    os.system(curlStr)
    os.system('mkdir ' + ticName)
    os.system('unzip ' + ticName + '.zip' + ' -d ' + ticName)
    os.system('rm ' + ticName + '.zip')
    fitsList = glob.glob(ticName + '/*.fits')
    print(fitsList)
    return fitsList
    
def get_cadence_data(fitsFile):
    fitsData = fits.open(fitsFile)
    cadenceData = fitsData[1].data # get the binary table
    binHeader = fitsData[1].header # for the binary table keywords
    priHeader = fitsData[0].header # for the primary table keywords
    
    pixelData = {}

    # read important keywords
    # the reference pixel coordinates are the low corner of the central pixel
    pixelData["referenceCol"] = binHeader["1CRPX4"] # reference pixel along axis 1
    pixelData["referenceRow"] = binHeader["2CRPX4"] # reference pixel along axis 1
    pixelData["referenceRa"] = binHeader["1CRVL4"] # reference pixel ra in degrees
    pixelData["referenceDec"] = binHeader["2CRVL4"] # reference pixel dec in degrees
    pixelData["cornerCol"] = binHeader["1CRV4P"] # corner ra in degrees
    pixelData["cornerRow"] = binHeader["2CRV4P"] # corner dec in degrees
    pixelData["sector"] = priHeader["SECTOR"] # FFI sector
    pixelData["camera"] = priHeader["CAMERA"] # FFI camera
    pixelData["ccd"] = priHeader["CCD"] # FFI ccd

    pixelData["rawTime"] = np.zeros(len(cadenceData))
    pixelData["ffiBarycentricCorrection"] = np.zeros(len(cadenceData))
    pixelData["flux"] = np.zeros((len(cadenceData), cadenceData[0][4].shape[0], cadenceData[0][4].shape[1]))
    pixelData["fluxErr"] = np.zeros((len(cadenceData), cadenceData[0][4].shape[0], cadenceData[0][4].shape[1]))
    pixelData["quality"] = np.zeros(len(cadenceData))
    for i in range(len(pixelData["rawTime"])):
        pixelData["rawTime"][i] = cadenceData[i][0]
        pixelData["ffiBarycentricCorrection"][i] = cadenceData[i][1]
        pixelData["flux"][i,:,:] = cadenceData[i][4]
        pixelData["fluxErr"][i,:,:] = cadenceData[i][5]
        pixelData["quality"][i] = cadenceData[i][8]
        
    # perform the barycentric correction
    # Tess cut time is barycentric correccted to the center of the source FFI
    # so compute the spacecraft clock time by subtracting the supplied
    # barycentric correction, and them computing the barycentric correction
    # for the target RA and Dec.  We use the reference pixel RA and Dec
    baryCorrector = bc.barycentricCorrection()
    tessToJulianOffset = 2457000
    spacecraftTime = pixelData["rawTime"] - pixelData["ffiBarycentricCorrection"]
    pixelData["barycentricCorrection"], _ = baryCorrector.computeCorrection(spacecraftTime + tessToJulianOffset,
                                             pixelData["referenceRa"], pixelData["referenceDec"])
    pixelData["time"] = spacecraftTime + pixelData["barycentricCorrection"]
        
#    print("time: " + str([np.min(pixelData["time"]), np.max(pixelData["time"])]))
    return pixelData
    
def find_transit_times(pixelData, planetData):

    transitTimes = [];
    dt = np.min(np.diff(pixelData["time"])) # days
    firstTransitTime = np.ceil((pixelData["time"][0] - planetData["epoch"])/planetData["period"])*planetData["period"] + planetData["epoch"]
    n = np.ceil((pixelData["time"][0] - planetData["epoch"])/planetData["period"]);
    while planetData["epoch"] + n*planetData["period"] < pixelData["time"][-1]:
      transitTimes = np.append(transitTimes, planetData["epoch"] + n*planetData["period"])
      n = n+1;

    transitIndex = [];
    bufferRatio = 0.5
    for t in transitTimes:
        # don't get tripped up by transits in gaps
        ii = np.abs(pixelData["time"] - t).argmin()
        if np.abs(pixelData["time"][ii] - t) < bufferRatio*dt: # so we don't pick up one cadence over
            if np.abs(pixelData["time"][ii] - t) > bufferRatio*dt:
                print("got large cadence difference: " + str(pixelData["time"][ii] - t))
            transitIndex = np.append(transitIndex, np.abs(pixelData["time"] - t).argmin())
    transitIndex = transitIndex.astype(int)
    
    return transitTimes, transitIndex


def find_transits(pixelData, planetData, allowedBadCadences = 0):
    transitTimes, transitIndex = find_transit_times(pixelData, planetData)
      
    durationDays = planetData["durationHours"]/24;
    transitAverageDurationDays = 0.9*durationDays/2;
    inTransitIndices = [];
    outTransitIndices = [];
    # Center of the out transit is n cadences + half the duration + half the duration away from the center of the transit
    dt = np.min(np.diff(pixelData["time"])) # days
    outTransitBuffer = dt + durationDays
#    print("allowedBadCadences = " + str(allowedBadCadences))
#    print("durationDays = " + str(durationDays))
#    print("outTransitBuffer = " + str(outTransitBuffer))
#    print("transitAverageDurationDays = " + str(transitAverageDurationDays))
    expectedInTransitLength = np.floor(2*transitAverageDurationDays/dt)
#    print("expected number in transit = ", str(expectedInTransitLength))
    for i in transitIndex:
        thisTransitInIndices = np.argwhere(
            (np.abs(pixelData["time"][i] - pixelData["time"]) < transitAverageDurationDays))
    
        thisTransitOutIndices = np.argwhere(
            (np.abs((pixelData["time"][i] - outTransitBuffer) - pixelData["time"]) < transitAverageDurationDays)
            | (np.abs((pixelData["time"][i] + outTransitBuffer) - pixelData["time"]) < transitAverageDurationDays))
            
#        print(sum(pixelData["quality"][thisTransitInIndices] > 0) + sum(pixelData["quality"][thisTransitOutIndices] > 0))
        if (len(thisTransitInIndices) < expectedInTransitLength) | (len(thisTransitOutIndices) < 2*expectedInTransitLength) | (sum(pixelData["quality"][thisTransitInIndices] > 0) + sum(pixelData["quality"][thisTransitOutIndices] > 0) > allowedBadCadences):
            continue
#        print("transit " + str(i) + ":")
#        print([len(thisTransitInIndices), expectedInTransitLength])
#        print([len(thisTransitOutIndices), 2*expectedInTransitLength])
#        if np.any(pixelData["quality"][thisTransitInIndices] > 0):
#            print("in transit bad quality flags: " + str(pixelData["quality"][thisTransitInIndices]))
#        if np.any(pixelData["quality"][thisTransitOutIndices] > 0):
#            print("out transit bad quality flags: " + str(pixelData["quality"][thisTransitOutIndices]))

        inTransitIndices = np.append(inTransitIndices, thisTransitInIndices[pixelData["quality"][thisTransitInIndices] == 0])
        outTransitIndices = np.append(outTransitIndices, thisTransitOutIndices[pixelData["quality"][thisTransitOutIndices] == 0])

    inTransitIndices = np.array(inTransitIndices).astype(int)
    outTransitIndices = np.array(outTransitIndices).astype(int)

    return inTransitIndices, outTransitIndices, transitIndex

def make_difference_image(pixelData, inTransitIndices, outTransitIndices):
    meanInTransit = np.mean(pixelData["flux"][inTransitIndices,::-1,:], axis=0)
    meanInTransitSigma = np.sqrt(np.mean(pixelData["fluxErr"][inTransitIndices,::-1,:]**2, axis=0)/len(inTransitIndices))
    meanOutTransit = np.mean(pixelData["flux"][outTransitIndices,::-1,:], axis=0)
    meanOutTransitSigma = np.sqrt(np.mean(pixelData["fluxErr"][outTransitIndices,::-1,:]**2, axis=0)/len(outTransitIndices))
    diffImage = meanOutTransit-meanInTransit
    diffImageSigma = (meanOutTransit-meanInTransit)/np.sqrt((meanInTransitSigma/np.sqrt(len(inTransitIndices)))+(meanOutTransitSigma/np.sqrt(len(outTransitIndices))))
    
    diffImageData = {}
    diffImageData["diffImage"] = diffImage
    diffImageData["diffImageSigma"] = diffImageSigma
    diffImageData["meanInTransit"] = meanInTransit
    diffImageData["meanInTransitSigma"] = meanInTransitSigma
    diffImageData["meanOutTransit"] = meanOutTransit
    diffImageData["meanOutTransitSigma"] = meanOutTransitSigma
    return diffImageData
    
flux12 = 3.6e8/(30*60)

def mag2b(mag):
    return (100**(1/5))**(-mag)

def mag2flux(mag):
    return flux12*mag2b(mag)/mag2b(12)

def make_stellar_scene(pixelData, ticData, ticName, dMagThreshold = 4):
    catalogData = {}
    
    # compute mjd for the Gaia epoch J2015.5 = 2015-07-02T21:00:00
    t = Time("2015-07-02T21:00:00", format='isot', scale='utc')
    bjdJ2015p5 = t.jd - 2457000
    bjd = np.mean(pixelData["time"])
    mas2deg = 1/(3600*1000)
    dt = (bjd - bjdJ2015p5)/365
#    print("dt = " + str(dt) + " years")

    searchRadius = (np.linalg.norm([pixelData["flux"].shape[1], pixelData["flux"].shape[2]]))*21/3600/2 # assumes 21 arcsec pixels
    ticCatalog = get_tic(ticData["raDegrees"], ticData["decDegrees"], searchRadius)
#    print(list(ticCatalog))
    dRa = mas2deg*dt*ticCatalog["pmRA"]/np.cos(ticCatalog["Dec_orig"]*np.pi/180)
    dRa[np.isnan(dRa)] = 0
    dDec = mas2deg*dt*ticCatalog["pmDEC"]
    dDec[np.isnan(dDec)] = 0
#    print("mean dRa in arcsec = " + str(3600*np.mean(dRa)))
#    print("mean dDec in arcsec = " + str(3600*np.mean(dDec)))
    ticCatalog["correctedRa"] = ticCatalog["RA_orig"] + dRa
    ticCatalog["correctedDec"] =  ticCatalog["Dec_orig"] + dDec

    targetIndex = np.where(np.array(ticCatalog["ID"]).astype(int)==ticData["id"])[0][0]

    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
    refColPix, refRowPix, scinfo = trdp.tess_stars2px_function_entry(
        ticData["id"], pixelData["referenceRa"], pixelData["referenceDec"], aberrate=True, trySector=pixelData["sector"])
    onPix = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])
    outID = outID[onPix]
    catalogData["refColPix"] = refColPix[onPix]
    catalogData["refRowPix"] = refRowPix[onPix]
#    print([catalogData["refColPix"], catalogData["refRowPix"]])

    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
    targetColPix, targetRowPix, scinfo = trdp.tess_stars2px_function_entry(
        ticData["id"], ticCatalog["correctedRa"][targetIndex], ticCatalog["correctedDec"][targetIndex], aberrate=True, trySector=pixelData["sector"], scInfo=scinfo)
    onPix = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])
    outID = outID[onPix]
    catalogData["targetColPix"] = targetColPix[onPix]
    catalogData["targetRowPix"] = targetRowPix[onPix]
#    print([catalogData["targetColPix"], catalogData["targetRowPix"]])

    ticID, outEclipLong, outEclipLat, outSec, outCam, outCcd, ticColPix, ticRowPix, scinfo \
        = trdp.tess_stars2px_function_entry(
            ticCatalog["ID"], ticCatalog["correctedRa"], ticCatalog["correctedDec"],
            aberrate=True, trySector=pixelData["sector"], scInfo=scinfo)
    theseStars = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])

    separation = 3600*np.sqrt(np.cos(ticCatalog["correctedDec"][targetIndex]*np.pi/180)**2 * (ticCatalog["correctedRa"] - ticCatalog["correctedRa"][targetIndex])**2
                                    + (ticCatalog["correctedDec"] - ticCatalog["correctedDec"][targetIndex])**2)

    catalogData["ticID"] = ticID[theseStars]
    catalogData["ticColPix"] = ticColPix[theseStars]
    catalogData["ticRowPix"] = ticRowPix[theseStars]
    catalogData["separation"] = separation[np.isin(ticCatalog["ID"], catalogData["ticID"])]
    catalogData["ticMag"] = ticCatalog["Tmag"][np.isin(ticCatalog["ID"], catalogData["ticID"])]
    catalogData["ticFlux"] = mag2flux(catalogData["ticMag"])
    catalogData["ticFluxNorm"] = np.sqrt(0.999*catalogData["ticFlux"]/np.max(catalogData["ticFlux"]) + 0.001)
#    print([len(catalogData["ticID"]), len(catalogData["ticMag"])])

#    extent = (pixelData["cornerCol"], pixelData["cornerCol"] + 20, pixelData["cornerRow"], pixelData["cornerRow"] + 20)
#    extent = (pixelData["cornerRow"]-0.5, pixelData["cornerRow"]-0.5 + 20, pixelData["cornerCol"]-0.5, pixelData["cornerCol"]-0.5 + 20)
    catalogData["extent"] = (pixelData["cornerCol"], pixelData["cornerCol"] + pixelData["flux"].shape[1],
        pixelData["cornerRow"], pixelData["cornerRow"] + pixelData["flux"].shape[2])
#    print(pixelData["flux"].shape)
#    print(catalogData["extent"])
    catalogData["dRow"] = catalogData["refRowPix"] - (pixelData["referenceRow"] + catalogData["extent"][2] - 0.5)
    catalogData["dCol"] = catalogData["refColPix"] - (pixelData["referenceCol"] + catalogData["extent"][0] - 0.5)

    closeupOffset = 8
    closeupSize = 5
    catalogData["extentClose"] = (pixelData["cornerCol"] + 8, pixelData["cornerCol"] + 8 + closeupSize,
        pixelData["cornerRow"] + 8, pixelData["cornerRow"]  + 8 + closeupSize)
    
    return catalogData
    
def draw_pix_catalog(pixArray, catalogData, ax=None, close=False, annotate=False, magColorBar=False, pixColorBar=True, pixColorBarLabel=False, filterStars=False, dMagThreshold=4, targetID=None, fs=18, ss=400):
    if ax == None:
        ax = plt.gca()
    if targetID == None:
        targetIndex = 0
    else:
        targetIndex = ((ticCatalog["ID"]).astype(int)==targetID)[0]
    if close:
        ex='extentClose'
        pixArray=pixArray[7:12,8:13]
    else:
        ex='extent'
    im = ax.imshow(pixArray, cmap='jet', extent=catalogData[ex])
    if pixColorBar:
        cbh = plt.colorbar(im, ax=ax)
        cbh.ax.tick_params(labelsize=fs-2)
    if pixColorBarLabel:
        cbh.ax.set_ylabel("Pixel Flux [e$^-$/sec]", fontsize=fs-2)
    if not close:
        ax.plot([catalogData[ex][0], catalogData[ex][1]], [catalogData["targetRowPix"] - catalogData["dRow"],catalogData["targetRowPix"] - catalogData["dRow"]], 'r', alpha = 0.6)
        ax.plot([catalogData["targetColPix"] - catalogData["dCol"],catalogData["targetColPix"] - catalogData["dCol"]], [catalogData[ex][2], catalogData[ex][3]], 'r', alpha = 0.6)
    ax.plot(catalogData["targetColPix"] - catalogData["dCol"], catalogData["targetRowPix"] - catalogData["dRow"], 'm*', zorder=100, ms=fs-2)
    if ss > 0:
        targetMag = catalogData["ticMag"][targetIndex]
        if filterStars:
            idx = (catalogData["ticMag"]-targetMag) < dMagThreshold
        else:
            idx = range(len(catalogData['ticID']))
        star_gs = ax.scatter(catalogData["ticColPix"][idx]  - catalogData["dCol"], catalogData["ticRowPix"][idx] - catalogData["dRow"], cmap='BuGn',
            c=catalogData["ticMag"][idx], s=ss*catalogData["ticFluxNorm"][idx], edgeColors="w", linewidths=0.5, alpha=1)
        if magColorBar:
            cbh2 = plt.colorbar(star_gs, ax=ax)
            cbh2.ax.set_ylabel('T mag', fontsize=fs-2)
            cbh2.ax.tick_params(labelsize=fs-2)
        if annotate:
            bbox = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
            pscale = bbox.width * plt.gcf().dpi
            for s in range(len(catalogData['ticID'])):
                px = catalogData["ticColPix"][s] - catalogData["dCol"]
                py = catalogData["ticRowPix"][s] - catalogData["dRow"]
                ticMag = catalogData["ticMag"][s]
                if ((ticMag-targetMag < dMagThreshold) & (px >= catalogData[ex][0]) & (px <= catalogData[ex][1]) & (py > catalogData[ex][2]) & (py < catalogData[ex][3])):
                    ax.text(px, py + 1*20/pscale, str(s), color="w", fontsize = fs-2, path_effects=[pe.withStroke(linewidth=1,foreground='black')])
    ax.tick_params(axis='both', which='major', labelsize=fs-2)
    ax.set_xlim(catalogData[ex][0], catalogData[ex][1])
    ax.set_ylim(catalogData[ex][2], catalogData[ex][3])

def draw_difference_image(diffImageData, pixelData, ticData, planetData, catalogData, ticName = None, dMagThreshold = 4):

    f = open(ticName + "/ticKey_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".txt", 'w')
    f.write("# index, TIC ID, TMag, separation (arcsec)\n")
    for s, id in enumerate(catalogData["ticID"]):
#        if (catalogData["ticMag"][s]-catalogData["ticMag"][0] < dMagThreshold):
        f.write(str(s) + ", " + str(id) + ", " + str(catalogData["ticMag"][s]) + ", " + str(np.round(catalogData["separation"][s], 3)) + "\n")
    f.close()

    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["diffImage"], catalogData, ax=ax, dMagThreshold = dMagThreshold)
    plt.title("diff image");
    plt.savefig(ticName + "/diffImage_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["diffImage"], catalogData, ax=ax, close=True)
    plt.title("diff image close");
    plt.savefig(ticName + "/diffImageClose_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["diffImageSigma"], catalogData, ax=ax, dMagThreshold = dMagThreshold)
    plt.title("SNR diff image");
    plt.savefig(ticName + "/diffImageSNR_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["diffImageSigma"], catalogData, ax=ax, close=True)
    plt.title("SNR diff image close");
    plt.savefig(ticName + "/diffImageSNRClose_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, ax=ax, magColorBar=True, dMagThreshold = dMagThreshold)
    plt.title("Direct image");
    plt.savefig(ticName + "/directImage_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, ax=ax, annotate=True, magColorBar=True, dMagThreshold = dMagThreshold)
    plt.title("Direct image");
    plt.savefig(ticName + "/directImageAnnotated_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(12,10))
    draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, ax=ax, close=True)
    plt.title("Direct image close");
    plt.savefig(ticName + "/directImageClose_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

def draw_lc_transits(pixelData, planetData, inTransitIndices, outTransitIndices, transitIndex, ticName = None, apCenter = [10,10]):
#    apFlux = pixelData["flux"][:,8:13,8:13]
    apFlux = pixelData["flux"][:,apCenter[0]-2:apCenter[0]+3,apCenter[1]-2:apCenter[1]+3]
    lc = np.sum(np.sum(apFlux,2),1)
    plt.figure(figsize=(15, 5));
    plt.plot(pixelData["time"][pixelData["quality"]==0], lc[pixelData["quality"]==0], label="flux")
    plt.plot(pixelData["time"][pixelData["inOtherTransit"]==1], lc[pixelData["inOtherTransit"]==1], 'y+', ms=20, alpha = 0.6, label="in other transit")
    plt.plot(pixelData["time"][pixelData["quality"]>0], lc[pixelData["quality"]>0], 'rx', ms=10, label="quality problems")
    plt.plot(pixelData["time"][inTransitIndices], lc[inTransitIndices], 'd', ms=10, alpha = 0.6, label="in transit")
    plt.plot(pixelData["time"][outTransitIndices], lc[outTransitIndices], 'o', ms=10, alpha = 0.5, label="out of transit")
    plt.plot(pixelData["time"][transitIndex], lc[transitIndex], 'b*', ms=10, alpha = 1, label="transit center", zorder=100)
    plt.legend()
    plt.xlabel("time");
    plt.ylabel("flux (e-/sec)");
    if ticName != None:
        plt.savefig(ticName + "/lcTransits_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

def get_tic(ra, dec, radiusDegrees):
    return Catalogs.query_region(str(ra) + " " + str(dec), radius=radiusDegrees, catalog="TIC")
