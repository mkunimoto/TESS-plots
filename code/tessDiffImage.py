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

flux12 = 3.6e8/(30*60)

def mag2b(mag):
    return (100**(1/5))**(-mag)

def mag2flux(mag):
    return flux12*mag2b(mag)/mag2b(12)

def get_tic(ra, dec, radiusDegrees):
    return Catalogs.query_region(str(ra) + " " + str(dec), radius=radiusDegrees, catalog="TIC")

class tessDiffImage:
    def __init__(self, ticData, spiceFileLocation = ".", nPixOnSide = 20, dMagThreshold = 4, allowedBadCadences = 0):
        self.ticData = ticData
        self.spiceFileLocation = spiceFileLocation
        self.nPixOnSide = nPixOnSide
        self.dMagThreshold = dMagThreshold
        self.allowedBadCadences = allowedBadCadences
        self.ticName = "tic" + str(self.ticData["id"])

        self.baryCorrector = bc.barycentricCorrection(self.spiceFileLocation)


    def make_ffi_difference_image(self, thisPlanet=None, allowedBadCadences = None):
        if allowedBadCadences == None:
            allowedBadCadences = self.allowedBadCadences
            
        fitsList = self.get_tess_cut()
        for fitsFile in fitsList:
            pixelData = self.get_cadence_data(fitsFile)
            # check that this is the camera and sector that we want here
            if (self.ticData["cam"] != None) & ((self.ticData["sector"] != pixelData["sector"]) | (self.ticData["cam"] != pixelData["camera"])):
                continue

            sectorQflags = np.array([])
            if self.ticData["qualityFiles"] != None:
                for sectorList in self.ticData["qualityFiles"]:
                    if sectorList[0] == pixelData["sector"]:
                        for fname in sectorList[1]:
                            sectorQflags = np.append(sectorQflags, np.loadtxt(fname, usecols=(1)))
            elif self.ticData["qualityFlags"] != None:
                for sectorList in self.ticData["qualityFlags"]:
                    if sectorList[0] == pixelData["sector"]:
                        sectorQflags = sectorList[1]
                
            if len(sectorQflags) == len(pixelData["quality"]):
                pixelData["quality"] = pixelData["quality"] + sectorQflags
            else:
                print("no sector quality flags of the same length as FFI quality flags")

            catalogData = self.make_stellar_scene(pixelData)
            
            if thisPlanet == None:
                for p, planetData in enumerate(self.ticData["planetData"]):
                    planetData["planetIndex"] = p
                    planetData["planetID"] = p
                    self.make_planet_difference_image(self.ticData, planetData, pixelData, catalogData, allowedBadCadences = allowedBadCadences)
            else:
                planetData = self.ticData["planetData"][thisPlanet]
                planetData["planetIndex"] = thisPlanet
                planetData["planetID"] = thisPlanet
                self.make_planet_difference_image(planetData, pixelData, catalogData, allowedBadCadences = allowedBadCadences)

        #       print('rm ' + self.ticName + '/*.fits')
        os.system('rm ' + self.ticName + '/*.fits')

    def make_planet_difference_image(self, planetData, pixelData, catalogData, allowedBadCadences = None):
        if allowedBadCadences == None:
            allowedBadCadences = self.allowedBadCadences
            
        # for each planet, make an array of cadences that are in the transits of other planets
        inOtherTransitIndices = [];
        for pi, otherPlanet in enumerate(self.ticData["planetData"]):
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

        inTransitIndices, outTransitIndices, transitIndex, diffImageData = self.find_transits(pixelData, planetData, allowedBadCadences = allowedBadCadences)
#        diffImageData = self.make_difference_image(pixelData, inTransitIndices, outTransitIndices)
        self.draw_difference_image(diffImageData, pixelData, planetData, catalogData)
        self.draw_lc_transits(pixelData, planetData, inTransitIndices, outTransitIndices, transitIndex)

        f = open(self.ticName + "/imageData_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pickle", 'wb')
        pickle.dump([diffImageData, catalogData, pixelData, inTransitIndices, outTransitIndices, transitIndex, planetData], f, pickle.HIGHEST_PROTOCOL)
        f.close()


    def get_tess_cut(self, fitsNum = 0):
        ticNumber = self.ticData["id"]
        ra = self.ticData["raDegrees"]
        dec = self.ticData["decDegrees"]
        sector = self.ticData["sector"]
        ticName = self.ticName
        if sector == None:
            curlStr = 'curl "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=' \
                        + str(ra) + '&dec=' + str(dec) + '&y=' + str(self.nPixOnSide) + '&x=' + str(self.nPixOnSide) \
                        + '" --output ' + ticName + '.zip'
        else:
            curlStr = 'curl "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=' \
                        + str(ra) + '&dec=' + str(dec) + '&y=' + str(self.nPixOnSide) + '&x=' + str(self.nPixOnSide) + '&sector=' + str(sector) \
                        + '" --output ' + ticName + '.zip'
        print(curlStr)
        os.system(curlStr)
        os.system('mkdir ' + ticName)
        os.system('unzip ' + ticName + '.zip' + ' -d ' + ticName)
        os.system('rm ' + ticName + '.zip')
        fitsList = glob.glob(ticName + '/*.fits')
        print(fitsList)
        return fitsList
        
    def get_cadence_data(self, fitsFile):
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
        tessToJulianOffset = 2457000
        spacecraftTime = pixelData["rawTime"] - pixelData["ffiBarycentricCorrection"]
        pixelData["barycentricCorrection"], pixelData["validBarycentricCorrection"] = self.baryCorrector.computeCorrection(spacecraftTime + tessToJulianOffset,
                                                 pixelData["referenceRa"], pixelData["referenceDec"])
        pixelData["time"] = spacecraftTime + pixelData["barycentricCorrection"]
        if np.any(pixelData["validBarycentricCorrection"] == False):
            print("Some cadences have an invalid barycentric correction")
            
    #    print("time: " + str([np.min(pixelData["time"]), np.max(pixelData["time"])]))
        return pixelData
        
    def find_transit_times(self, pixelData, planetData):
        dt = np.min(np.diff(pixelData["time"]))
        nTransit = np.round((pixelData["time"] - planetData["epoch"])/planetData["period"]).astype(int)
        transitTimes = np.unique(planetData["epoch"] + planetData["period"] * nTransit)
        transitIndex = np.array([np.abs(pixelData["time"] - t).argmin() for t in transitTimes])
        bufferRatio = 0.5
        flagGaps = np.abs(pixelData["time"][transitIndex] - transitTimes) > bufferRatio*dt
        for i in np.nonzero(flagGaps)[0]:
            print("large cadence difference: " + str(pixelData["time"][transitIndex][i] - transitTimes[i]))
        transitTimes = transitTimes[~flagGaps]
        transitIndex = transitIndex[~flagGaps]
        return transitTimes, transitIndex

    def find_transits(self, pixelData, planetData, allowedBadCadences = None):
        if allowedBadCadences == None:
            allowedBadCadences = self.allowedBadCadences
            
        transitTimes, transitIndex = self.find_transit_times(pixelData, planetData)
          
        durationDays = planetData["durationHours"]/24;
        transitAverageDurationDays = 0.9*durationDays/2;
        # Center of the out transit is n cadences + one duration away from the center of the transit
        dt = np.min(np.diff(pixelData["time"])) # days
        outTransitBuffer = dt + durationDays
    #    print("allowedBadCadences = " + str(allowedBadCadences))
    #    print("durationDays = " + str(durationDays))
    #    print("outTransitBuffer = " + str(outTransitBuffer))
    #    print("transitAverageDurationDays = " + str(transitAverageDurationDays))
        expectedInTransitLength = np.floor(2*transitAverageDurationDays/dt)
    #    print("expected number in transit = ", str(expectedInTransitLength))
        inTransitIndices = []
        outTransitIndices = []
        nBadCadences = []
        DiffImageDataList = []
        for i in transitIndex:
            thisTransitInIndices = np.nonzero(
                (np.abs(pixelData["time"][i] - pixelData["time"]) < transitAverageDurationDays))[0]
            thisTransitOutIndices = np.nonzero(
                (np.abs(pixelData["time"][i] - pixelData["time"]) > (outTransitBuffer - transitAverageDurationDays))
                & (np.abs(pixelData["time"][i] - pixelData["time"]) < (outTransitBuffer + transitAverageDurationDays)))[0]
            thisTransitBadCadences = np.sum(pixelData["quality"][thisTransitInIndices] != 0) + np.sum(pixelData["quality"][thisTransitOutIndices] != 0)
    #        print(sum(pixelData["quality"][thisTransitInIndices] > 0) + sum(pixelData["quality"][thisTransitOutIndices] > 0))
            if (len(thisTransitInIndices) < expectedInTransitLength) | (len(thisTransitOutIndices) < 2*expectedInTransitLength):
                continue
    #        print("transit " + str(i) + ":")
    #        print([len(thisTransitInIndices), expectedInTransitLength])
    #        print([len(thisTransitOutIndices), 2*expectedInTransitLength])
    #        if np.any(pixelData["quality"][thisTransitInIndices] > 0):
    #            print("in transit bad quality flags: " + str(pixelData["quality"][thisTransitInIndices]))
    #        if np.any(pixelData["quality"][thisTransitOutIndices] > 0):
    #            print("out transit bad quality flags: " + str(pixelData["quality"][thisTransitOutIndices]))
    
            thisTransitInIndices = thisTransitInIndices[pixelData["quality"][thisTransitInIndices] == 0].tolist()
            thisTransitOutIndices = thisTransitOutIndices[pixelData["quality"][thisTransitOutIndices] == 0].tolist()
            DiffImageDataList.append(self.make_difference_image(pixelData, thisTransitInIndices, thisTransitOutIndices))

            inTransitIndices.append(thisTransitInIndices)
            outTransitIndices.append(thisTransitOutIndices)
            nBadCadences.append(thisTransitBadCadences)
            
            
        alert=False
        if np.min(nBadCadences) > allowedBadCadences:
            print("No good transits based on %i allowed bad cadences; using transit with %i bad cadences." % (allowedBadCadences, np.min(nBadCadences)))
            alert=True
        goodTransits = (nBadCadences <= np.max([allowedBadCadences, np.min(nBadCadences)]))
        
        diffImageData = {}
        nTranitImages = 0
        diffImageData["diffImage"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        diffImageData["diffImageSigma"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        diffImageData["diffSNRImage"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        diffImageData["meanInTransit"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        diffImageData["meanInTransitSigma"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        diffImageData["meanOutTransit"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        diffImageData["meanOutTransitSigma"] = np.zeros(DiffImageDataList[0]["diffImage"].shape)
        for i in range(len(DiffImageDataList)):
            if goodTransits[i]:
                diffImageData["diffImage"] += DiffImageDataList[i]["diffImage"]
                diffImageData["diffImageSigma"] += DiffImageDataList[i]["diffImageSigma"]**2
                diffImageData["meanInTransit"] += DiffImageDataList[i]["meanInTransit"]
                diffImageData["meanInTransitSigma"] += DiffImageDataList[i]["meanInTransitSigma"]**2
                diffImageData["meanOutTransit"] += DiffImageDataList[i]["meanOutTransit"]
                diffImageData["meanOutTransitSigma"] += DiffImageDataList[i]["meanOutTransitSigma"]**2
                nTranitImages += 1
        diffImageData["diffImage"] /= nTranitImages
        diffImageData["diffImageSigma"] = np.sqrt(diffImageData["diffImageSigma"])/nTranitImages
        diffImageData["meanInTransit"] /= nTranitImages
        diffImageData["meanInTransitSigma"] = np.sqrt(diffImageData["meanInTransitSigma"])/nTranitImages
        diffImageData["meanOutTransit"] /= nTranitImages
        diffImageData["meanOutTransitSigma"] = np.sqrt(diffImageData["meanOutTransitSigma"])/nTranitImages
        diffImageData["diffSNRImage"] = diffImageData["diffImage"]/diffImageData["diffImageSigma"]

        
        inTransitIndices = np.unique(sum(np.array(inTransitIndices)[goodTransits].tolist(), []))
        outTransitIndices = np.unique(sum(np.array(outTransitIndices)[goodTransits].tolist(), []))
        planetData["badCadenceAlert"] = alert
        
        return inTransitIndices, outTransitIndices, transitIndex, diffImageData

    def make_difference_image(self, pixelData, inTransitIndices, outTransitIndices):
        meanInTransit = np.mean(pixelData["flux"][inTransitIndices,::-1,:], axis=0)
        meanInTransitSigma = np.sqrt(np.mean(pixelData["fluxErr"][inTransitIndices,::-1,:]**2, axis=0)/len(inTransitIndices))
        meanOutTransit = np.mean(pixelData["flux"][outTransitIndices,::-1,:], axis=0)
        meanOutTransitSigma = np.sqrt(np.mean(pixelData["fluxErr"][outTransitIndices,::-1,:]**2, axis=0)/len(outTransitIndices))
        diffImage = meanOutTransit-meanInTransit
        diffImageSigma = np.sqrt((meanInTransitSigma**2)+(meanOutTransitSigma**2))
        diffSNRImage = diffImage/diffImageSigma

        diffImageData = {}
        diffImageData["diffImage"] = diffImage
        diffImageData["diffImageSigma"] = diffImageSigma
        diffImageData["diffSNRImage"] = diffSNRImage
        diffImageData["meanInTransit"] = meanInTransit
        diffImageData["meanInTransitSigma"] = meanInTransitSigma
        diffImageData["meanOutTransit"] = meanOutTransit
        diffImageData["meanOutTransitSigma"] = meanOutTransitSigma
        return diffImageData
        
    def make_stellar_scene(self, pixelData):
        catalogData = {}
        
        # compute mjd for the Gaia epoch J2015.5 = 2015-07-02T21:00:00
        t = Time("2015-07-02T21:00:00", format='isot', scale='utc')
        bjdJ2015p5 = t.jd - 2457000
        bjd = np.mean(pixelData["time"])
        mas2deg = 1/(3600*1000)
        dt = (bjd - bjdJ2015p5)/365
    #    print("dt = " + str(dt) + " years")

        searchRadius = (np.linalg.norm([pixelData["flux"].shape[1], pixelData["flux"].shape[2]]))*21/3600/2 # assumes 21 arcsec pixels
        ticCatalog = get_tic(self.ticData["raDegrees"], self.ticData["decDegrees"], searchRadius)
    #    print(list(ticCatalog))
        dRa = mas2deg*dt*ticCatalog["pmRA"]/np.cos(ticCatalog["Dec_orig"]*np.pi/180)
        dRa[np.isnan(dRa)] = 0
        dDec = mas2deg*dt*ticCatalog["pmDEC"]
        dDec[np.isnan(dDec)] = 0
    #    print("mean dRa in arcsec = " + str(3600*np.mean(dRa)))
    #    print("mean dDec in arcsec = " + str(3600*np.mean(dDec)))
        ticCatalog["correctedRa"] = ticCatalog["RA_orig"] + dRa
        ticCatalog["correctedDec"] =  ticCatalog["Dec_orig"] + dDec

        targetIndex = np.where(np.array(ticCatalog["ID"]).astype(int)==self.ticData["id"])[0][0]

        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
        refColPix, refRowPix, scinfo = trdp.tess_stars2px_function_entry(
            self.ticData["id"], pixelData["referenceRa"], pixelData["referenceDec"], aberrate=True, trySector=pixelData["sector"])
        onPix = (outSec == pixelData["sector"]) & (outCam == pixelData["camera"]) & (outCcd == pixelData["ccd"])
        outID = outID[onPix]
        catalogData["refColPix"] = refColPix[onPix]
        catalogData["refRowPix"] = refRowPix[onPix]
    #    print([catalogData["refColPix"], catalogData["refRowPix"]])

        outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
        targetColPix, targetRowPix, scinfo = trdp.tess_stars2px_function_entry(
            self.ticData["id"], ticCatalog["correctedRa"][targetIndex], ticCatalog["correctedDec"][targetIndex], aberrate=True, trySector=pixelData["sector"], scInfo=scinfo)
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
        
    def draw_pix_catalog(self, pixArray, catalogData, ax=None, close=False, dMagThreshold = None, annotate=False, magColorBar=False, pixColorBar=True, pixColorBarLabel=False, filterStars=False, targetID=None, fs=18, ss=400):
        if dMagThreshold == None:
            dMagThreshold = self.dMagThreshold
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
                idx = (catalogData["ticMag"]-targetMag) < self.dMagThreshold
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
                    if ((ticMag-targetMag < self.dMagThreshold) & (px >= catalogData[ex][0]) & (px <= catalogData[ex][1]) & (py > catalogData[ex][2]) & (py < catalogData[ex][3])):
                        ax.text(px, py + 1*20/pscale, str(s), color="w", fontsize = fs-2, path_effects=[pe.withStroke(linewidth=1,foreground='black')])
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
        ax.set_xlim(catalogData[ex][0], catalogData[ex][1])
        ax.set_ylim(catalogData[ex][2], catalogData[ex][3])

    def draw_difference_image(self, diffImageData, pixelData, planetData, catalogData, dMagThreshold = None):
        if dMagThreshold == None:
            dMagThreshold = self.dMagThreshold

        f = open(self.ticName + "/ticKey_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".txt", 'w')
        f.write("# index, TIC ID, TMag, separation (arcsec)\n")
        for s, id in enumerate(catalogData["ticID"]):
    #        if (catalogData["ticMag"][s]-catalogData["ticMag"][0] < dMagThreshold):
            f.write(str(s) + ", " + str(id) + ", " + str(catalogData["ticMag"][s]) + ", " + str(np.round(catalogData["separation"][s], 3)) + "\n")
        f.close()
        if planetData["badCadenceAlert"]:
            alertText = ", no good transits!!!"
            alertColor = "r"
        else:
            alertText = ""
            alertColor = "k"
        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffImage"], catalogData, ax=ax, dMagThreshold = dMagThreshold)
        plt.title("diff image" + alertText, color=alertColor);
        plt.savefig(self.ticName + "/diffImage_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffImage"], catalogData, ax=ax, close=True)
        plt.title("diff image close" + alertText, color=alertColor);
        plt.savefig(self.ticName + "/diffImageClose_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffSNRImage"], catalogData, ax=ax, dMagThreshold = dMagThreshold)
        plt.title("SNR diff image" + alertText, color=alertColor);
        plt.savefig(self.ticName + "/diffImageSNR_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["diffSNRImage"], catalogData, ax=ax, close=True)
        plt.title("SNR diff image close" + alertText, color=alertColor);
        plt.savefig(self.ticName + "/diffImageSNRClose_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')
        
        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, ax=ax, magColorBar=True, dMagThreshold = dMagThreshold)
        plt.title("Direct image");
        plt.savefig(self.ticName + "/directImage_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, ax=ax, annotate=True, magColorBar=True, dMagThreshold = dMagThreshold)
        plt.title("Direct image");
        plt.savefig(self.ticName + "/directImageAnnotated_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(12,10))
        self.draw_pix_catalog(diffImageData["meanOutTransit"], catalogData, ax=ax, close=True)
        plt.title("Direct image close");
        plt.savefig(self.ticName + "/directImageClose_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

    def draw_lc_transits(self, pixelData, planetData, inTransitIndices, outTransitIndices, transitIndex, apCenter = [10,10]):
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
        
        plt.savefig(self.ticName + "/lcTransits_planet" + str(planetData["planetID"]) + "_sector" + str(pixelData["sector"]) + "_camera" + str(pixelData["camera"]) + ".pdf",bbox_inches='tight')

