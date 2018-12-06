#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# class to operate with the spectra of the ZTF domeflat illuminator LEDs.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os
import pandas as pd

# go figure where you put the LED spectra file
package_path        = os.path.dirname(os.path.abspath(__file__))
pkg_led_spec_file   = os.path.join(package_path, 'data', 'LED_spectra_bd2.csv')

# some values for the LED peaks
led_peaks = {
        'led0': 382,
        'led1': 403,
        'led2': 451.4,
        'led3': 479.8,
        'led4': 499.9,
        'led5': 525.9,
        'led6': 541.4,
        'led7': 593.5,
        'led8': 620.7,
        'led9': 633.1,
        'led10': 652.9,
        'led11': 738.9,
        'led12': 833.6,
        'led13': 864.6,
        'led14': 951.5
        }

ztf_bands = {
        'G': [415.6, 543.4],
        'R': [565.9, 714.4],
        'I': [719.5, 924.5]
        }

def wavelength_to_rgb(wavelength, gamma=0.8):
    """
        taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
        This converts a given wavelength of light to an 
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).

        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
        Additionally alpha value set to 0.5 outside range
    """
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


class illumled:
    """
        hold info on one of the illuminator LEDs.
    """


    def __init__(self, led_num, wlen, intensity):
        """
            Parameters:
            -----------
                
                led_num: `int`
                    LED ID number, from 0 to 14.
                
                wlen, intensity: `array-like`
                    wavelengths and intensities for the measured spectra
        """
        
        if led_num not in list(range(15)):
            raise ValueError("invalid LED number. Must be between 0 and 14.")
        
        self.number     = led_num
        self.name       = 'led%d'%self.number
        self.peak_wlen  = led_peaks[self.name]
        
        # assing some sort of wlen dep color
        if led_num == 0:
            col='0.2'
        elif led_num == 1:
            col='0.4'
        elif led_num == 6:
            col='0.8'
        else:
            col=wavelength_to_rgb(self.peak_wlen, 1)
        self.led_color = col
        
        # read intensity
        self.wlen = wlen
        self.intensity = intensity


    def interp_spectra(self):
        """
            interpolate the intensity measurements and create a function.
        """
        
        pass


    def __add__(self, other):
        """
        """
        pass


class illumleds:
    """
        this class will create a set of LEDs objects, one for each of the 15 LEDs
    """
    
    @staticmethod
    def load(led_num=None, led_spec_file=None, **kwargs):
        """
            read the emission spectra for each of the LEDs and
            create the corresponding objects.
            
            Parameters:
            -----------
                
                led_num: `int`, `list` or None
                    ID of the LED you want to load. Must be between 0 and 14. If list
                    or None, the desired (of all, in case of None) LEDs are returned 
                    into a dict.
                
                led_spec_file: `str`
                    path to a csv file containing the spectra for the LEDs.
                    This file should contain, as a first column, the wavelength 
                    in nm, and then, in each of the following columns, the 
                    intensity of each LED emission at that wlength. If None, 
                    use the default provided in the ztftoolbox.data directory.
                    
                kwargs:
                    to be passed to pandas.read_csv, the function used to read the file.
            
            Returns:
            --------
                
                dict of ztfleds objects.
        """
        
        # read the data file
        if led_spec_file is None:
            led_spec_file = pkg_led_spec_file
        led_curves = pd.read_csv(led_spec_file, **kwargs)
        
        # decide which LED you want
        if led_num is None:
            led_ids = list(range(0, 15))
        elif type(led_num) in [float, int]:
            led_ids = [led_num]
        else:
            led_ids = led_num
        
        # read the wlength (same for all the LEDs)
        wlen = led_curves['Wavelength(nm)'].values
        
        # read the intensities for all the desired LEDs and create the objs
        out = {}
        for led_id in led_ids:
            tag = "led%d"%led_id
            led = illumled(led_id, wlen, led_curves[tag].values)
            out[tag] = led
        
        # if you just have one LED, return the object, else the dic
        return out.popitem()[1] if len(out) == 1 else out


