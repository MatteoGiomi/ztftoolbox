import numpy as np

median = np.median

def combine_monochrome_flats(monochrome, led_weights):
    """
        Parameters:
        -----------
            
            monochrome: `list`
                list of 2D array in which each element is a flat image taken 
                with a single LED.
            
            led_weights: `array-like`
                Intensities of the different LEDs in each band. There should be N-1 
                intensities where N is the len of monochrom, since we are only 
                interested in relative intensities. 
            
    """
    if len(x) == 4:
        wav = weights[0]*x[0] + weights[1]*x[1] + weights[2]*x[2] + weights[3]*x[3]
    elif len(x) == 3:
        wav = weights[0]*x[0] + weights[1]*x[1] + weights[2]*x[2]
    # renormalize
    wav = wav / median(wav)
    return wav




def objective_func(pars, x, data=None, progress=None):
    """
        Parameters:
        -----------
            
            pars: `lmfit.Parameters()`
                parameters of the model. Must contain the intensities 
                of the different LEDs in each band. There should be N-1 weights, 
                since we are only interested in relative intensities. 
            
            x: `list`
                list of 2D array in which each element is a flat image taken 
                with a single LED.
            
            data: `np.ndarray` or None
                target image you want to reproduce. If None, the result would
                be the model evaluated for the given parameters.
            
            progress: `dict` or None
                dictionary storing the values of the parameters and cost function
                at each fit iteration. This dict must have a `w%d` key for each weight
                and a key `cost` containing the value of the cost function. The type
                of all the corresponding values is `list`
    """
    
    # unpack parameters: extract value attribute for each parameter
    parvals = pars.valuesdict()
    weights = [parvals['w%d'%iled] for iled in range(len(x))]
    
    # combine the monochromatic flats according to weights (normalize to mean)
    if len(x) == 4:
        wav = weights[0]*x[0] + weights[1]*x[1] + weights[2]*x[2] + weights[3]*x[3]
    elif len(x) == 3:
        wav = weights[0]*x[0] + weights[1]*x[1] + weights[2]*x[2]
    # renormalize
    wav = wav / median(wav)

