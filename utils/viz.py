import numpy as np 
import matplotlib
import seaborn as sns 

class viz:
    '''Define the default visualize configure
    '''
    dBlue   = np.array([ 56,  56, 107]) / 255
    Blue    = np.array([ 46, 107, 149]) / 255
    lBlue   = np.array([241, 247, 248]) / 255
    
    dRed    = np.array([108,  14,  17]) / 255
    Red     = np.array([199, 111, 132]) / 255
    lRed    = np.array([253, 237, 237]) / 255

    dYellow = np.array([129, 119,  14]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    lYellow2= np.array([166, 201, 222]) / 255
    lYellow = np.array([252, 246, 238]) / 255
    
    Green   = .95 * np.array([  0, 135, 149]) / 255
    Purple  = .95 * np.array([108,  92, 231]) / 255
    Palette = [Blue, Red, Green, Yellow, Purple]
    Greens  = [np.array([8,154,133]) / 255, np.array([118,193,202]) / 255] 
    dpi     = 200
    sfz, mfz, lfz = 11, 13, 16
    lw, mz  = 2.5, 6.5

    BluesMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizBlues',   [lBlue, Blue])
    RedsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizReds',    [lRed, dRed])
    YellowsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'vizYellows', [lYellow, Yellow])

    @staticmethod
    def get_style(): 
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
        
viz.get_style()