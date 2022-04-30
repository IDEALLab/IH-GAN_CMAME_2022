"""
Utility functions for data processing

Author(s): Wei Chen (wchen459@gmail.com)
"""

import numpy as np


class Normalizer(object):
    
    def __init__(self, data=None, bounds=None):
        assert (data is not None) or (bounds is not None)
        if data is not None:
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            self.bounds = np.vstack((self.min, self.max))
        else:
            self.min = bounds[0]
            self.max = bounds[1]
            self.bounds = bounds
        
    def transform(self, raw_data):
        scaled_data = (raw_data-self.min)/(self.max-self.min)
        scaled_data = 2.*scaled_data - 1.
        return scaled_data
    
    def inverse_transform(self, scaled_data):
        raw_data = (scaled_data + 1.)/2.
        raw_data = raw_data*(self.max-self.min) + self.min
        return raw_data

def preprocess_design_variables(design_variables, normalizer):
    alpha = design_variables[:,:3]
    t = normalizer.transform(design_variables[:,3:])
    return np.hstack((alpha, t))
    
def postprocess_design_variables(design_variables, normalizer):
    alpha = design_variables[:,:3]
    t = normalizer.inverse_transform(design_variables[:,3:])
    return np.hstack((alpha, t))

def preprocess_material_properties(material_properties, normalizer):
    material_properties = normalizer.transform(material_properties)
    return material_properties

def postprocess_material_properties(material_properties, normalizer):
    material_properties = normalizer.inverse_transform(material_properties)
    return material_properties

