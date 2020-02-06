__author__ = 'Brian M Anderson'
# Created on 1/2/2020
import os
import pandas as pd
from scipy.ndimage.measurements import center_of_mass
from Ray_Tracing.Utilities import *
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack


images_path = r'K:\Morfeus\BMAnderson\test\cross'
recurrence_reader = Dicom_to_Imagestack(arg_max=False, Contour_Names=['Test_Ablation','Test_Cross'])
recurrence_reader.Make_Contour_From_directory(images_path)

mask = recurrence_reader.mask
ablation_base = mask[...,1]
cross_base = mask[...,2]

centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_base))
spacing = recurrence_reader.annotation_handle.GetSpacing()
output = create_output_ray(centroid_of_ablation_recurrence, spacing=spacing, ref_binary_image=cross_base,
                           margin_rad=np.deg2rad(0), margin=100)
recurrence_reader.with_annotations(output, output_dir=os.path.join(images_path, 'new_RT'),
                                   ROI_Names=['cone_cross_fixed'])