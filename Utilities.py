__author__ = 'Brian M Anderson'
# Created on 12/30/2019
import numpy as np
from skimage import morphology


def cartesian_to_polar(zxy):
    '''
    :param xyz: array of zxy cooridnates
    :return: polar coordinates in the form of: radius, rotation away from the z axis (phi), and rotation from the
    negative y axis (theta)
    '''
    # xyz = np.stack([x, y, z], axis=-1)
    input_shape = zxy.shape
    reshape = False
    if len(input_shape) > 2:
        reshape = True
        zxy = np.reshape(zxy,[np.prod(zxy.shape[:-1]),3])
    polar_points = np.empty(zxy.shape)
    # ptsnew = np.hstack((xyz, np.empty(xyz.shape)))
    xy = (zxy[:,2]**2 + zxy[:,1]**2)
    polar_points[:,0] = np.sqrt(xy + zxy[:,0]**2)
    polar_points[:,1] = np.arctan2(np.sqrt(xy), zxy[:,0])  # for elevation angle defined from Z-axis down, from 0 to pi
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    #  The theta values are just repeats across the z axis, so don't bother with a large matrix conversion
    polar_points[:,2] = np.arctan2(zxy[:,2], zxy[:,1])
    # polar_points[:, 2][polar_points[:, 2] < 0] += 2 * np.pi  # range from 0 to 2 pi
    if reshape:
        polar_points = np.reshape(polar_points,input_shape)
    return polar_points


def polar_to_cartesian(polar):
    '''
    :param polar: in the form of radius, angle away from z axis, and angle from the negative y axis
    :return: z, x, y intensities as differences from the origin
    '''
    cartesian_points = np.empty(polar.shape).astype('float16')
    from_x = polar[:,2].astype('float16')
    xy_plane = (np.sin(polar[:,1])*polar[:,0]).astype('float16')
    cartesian_points[:,0] = (np.cos(polar[:,1])*polar[:,0]).astype('float16')
    cartesian_points[:,1] = (np.cos(from_x)*xy_plane).astype('float16')
    cartesian_points[:,2] = (np.sin(from_x)*xy_plane).astype('float16')
    return cartesian_points.astype('float16')


def create_distance_field(image,origin, spacing=(0.975,0.975,5.0)):
    '''
    :param image:
    :param origin:
    :param spacing:
    :return: polar coordinates, in: [radius, rotation away from the z axis (phi), and rotation from the y axis (theta)]
    '''
    array_of_points = np.transpose(np.asarray(np.where(image==1)),axes=(1,0)).astype('float16')
    spacing_aranged = np.asarray([spacing[2],spacing[0],spacing[1]]).astype('float16')
    differences = (array_of_points - origin)*spacing_aranged
    polar_coordinates = cartesian_to_polar(differences.astype('float64'))
    return polar_coordinates.astype('float16')


def create_output_ray(centroid, ref_binary_image, spacing, margin=100, margin_rad=np.deg2rad(5),
                      target_centroid=None, min_max_only=False):
    labels = morphology.label(ref_binary_image, connectivity=1)  # Could have multiple recurrence sites
    output_ref = np.zeros(ref_binary_image.shape).astype('float16')
    output_ref = np.expand_dims(output_ref, axis=-1).astype('float16')
    if target_centroid is not None:
        output_ref = np.repeat(output_ref, repeats=3, axis=-1).astype('float16')
    else:
        output_ref = np.repeat(output_ref, repeats=2, axis=-1).astype('float16')
    for label_value in range(1, np.max(labels) + 1):
        print('Iterating for mask values {} of {}'.format(label_value,np.max(labels)))
        recurrence = np.zeros(ref_binary_image.shape).astype('float16')
        recurrence[labels == label_value] = 1
        polar_cords = create_distance_field(recurrence, origin=centroid, spacing=spacing)
        polar_cords = np.round(polar_cords, 3).astype('float16')
        polar_cords = polar_cords[:, 1:].astype('float16')
        '''
        We now have the min/max phi/theta for pointing the recurrence_ablation site to the recurrence

        Now, we take those coordinates and see if, with the ablation to minimum ablation site overlap

        Note: This will turn a star shape into a square which encompasses the star!
        '''
        k = np.zeros(output_ref.shape[:-1]).astype('float16')
        int_centroid = [int(i) for i in centroid]
        k[tuple(int_centroid)] = -5
        k[labels == label_value] = polar_cords[...,1]
        # output[..., 1] += define_cone(polar_cords, centroid, ref_binary_image, spacing,
        #                               margin=margin, min_max=min_max,
        #                               margin_rad=margin_rad)
        output_ref[..., 1] += define_cone(polar_cords, centroid, ref_binary_image, spacing, margin=margin,
                                          margin_rad=margin_rad, min_max_only=min_max_only)
        if target_centroid is not None:
            output_ref[..., 2] += define_cone(polar_cords, target_centroid, ref_binary_image, spacing,
                                              margin=margin, margin_rad=margin_rad, min_max_only=min_max_only)
    output_ref[output_ref > 0] = 1
    return output_ref


def define_cone(polar_cords_base, centroid_of_ablation_recurrence, liver_recurrence, spacing, margin=100,
                margin_rad=np.deg2rad(2), min_max_only=False):
    '''
    :param polar_cords_base: polar coordinates from ablation_recurrence centroid to recurrence, come in [phi, theta]
    where theta ranges from 0 to pi and -0 to -pi
    :param centroid_of_ablation_recurrence: centroid of ablation recurrence
    :param liver_recurrence: shape used to make output
    :param margin: how far would you like to look, in mm
    :param margin_rad: degrees of wiggle allowed, recommend 5 degrees (in radians)
    :param min_max_only: should you only worry about min/max? Much faster if True, but a cross turns into a square
    :return:
    '''
    polar_cords_base = polar_cords_base.astype('float16')
    if polar_cords_base.shape[1] == 3:
        polar_cords_base = polar_cords_base[:,1:]
    cone_cords_base_reshaped = create_distance_field(np.ones(liver_recurrence.shape), spacing=spacing,
                                                     origin=centroid_of_ablation_recurrence).astype('float16')
    cone_cords_base_reshaped = np.reshape(cone_cords_base_reshaped, newshape=liver_recurrence.shape+(3,)).astype('float16')
    theta_values = cone_cords_base_reshaped[0,...,2].astype('float16')  # Thetas are just repeated
    output = np.zeros(theta_values.shape).astype('float16')
    '''
    Can reduce size of search by looking at min margin 2D
    '''
    min_margin = np.min(cone_cords_base_reshaped[..., 0], axis=0).astype('float16')
    min_margin_indexes = np.where(min_margin <= margin)
    theta_values = theta_values[min_margin_indexes].astype('float16')
    '''
    Mask where potential theta values are
    '''
    difference = np.min(np.abs(theta_values[:,None] - polar_cords_base[:,1]),axis=-1).astype('float16')
    within_theta_mask = difference <= margin_rad
    output[min_margin_indexes] = within_theta_mask.astype('float16')
    output = np.repeat(output[None,...],liver_recurrence.shape[0],axis=0).astype('float16')
    '''
    Then mask to be within the margin in 3D sense
    '''
    outside_margin_indexes = np.where(cone_cords_base_reshaped[...,0] > margin)
    output[outside_margin_indexes] = 0
    '''
    Now, make sure it falls within phi range
    '''
    mask_indexes = np.where(output == 1)

    phi_values = cone_cords_base_reshaped[mask_indexes][...,1].astype('float16')
    phi_differences = np.abs(phi_values.flatten()[:, None] - polar_cords_base[:, 0]).astype('float16')
    difference = np.min(phi_differences, axis=-1).astype('float16')
    within_phi_mask = difference <= margin_rad
    if not min_max_only:
        theta_values_cone = cone_cords_base_reshaped[mask_indexes][within_phi_mask][...,2].astype('float16')
        theta_differences_in_mask = np.abs(theta_values_cone[:, None] - polar_cords_base[:, 1]).astype('float16')
        del polar_cords_base, theta_values_cone
        phi_differences_in_mask = phi_differences[within_phi_mask].astype('float16')
        summed_difference = np.sqrt(np.square(phi_differences_in_mask).astype('float16') + np.square(theta_differences_in_mask).astype('float16')).astype('float16')
        del phi_differences_in_mask
        min_dif = np.min(summed_difference,axis=1).astype('float16')
        del summed_difference
        within_phi_and_theta = min_dif <= margin_rad
        mask_vals = np.zeros(within_phi_mask.shape).astype('float16')
        mask_vals[within_phi_mask] = within_phi_and_theta.astype('float16')
        output[mask_indexes] = mask_vals.astype('float16')
    else:
        output[mask_indexes] = within_phi_mask.astype('float16')

    return output.astype('float16')


def main():
    pass


if __name__ == '__main__':
    main()
