'''
Created: Saturday Sept. 26 2023 || Modified: 9th Nov 2026
Author: Amit, asingh38@crimson.ua.edu, amitdravid@outlook.com
Note: In all functions considered-
      1. Active rotation (reference frame is fixedm vector is rotated), unless specified to passive
      2. The angles are in radians, unless specified
'''
## This code contaions all the function for all analysis related to texture and grains matching, twinning
import pandas as pd
import numpy as np
import json
import os
import glob
import time
from math import pi, cos, sin, acos, sqrt, tan, degrees
import random

from lib_mathematics import greatest_common_divisor
#% The functions essential for misorientation and grain matching:

##########################################################################
#%% 0. Gerneal functions 
##########################################################################
def repmat(value, size):
    result = []
    for i in range(size):
        result.append(value)
    return result

## 0.2 Rotation related
def rot2mat(rot_data):
    '''
    Input: rot_data = list of rotation/orientation matrix in follwomg order-
              [R11,R12,R13,R21,R22,R23,R31,R32,R33]
    Output: Orientation/rotation in matrix form
    Note: This function is same function as "list_to_matrix_nxn" of function defined in lib_mathematics
    '''
    # Eements of the rotation hasto be in seperate 
    rot = np.array([[rot_data[0],rot_data[1],rot_data[2]],
                    [rot_data[3],rot_data[4],rot_data[5]],
                    [rot_data[6],rot_data[7],rot_data[8]]])
    return rot


## 0.3 Joining matrices
def join_2d_matrices(mat1, mat2):
    result = []
    s11 = len(mat1)
    for k in range(s11):
        temp = []
        for item in mat1[k]:
            temp.append(item)
        for item in mat2[k]:
            temp.append(item)
        result.append(temp)
    return result


##########################################################################
#%% 1. Orientations conversion related functions
##########################################################################
#%%1.1 Quaternions_to_other i.e. quat2X, X could be anyother form 
## 1.1.1
def quat_conj(q): 
    '''
    Input: q = Quternion vector, an array of 4x1 i.e. [q0 q1 q2 q3]
    Output: q_conj = Quternion conjugate i.e. [q0 -q1 -q2 -q3]
    Note: Both input/output are in radinas
    '''
    q = np.asarray(q, dtype=float)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_conj 

## 1.1.2
def quat_inv(q):
    '''
    Input: q = Quternion vector, an array of 4x1 i.e. [q0 q1 q2 q3]
    Output: q_inv = which means exact opposite rotation—rotating by the same angle, 
            but in the opposite direction along the same axis. It negates only the vector part of quternion
            = q_conj/(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    Note: Both input/output are in radinas
    '''
    q = np.asarray(q, dtype=float)
    q_conj = quat_conj(q)
    q_inv = q_conj/np.dot(q, q)
    return q_inv

## 1.1.3 
def quat_rev(quat):
    # reversed quternions i.e. only scalar part is -ve
    # It make 1st term of quternions -ve 
    quat = np.atleast_2d(quat)
    r = np.zeros(quat.shape)
    r[0,0] = -quat[0, 0]
    r[0, 1:] = quat[0, 1:]
    return r

## 1.1.4
def quat_prod(P, Q):
    '''
    Inputs: P and Q are two quternions arrays
    Output: q_prod = normalized quternion product of P and Q
    Ref: https://doi.org/10.1515/9780691211701, Equation 7.1
    Note: Both input/output are in radinas
    '''
    P = np.asarray(P, dtype = float)
    Q = np.asarray(Q, dtype = float)
    p0, p1, p2, p3 = P[0], P[1], P[2], P[3]
    q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]    
    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0    
    r_vec = [r0, r1, r2, r3]
    q_prod = r_vec/np.linalg.norm(r_vec)
    return q_prod

## 1.1.5
def quat_prods(q1, q2):
    ''' 
    Inputs: q1 = a quternion vector
            q2 = either a quat-vcetor or array/list of quternion-vectors (n number of quternions)
    Output: Q =  product quternions between q1 and q2
    Note: Both input/output are in radinas
    '''
    q1 = np.asarray(q1).reshape(-1, 4)[0]
    q2 = np.asarray(q2).reshape(-1, 4)
    Q = np.zeros_like(q2, dtype=float)
    for i in range(len(q2)):
        Q[i, :] = quat_prod(q1, q2[i, :])
    return Q

## 1.1.6
def quat2rod(quat):
    '''
    quat2rod: Convert a quternion to a rodrigues vector
    Input: quat = Quternion
    Output: r = rodrigues vector
    Ref: Introduction to texture analysis (3rd Edition) by Oralf Engler, S. Zaefferer, Randle
    '''
    quat = np.asarray(quat, dtype=float)
    quat = quat/np.linalg.norm(quat)   # ensure unit quaternion
    if np.isclose(quat[0], 0.0):
        raise ValueError("Rotation of 180 degrees: Rodrigues vector is infinite.")
    r = quat[1:] / quat[0]
    # #Old, Numerically less stable, at 0 and 180 degree
    # omega = 2*acos(quat[0])
    # n = quat[1:]/sin(omega/2)
    # r = tan(omega/2)*n
    return r

## 1.1.7
def quat2rot(q): 
    ''' 
    Input: q = a quternion
    Output: rot_mat = rotation matrix (in active rotation form)
    Note: Both input/output are in radinas
    '''
    q = np.asarray(q, dtype = float)
    q_norm = q/np.linalg.norm(q)
    x0, x1, x2, x3 = q_norm[0], q_norm[1], q_norm[2], q_norm[3]
    ## For Active transformation, The reference frame is fixed, vector can rotate
    # calculating rotation matrix component
    r11 = 2*(x0**2 + x1**2)-1
    r12 = 2*(x1*x2 - x0*x3)
    r13 = 2*(x1*x3 + x0*x2)
    r21 = 2*(x1*x2 + x0*x3)
    r22 = 2*(x0**2 + x2**2)-1
    r23 = 2*(x2*x3 - x0*x1)
    r31 = 2*(x1*x3 - x0*x2)
    r32 = 2*(x2*x3 + x0*x1)
    r33 = 2*(x0**2 + x3**2)-1
    # Rotation matrix 
    rot_mat = np.array([[r11, r12, r13],
                        [r21, r22, r23],
                        [r31, r32, r33]])
    ## For passive transformation, (Vector is fixed, reference frame can rotate)
    # rot_mat = rot_mat.T #  T = transpose()
    return rot_mat 
  
## 1.1.8
def quat2axisangle(q, degree=False):
    ''' 
    Input: q- quternion vector, in radians
    Outpot: angle(radians, defalut), axis of the quternion, in same order
    Note: Output, could be chnged to degree
    '''
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    # q and -q represent the same rotation.
    if q[0] < 0:
        q = -q
    qv_norm = np.linalg.norm(q[1:4])
    ## Angle
    angle = 2.0*np.arctan2(qv_norm, q[0])
    if degree:
        angle = np.degrees(angle)
    ## Axis
    if qv_norm < 1e-12:
        axis = None
    else:
        axis = q[1:4]/qv_norm
    return angle, axis 
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%1.2 Rotation-matrix_to_other i.e. rot2X, X could be anyother form 
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## 1.2.1 
def rot2quat(ori_mat):
    '''
    Description: Function to convert rotation matrix to quaternion
    Input: Orientation/rotation matrix
    Output: Normalized quternion
    Ref: doi: 10.1088/0965-0393/23/8/083501
    Note: 1. Both input/output are in radians
          2. This function work, but it is less stable than "rot2quat_scipy"
          So, for most of the calculations "rot2quat_scipy" is used 
    '''
    R = ori_mat
    P = 1 ### P = -1 for passive rotation and P = 1 active rotation (i.e when ref. frame is fixed, but vector can rotate)
    q_0 = 0.5*(1+R[0,0]+R[1,1]+R[2,2])**0.5
    q_1 = P*0.5*(1+R[0,0]-R[1,1]-R[2,2])**0.5
    q_2 = P*0.5*(1-R[0,0]+R[1,1]-R[2,2])**0.5
    q_3 = P*0.5*(1-R[0,0]-R[1,1]+R[2,2])**0.5
    if R[2,1]<R[1,2]:
        q_1 = -q_1
    if R[0,2]<R[2,0]:
        q_2 = -q_2
    if R[1,0]<R[0,1]:
        q_3 = -q_3
    q_vec = [q_0, q_1, q_2, q_3]
    q = q_vec/ np.linalg.norm(q_vec)
    return q
    
## 1.2.2
def rot2quat_scipy(ori_mat):
    '''
    Description: 1.Function to convert rotation matrix to quaternion, scalar-first quaternion [q0, q1, q2, q3]
    Input: Orientation/rotation matrix, should be in "active rotation" 
    Output: Normalized quternion
    Note: 1. Assumes ori_mat represents an active rotation.
         2. Both input/output are in radians
    Ref: doi: 10.1088/0965-0393/23/8/083501
    '''
    # from Scipy, which is equivalent to our "rot2quat" function
    from scipy.spatial.transform import Rotation as ROT
    # ori_mat is rotation matrix
    r = ROT.from_matrix(ori_mat) # This follows active rotation convetion, i.e. frame is fixed vectors are rotated
    q1 = r.as_quat() ## SciPy ordering: [qx, qy, qz, qw]
    # The output quat is already normalized
    # Convert to scalar-first ordering: [qw, qx, qy, qz], because SciPy return the quternion as [qx,qy,qz,qw], qw is scalar or angle
    q = np.array([q1[3], q1[0], q1[1], q1[2]]) # As defult is Active Rotation, 
    # q = quat_conj(q)  ## for passive rotation which can be achieved by using "quat_conj", which makes -ve to the vector part
    q_norm = q/np.linalg.norm(q)
    return q_norm

## 1.2.3
def rot2eulr(rot_mat, rot_sequence="ZXZ", degree = False):
    '''
    This function does Convert the Rotation matrix to Euler-angle
    Input: rot_mat in matrix form or array form of 3X3, should be of active rotation form
    Output: list of Euler-Angles 
    Note: 1. In this conversion the "ZXZ" or "ZYZ" rotation sequence followed
           Default is "ZXZ"
         2. Both input/output are in radians,  not specified
    Ref: NASA report on "Euler-angles, rotation matrix for space shuttle", 1977, pg.23
            https://ntrs.nasa.gov/citations/19770019231
    '''
    import math
    ## Checking whether Given Matrix is ROtation matrix or not
    det_rot_m = np.linalg.det(rot_mat)
    machine_precision = 1e-6
    if abs(det_rot_m -1)< machine_precision:
        # print(f"Matrix is Rotation Matrix with det {det_rot_m:0.3f}")
        rot_mat = rot_mat.T # This transpose(T) is done, bcoz it was found the rot_mat has switched rows-column
        R11, R12, R13 = rot_mat[0,0], rot_mat[0,1], rot_mat[0,2]
        R21, R22, R23 = rot_mat[1,0], rot_mat[1,1], rot_mat[1,2]
        R31, R32, R33 = rot_mat[2,0], rot_mat[2,1], rot_mat[2,2]
        if rot_sequence=="ZXZ":
            #%% Finding Euler-Angles
            phi1 = math.atan(R13/(-R23))
            phi = math.atan(np.sqrt(1-(R33**2))/R33)
            phi2 = math.atan(R31/R32)
            eulr_angle = [phi1, phi, phi2]
            if degree:
                eulr_angle = [phi1*(180/math.pi), phi*(180/math.pi), phi2*(180/math.pi)]
            return eulr_angle 
        elif rot_sequence=="ZYZ": 
            #%% Finding Euler-Angles
            phi1 = math.atan(R23/(R13))
            phi = math.atan(np.sqrt(1-(R33**2))/R33)
            phi2 = math.atan(R32/(-R31))
            eulr_angle = [phi1, phi, phi2]
            if degree:
                eulr_angle = [phi1*(180/math.pi), phi*(180/math.pi), phi2*(180/math.pi)]
            return eulr_angle 
        else:
            raise ValueError(f"The Rotation sequence should be either 'ZXZ' or 'ZYZ'")
    else: 
        raise ValueError(f"Opps! Matrix is not Rotation matrix!:")
    return None

#1.2.4
def rot2axisangle(rot, degree=False):
    '''
    Input = Rotation matrix
    Output: Angle (radian, default), axis
    Note: Axis is already normalized not need to normalize it since, rottation matrix itself is orthogonal
    '''
    # Angle of rotation
    R11, R22, R33 = rot[0,0],rot[1,1], rot[2,2]
    angle = acos(((R11+R22+R33)-1)/2)
    if degree:
        angle = degrees(angle)
    # Axis of rotation
    R12, R21 = rot[0,1], rot[1,0]
    R23, R32 = rot[1,2], rot[2,1]
    R13, R31 = rot[0,2], rot[2,0]
    n1 = (R23-R32)/(2*sin(angle))
    n2 = (R31-R13)/(2*sin(angle))
    n3 = (R12-R21)/(2*sin(angle))
    axis = np.array([n1,n2,n3])
    axis = np.linalg.norm(axis)
    return angle, axis 

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%1.3 Axis-angle_to_other i.e. axisangle2X, X could be anyother form 
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## 1.3.1
def axisangle2rot(angle, axis):
    '''
    Input: axis =  input axis as list or in array as [r1,r2,r3]
           angle = input angle in radian
    output: Rotation matrix, in Active rotation form
    Note: Both input/output are in radians
    Ref: Axis-angle to Rot from Texture analysis by Randle, ISBN 9781420063653
        https://euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    '''
    axis = np.asarray(axis, dtype=float)
    n_axis = np.linalg.norm(axis) # normalizing the axis vector
    r1, r2, r3 = axis[0]/n_axis, axis[1]/n_axis, axis[2]/n_axis # normalized axis
    c, s = cos(angle), sin(angle) # sin and cosine
    # Rotation matrix, in active form
    rot = np.array([[c+(1-c)*r1**2, r1*r2*(1-c)-r3*s, r1*r3*(1-c)+r2*s],
                    [r1*r2*(1-c)+r3*s, c+(1-c)*r2**2, r2*r3*(1-c)-r1*s],
                    [r1*r3*(1-c)-r2*s, r2*r3*(1-c)+r1*s, c+(1-c)*r3**2]])

    ## Alternative: from Rodrigues formula
    # K = np.array([[0, -r3, r2],
    #               [r3, 0, -r1],
    #               [-r2, r1, 0]])
    # rot = np.eye(3) + s*K +(1-c)*np.dot(K,K)
    return rot

## 1.3.2    
def axisangle2quat(angle, axis):
    '''
    Inputs: angle = a scalar
            axis = Carterian vector of 3-elements
            The outputs are in same order i.e. (angle, axis)
    Output: normalized quternion
    Note: Both input/output are in radians
    '''
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    halfangle = 0.5*angle
    q_vec = np.array([cos(halfangle),
            axis[0]*sin(halfangle),
            axis[1]*sin(halfangle),
            axis[2]*sin(halfangle)])
    q_norm = q_vec/ np.linalg.norm(q_vec)
    return q_norm

## 1.3.3
def axisangle2rod(angle, axis):
    '''
    Input = angle (in radian), and axis 
    Output = Rodrigues vector
    Note: 1. Both input/output are in radians

    '''
    rodvec = axis*tan(angle/2)
    return rodvec

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%1.4 Euler_to_other i.e. eulr2X, X could be anyother form 
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## 1.4.1
def eulr2quat(E):
    '''
    Input: Euler-angles(phi1, phi, phi2) in 'radians' and in ZXZ' rotation order
    Output: Quternion
    Note: 1. Both input/output are in radians
        2. There are total 12 possible sequences such as XYZ, YXZ,ZYX.. etc, but 
           Here ZXZ was selected because of texture sequences during measurements we flollow ZXZ order,
    Ref: Euler-Quaternions and Transformation Matrices, by D.M. Henderson, NASA Shuttle Program, July 1977 
    '''
    phi1, phi, phi2 = E[0], E[1], E[2]
    q_0 = cos(phi/2)*cos(phi1/2)*cos(phi2/2) - cos(phi/2)*sin(phi1/2)*sin(phi2/2)
    q_1 = sin(phi/2)*cos(phi1/2)*cos(phi2/2) + sin(phi/2)*sin(phi1/2)*sin(phi2/2)
    q_2 = sin(phi/2)*sin(phi1/2)*cos(phi2/2) - sin(phi/2)*cos(phi1/2)*sin(phi2/2)
    q_3 = cos(phi/2)*sin(phi1/2)*cos(phi2/2) + cos(phi/2)*cos(phi1/2)*sin(phi2/2)
    q_q = [q_0,q_1,q_2,q_3]
    eurquat = q_q/np.linalg.norm(q_q)
    return eurquat

## 1.4.2
def eulr2rot(E, rot_type="crystal2sample"):
    '''
    Input: Euler-angles(phi1, phi, phi2) in ZXZ' rotation order [phi1, phi, phi2]
           rot_type = the rotation type either "crystal2sample" or "sample2crystal" i.e. active or passive respectively
    Output: eurot = Orientation/Rotation matrix
    Note: 1. Euler-angles must be in "ZXZ" rotation order and follow Euler-Bunge notation
         2. Both input/output are in radians
    Ref: Introduction to Texture by Engler, Zafefferer, Randle, 2024
    '''
    ## For rotation from Sample to Crystal frame, is Passive-form of raotation
    phi1, phi, phi2 = E[0], E[1], E[2]
    E_phi1 = np.array([[cos(phi1), sin(phi1), 0], 
                       [-sin(phi1), cos(phi1), 0], 
                       [0, 0, 1]])
    E_phi = np.array([[1, 0, 0], 
                      [0, cos(phi), sin(phi)], 
                      [0, -sin(phi), cos(phi)]])
    E_phi2 = np.array([[cos(phi2), sin(phi2), 0], 
                       [-sin(phi2), cos(phi2), 0], 
                       [0, 0, 1]])
    eurot = np.matmul(E_phi2, np.matmul(E_phi,E_phi1))
    ## For rotation from Crystal to Sample frame, is Active-form of rotation
    if rot_type=="crystal2sample":
        eurot = eurot.T # T = transpose()
    elif rot_type != "sample2crystal":
        raise ValueError("rot_type must be 'crystal2sample' or 'sample2crystal'")
    return eurot

##1.4.3
def eulr2rod(euler_angles):
    '''
    This function conver the Euler-angles (format-ZXZ) to Rodrigues vectors
    Input: List or array of euler_angles in ZXZ' rotation order [phi1, phi, phi2]
    Output: Rodrigues vectors in list
    Note::1. There is no need to normalize the Rodrigues vectors
          2. Both input/output are in radians
    '''
    # Converting Euler-angles from degree to radian
    phi1, phi, phi2 = euler_angles[0], euler_angles[1], euler_angles[2]
    # Rodrigues vector calculations
    r1 = tan(phi/2)*(cos((phi1-phi2)/2)/cos((phi1+phi2)/2))
    r2 = tan(phi/2)*(sin((phi1-phi2)/2)/cos((phi1+phi2)/2))
    r3 = tan((phi1+phi2)/2)
    rod_vec = [r1,r2,r3]
    return rod_vec
## alternative of 1.4.3
def eulr2rod_alt(euler_angles):
    '''
    Input = List or array of euler_angles in ZXZ' rotation order [phi1, phi, phi2]
    Output: Rodrigues vector
    Note:1.  Both input/output are in radians
         2. Axis is already normalized not need to normalize it since, rottation matrix itself is orthogonal
    '''
    rotmat = eulr2rot(euler_angles)
    angle, axis = rot2axisangle(rotmat)
    rod_vec = axisangle2rod(angle, axis)
    return rod_vec

#1.4.4
def eulr2axisangle(euler_angles, degree = False):
    '''
    Input = List or array of euler_angles in ZXZ' rotation order [phi1, phi, phi2], should be in radians
    Output: Angle (in radian, defult), axis [array of three]
    Note: 1. Axis is already normalized not need to normalize it since, rottation matrix itself is orthogonal
         2. Both input is in radians
         3. Output could be in degree, if specified
    '''
    rotmat = eulr2rot(euler_angles)
    angle, axis = rot2axisangle(rotmat)
    if degree: 
        angle = np.degrees(angle)
    return angle, axis

##########################################################################
#%% 2. Hexagonal symmetry & fundamnetal region related functions
##########################################################################
## 2.1 Hexagonal symmetry
def HexSym(rot_type='quat'):
    # The orientaition should be in the "quat" form, Taken from as per the Prof. Dawson work through Dr. Kasemer/Dr. Jun
    p3 = pi/3
    p6 = pi/6
    ci = [cos(p6*x) for x in range(6)]
    si = [sin(p6*x) for x in range(6)]
    z6 = repmat(0, 6)
    w6 = repmat(1, 6)
    pi6 = repmat(pi, 6)
    ## Symmetry
    sixfold = [[p3*x for x in range(6)], z6, z6, w6]
    twofold = [pi6, ci, si, z6]    
    AngleAxis = join_2d_matrices(sixfold, twofold)
    Angle = AngleAxis[0]
    Axis = AngleAxis[1:]
    ## Estimating Symmetry operators
    Q = []
    for i, angle in enumerate(Angle):
        axis = [Axis[j][i] for j in range(3)]
        Q.append(axisangle2quat(angle, axis))
    symm = np.array(Q)
    return symm
HEXSYM = HexSym()

## 2.1 Hexagonal fundamental region
def Hexa_funda_region(q, sym_operators=HEXSYM):
    q = np.asarray(q, dtype=float)
    equiv_quats = quat_prods(q, sym_operators)
    imax = np.argmax(np.abs(equiv_quats[:, 0]))
    quat = equiv_quats[imax].copy()
    quat = quat/np.linalg.norm(quat)
    if quat[0] < 0:
        quat = -quat
    return quat

##########################################################################
#%% 3. Misorientation function: angle, axis and deviation check
##########################################################################
## 3.1
# def mis_ori(q1, q2, sym_operators=HEXSYM, degree=True):
#     qp = quat_prods(quat_rev(q1), q2)
#     mis = Hexa_funda_region(qp, sym_operators=HEXSYM)
#     angle = 2*acos(min(1, mis[0]))
#     if degree:
#         angle *= 180/pi
#     return angle

def mis_ori(q1, q2, sym_operators=HEXSYM, degree=False):
    '''
    mis_ori = Do calculate the misorientation between the two input quternions
    Input: q1, q2 = two qouernions
           sym_operators = HEXSYM hexagonal symmetry operators 12x4 set of quternions
    Output: mis_orientation-angle, axis, best relative quternions between q1, q2,  varinat(sym operator) indices
    The order of out put same as above
    Note: Both input/output are in radians
    '''
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    mis_angle = np.inf
    mis_axis = None
    best_qrel = None
    mis_indices = None
    # Generate all symmetry-equivalent orientations
    for i, sp in enumerate(sym_operators):
        q1_sym = quat_prod(q1, sp)
        for j, st in enumerate(sym_operators):
            q2_sym = quat_prod(q2, st)
            # Relative orientation
            q_mis = quat_prod(quat_inv(q1_sym), q2_sym)
            # Extract angle and axis
            angle, axis = quat2axisangle(q_mis)
            # Retain the minimum-angle representation
            if angle < mis_angle:
                mis_angle = angle
                mis_axis = axis
                best_qrel = q_mis
                mis_indices = (i, j)
    if degree:
        mis_angle = np.degrees(mis_angle)
    return mis_angle, mis_axis, best_qrel, mis_indices

##%% 3.2 Calculating measured misorientation axis-off and angle off
def misori_deviation_check(q_p, q_t, q_th, sym=HEXSYM, degree=False):
    '''
    Description: This calculate the deviation in angle and axis between two grains, compare to their theoretical misornetation
    Inputs: q_p, q_t, q_th = Quternion of grain-1(parent), grain-2 (twin), and theoretical (between parent/twin)
            sym = HEXSYM, hexgonal symmetry operators
            degree = False, if one want to convert to degree, then "True"
    Output: d_theta = misorientation angular-devaition bwteen grain-1/grain-2 and theoretical
            d_alpha = misorientation axis-devaition between grain-1/grain-2 and theoretical
    '''
    q_rel = quat_prod(quat_inv(q_p), q_t) #Projected in parent crystal
    angle_meas, axis_meas = quat2axisangle(q_rel)    
    best = None
    for i, sp in enumerate(sym):
        for j, st in enumerate(sym):
            q_tgt = quat_prod(quat_prod(quat_inv(sp), q_th), st)
            d_rot, _ = quat2axisangle(quat_prod(quat_inv(q_tgt), q_rel))
            if best is None or d_rot < best["delta_rotation"] - 1e-10:
                th_t, ax_t = quat2axisangle(q_tgt)
                if axis_meas is None or ax_t is None:
                    d_alpha = np.nan
                else:
                    d_alpha = np.arccos(np.clip(abs(axis_meas @ ax_t), 0.0, 1.0))
                best = dict(delta_theta=abs(angle_meas - th_t),
                            delta_alpha=d_alpha,
                            delta_rotation=d_rot,
                            theta_target=th_t, axis_target=ax_t, ij=(i, j))
    if degree:   # convert only angular quantities, once, at the end
        for k in ("delta_theta", "delta_alpha", "delta_rotation", "theta_target"):
            best[k] = np.degrees(best[k])
        angle_meas = np.degrees(angle_meas)
    return angle_meas, axis_meas, best

# def misori_deviation(q_p, q_t, q_th, sym=HEXSYM,degree=False):
#     '''
#     Description: This calculate the deviation in angle and axis between two grains, compare to their theoretical misornetation
#     Inputs: q_p, q_t, q_th = Quternion of grain-1(parent), grain-2 (twin), and theoretical (between parent/twin)
#             sym = HEXSYM, hexgonal symmetry operators
#             degree = False, if one want to convert to degree, then "True"
#     Output: d_theta = misorientation angular-devaition bwteen grain-1/grain-2 and theoretical
#             d_alpha = misorientation axis-devaition between grain-1/grain-2 and theoretical
#     Note: 1. This function is ulternative to "misori_deviation_check"
#                 2. It is based on "mis_ori" function 
#                 3. It is less numerical stable than "misori_deviation_check"
#     '''
#     th_m, ax_m, _, _ = mis_ori(q_p, q_t, sym)                      
#     th_t, ax_t, _, _ = mis_ori(np.array([1., 0., 0., 0.]), q_th, sym) # reduced theory
#     d_theta = abs(th_m - th_t)
#     if ax_m is None or ax_t is None:
#         d_alpha = np.nan
#     else:
#         d_alpha = np.pi / 2
#         for S in sym:
#             a_eq = quat2rot(S) @ ax_t
#             d_alpha = min(d_alpha,np.arccos(np.clip(abs(ax_m @ a_eq), 0.0, 1.0)))
#     if degree:
#         d_theta, d_alpha = np.degrees(d_theta), np.degrees(d_alpha)
#     return d_theta, d_alpha

##########################################################################
#%% 4. Twin orientations axis and angle:: still not completeed
##########################################################################
## Note: The following code has certain issue in the function, since it takes 
##       rodrigues vector in 3*1 form, which does not fit with previous codes
def rod2quat(rod_vec):
    # Taken form Dr. Kasemer
    # Calculate the cosine of half the rotation angle
    sqr = np.sum(rod_vec**2, axis=0)
    cosphiby2 = np.cos(np.arctan(np.sqrt(sqr)))
    # Construct the quaternion
    q = np.vstack((cosphiby2, cosphiby2 * rod_vec))
    quat = q/np.linalg.norm(q)
    return quat


def twinori(rod_vec):
    quat = rod2quat(rod_vec)
    print(quat.tolist())
    ori = Hexa_funda_region(quat, sym_operators=HEXSYM)

    return ori

##########################################################################
#%% 5. Plot of Rodrigues-Frank hexgonal fundametal region:: nOt worked
##########################################################################
def plot_rodrigues_hcp_funda_region(r=1,n_points=13, fs=(4,4)):
    """
    #Estimating vectors and plotting Hexagonal Rodrigues-Frank Fundamental region
    Parameters:
    r (float): Radius of the region (default: 1)
    n_points (int): Number of points to plot (default: 13)
    fs: figuresize(4,4)
    
    Returns:
    matplotlib.figure.Figure: The figure containing the hexgonal fundamental region (in Rodrigues form) plot
    """
    import matplotlib.pyplot as plt # Importing library
    theta = pi/12
    R1,R2,R3  = [], [],[]
    for i in range(n_points):
        R1.append(r*sin(2*theta*i))
        R2.append(r*cos(2*theta*i))
        R3.append(r*tan(theta)*((i+1)-i))
    R4 = -1*np.array(R3)
    #----------------------------------------------------------------
    # Constructing RF sapce
    fig = plt.figure(figsize = fs)
    ax = fig.add_subplot(projection ='3d')
    ax.grid(False)
    # Plot the upper and lower hexagons
    ax.plot3D(R1, R2, R3, c = 'gray')
    ax.plot3D(R1,R2,R4, c = 'gray')
    # Plot the vertical lines connecting upper and lower hexagons
    for i in range(13):
        x = [R1[i], R1[i]]
        y = [R2[i], R2[i]]
        z = [R3[i], R4[i]]
        ax.plot3D(x,y,z, c = 'gray')
    ax.set_aspect('equal')
    ax.set_axis_off()
    return fig, ax

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%Part-6: Conversion of planes-direction from Miller-Bravais(4-index) to Miller (3-index)
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##5.1  Direction conversion from 4-index to 3-index
def miller_bravais_to_miller_direction(four_index_array):
    '''
    Input: 4-indicies should be enter either list or in array
    Output: List of 3-indicies
    Ref: Section 5.3, Structure of Materials by Marc de Graef and M. McHenry
    '''
    U,V,T,W = four_index_array[0], four_index_array[1], four_index_array[2], four_index_array[3]
    if U + V + T != 0:
        raise ValueError("Invalid Miller-Bravais direction: U+V+T must be 0")
    u = 2*U+V
    v = 2*V+U
    w = W
    uvw = [u,v,w]
    gcd = abs(uvw[0]) # assummed greatest_common_divisor
    for num in uvw[1:]:
        gcd = greatest_common_divisor(gcd, abs(num))
    if gcd > 0:
        vec = [val//gcd for val in uvw] # "//" integer divsion and "/" float divison
    else: 
        vec = uvw
    return vec
##6.2  Plane conversion from 4-index to 3-index
def miller_bravais_to_miller_plane(four_index_array):
    '''
    Input: 4-indicies should be enter either list or in array
    Output: List of 3-indicies
    Ref: Section 5.3, Structure of Materials by Marc de Graef and M. McHenry
    '''
    H,K,I,L = four_index_array[0], four_index_array[1], four_index_array[2], four_index_array[3]
    h = H
    k = K
    l = L
    return [h,k,l]

##6.3  Direction conversion from 3-index to 4-index
def miller_to_miller_bravais_direction(three_index_array):
    '''
    Input: 3-indicies should be enter either list or in array
    Output: List of 4-indicies
    Ref: 1 Section 5.3, Structure of Materials by Marc de Graef and M. McHenry
    2. Appendix (I), Introduction to texture analysis by O. Engler and Valerie Randle
    '''
    u,v,w = three_index_array[0], three_index_array[1], three_index_array[2]
    # U = (2*u-v)/3
    # V = (2*v-u)/3
    # T = -(U+V)
    # W = w
    # UVTW = [U * 3, V * 3, T * 3, W * 3] 
    # or
    UVTW = [int(2*u - v), int(2*v - u), int(-(u + v)), int(3* w)]
    gcd = abs(UVTW[0]) # assummed greatest_common_divisor
    for num in UVTW[1:]:
        gcd = greatest_common_divisor(gcd, abs(num))
    if gcd>0:
        vec = [val//gcd for val in UVTW] # "//" integer divsion and "/" float divison
    else:
        vec = [UVTW]
    return vec
##6.4  Plane conversion from 3-index to 4-index
def miller_to_miller_bravais_plane(three_index_array):
    '''
    Input: 3-indicies should be enter either list or in array
    Output: List of 4-indicies
    Ref: Section 5.3, Structure of Materials by Marc de Graef and M. McHenry
    '''
    h,k,l = three_index_array[0], three_index_array[1], three_index_array[2]
    H = h
    K = k
    I = -(H+K)
    L = l
    return [H,K,I,L]

##6.5 Cartesian crystal lattice
# def cartesian_crystal_lattice(a,b,c,alpha,beta,gamma):
#     '''
#     Input: lattic parameters in order as a,b,c (in Angstrom)
#             alpha, beta, gamma are angles in (Radians)
#     output: Certesian crystal lattice, a 3X3 matrix, a  transformation matrix
#     Ref: Introduction to Texture analysis by Olaf Engler, Stefan Zaefferer and Valerie Randle
#     Warning!! We careful while using this function for less-symmetric crystals such as Triclinic, Hexagonal, Monoclinic, Rhombohedral(Trigonal),
#          Because, check for cartesian-frame alignement whether x-axis||a (a1) or y-axis||b (a2)
#          This function is for when Cartesian-frame's x-axis||a, or [100]
#     '''
#     ##
#     l11, l12, l13 = a, b*np.cos(gamma), c*np.cos(beta) #  cosine of [100] to a, b, c axis of crystal, a|| [100]
#     l21, l22, l23 = 0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/(np.sin(gamma)) # cosine of lattice a,b,c along crystal [010] axis
#     l31, l32, l33 = 0, 0, c*np.sqrt(1+2*(np.cos(alpha)*np.cos(beta)*np.cos(gamma))-((np.cos(alpha))**2 +(np.cos(beta))**2 +(np.cos(gamma))**2))/(np.sin(gamma))
#     crystal_lattice = np.array([[l11, l12, l13],
#                                [l21, l22, l23],
#                                [l31, l32, l33]])
#     return crystal_lattice

#The catesian matrix for HCP lattice if y||a2, and z||c
def cartesian_align_hcp_crystal(a,covera,crystal_align_to="a2||y"):
    '''
    Construct the direct-lattice basis matrix for HCP crystal.
    a      : basal lattice parameter
    covera : c/a ratio
    crystal_align_to = "a2||y" (default)
    Columns of M are the Cartesian lattice basis vectors.
    '''
    # IF a1||x and c||z
    if crystal_align_to == "a1||x": 
        crystal_lattice = a*np.array([[1,-1/2, 0],
                     [0,np.sqrt(3)/2,0],
                     [0,0,covera]])
    # IF a2||y and c||z, HEDM at APS follows this.
    elif crystal_align_to == "a2||y":
        crystal_lattice = a*np.array([[np.sqrt(3)/2, 0, 0],
                     [-1/2,1,0],
                     [0,0,covera]])
    else:
        raise ValueError(f"Opps!align_type must be 'a1||x' or 'a2||y'")
    return crystal_lattice

#%%6.6
def calc_theoretical_misori_quat_hcp(misori_angle, misori_axis, lattice_a, covera, crystal_align_to="a2||y", degree = True):
    '''
    This function calculate the "theoretical" quternion of given misorientation axis/angle for HCP strcuture
    Inputs: misori_angle = angle of misorientation (in degree)
            misori_axis = axis of misorientation, in 4-index notation, i.e. array of 4 elements e.g. [1,0,-1,0] ot [1,1,-2,0]
            lattice_a = unit lattice of HCP structure of considred material
            covera = c/a ration of HCP lattice of considred material
            crystal_align_to = It tells cartesian-frame alignemnet to HCP-crystal frame
    Output: q_th = Theoretical quternion for given angle/axis
    '''
    ## Axis 
    #First: Conversion of HCP misori axis from 4-indices to 3
    mis_axis_vec = miller_bravais_to_miller_direction(misori_axis)
    ## Aligning as per the cartesian-frame
    crystal_mat = cartesian_align_hcp_crystal(lattice_a,covera,crystal_align_to)
    mis_axis_m = np.matmul(crystal_mat, mis_axis_vec)
    mis_axis_th = mis_axis_m /np.linalg.norm(mis_axis_m)
    ## Angle
    if degree:
        misori_angle_th = np.radians(misori_angle)
    else:
        misori_angle_th = misori_angle
    q_th = axisangle2quat(misori_angle_th,mis_axis_th)
    q_th = q_th/np.linalg.norm(q_th)
    return q_th
    
