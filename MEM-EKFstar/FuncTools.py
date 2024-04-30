import numpy as np
import matplotlib.pyplot as plt

def plot_extent(ellipse, line_style, color, line_width):
    """
    PLOT_EXTENT plots the extent of an ellipse or circle
    Input:
           ellipse1,    1x5, parameterization of one ellispe [m1 m2 alpha l1 l2]
           line_style,  defined as in the Matlab plot function
           color,       defined as in the Matlab plot function
           line_width,  defined as in the Matlab plot function

    Output:
           handle_extent, the handle of the plot
    """
    center = ellipse[0:2]
    theta = ellipse[2]
    l = ellipse[3:5]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #rotation matrix

    alpha = np.arange(0, 2*np.pi+np.pi/100, np.pi/100)
    xunit = l[0]*np.cos(alpha)
    yunit = l[1]*np.sin(alpha)

    rotated = np.dot(R, np.vstack([xunit, yunit]))
    xpoints = rotated[0,:] + center[0]
    ypoints = rotated[1,:] + center[1]

    handle_extent = plt.plot(xpoints, ypoints, linestyle=line_style, color=color, linewidth=line_width)

    return handle_extent

def measurement_update(y, H, r, p, Cr, Cp, Ch, Cv):
    nk = y.shape[1] # number of measurements at time k
    
    for i in range(nk):
        CI, CII, M, F, Ftilde = get_auxiliary_variables(p, Cp, Ch)
        
        yi = y[:, i].reshape([2,1])
        
        # calculate moments for the kinematic state update
        yibar = H @ r
        Cry = Cr @ H.T
        Cy = H @ Cr @ H.T + CI + CII + Cv
        
        # udpate kinematic estimate
        r = r + Cry @ np.linalg.inv(Cy) @ (yi - yibar)
        Cr = Cr - Cry @ np.linalg.inv(Cy) @ Cry.T
        
        # Enforce symmetry of the covariance   
        Cr = (Cr + Cr.T) / 2
        
        # construct pseudo-measurement for the shape update
        Yi = F @ np.kron(yi - yibar, yi - yibar)
        
        # calculate moments for the shape update 
        Yibar = F @ np.reshape(Cy, [4, 1])
        CpY = Cp @ M.T
        CY = F @ np.kron(Cy, Cy) @ (F + Ftilde).T
        
        # update shape 
        p = p + CpY @ np.linalg.inv(CY) @ (Yi - Yibar)
        Cp = Cp - CpY @ np.linalg.inv(CY) @ CpY.T
        
        # Enforce symmetry of the covariance
        Cp = (Cp + Cp.T) / 2
    
    return r, p, Cr, Cp


def get_auxiliary_variables(p, Cp, Ch):
    alpha = p[0,0]
    l1 = p[1,0]
    l2 = p[2,0]
    S = np.dot(np.array([[np.cos(alpha), -np.sin(alpha)],
                         [np.sin(alpha), np.cos(alpha)]]),
               np.diag([l1, l2]))
    S1 = S[0, :]
    S2 = S[1, :]

    J1 = np.array([[-l1*np.sin(alpha), np.cos(alpha), 0],
                   [-l2*np.cos(alpha), 0, -np.sin(alpha)]])
    J2 = np.array([[l1*np.cos(alpha), np.sin(alpha), 0],
                   [-l2*np.sin(alpha), 0, np.cos(alpha)]])

    CI = np.dot(np.dot(S, Ch), S.T)
    CII = np.zeros((2, 2))
    CII[0, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J1)))
    CII[0, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J1)))
    CII[1, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J2)))
    CII[1, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J2)))

    M = np.vstack((2*np.dot(S1, np.dot(Ch, J1)),
                   2*np.dot(S2, np.dot(Ch, J2)),
                   np.dot(S1, np.dot(Ch, J2)) + np.dot(S2, np.dot(Ch, J1))))

    F = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 0, 0]])
    Ftilde = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])

    return CI, CII, M, F, Ftilde


def time_update(r, p, Cr, Cp, Ar, Ap, Cwr, Cwp):
    r = np.dot(Ar, r)
    Cr = np.dot(np.dot(Ar, Cr), Ar.T) + Cwr

    p = np.dot(Ap, p)
    Cp = np.dot(np.dot(Ap, Cp), Ap.T) + Cwp
    
    return r, p, Cr, Cp


def get_ground_truth():
    # Trajectory
    gt_orient = np.concatenate((np.tile(-np.pi/4, 20), np.arange(-np.pi/4, 0, np.pi/40), np.zeros(10),
                                 np.arange(0, 2*np.pi/4, np.pi/20), np.tile(2*np.pi/4, 20),
                                 np.arange(2*np.pi/4, np.pi, np.pi/20), np.tile(np.pi, 20)), axis=None)
    
    # Assume object is aligned along its velocity
    gt_vel = np.vstack(((500/36)*np.cos(gt_orient), (500/36)*np.sin(gt_orient)))
    gt_length = np.tile(np.vstack((340/2, 80/2)), (1, gt_vel.shape[1]))

    time_steps = gt_vel.shape[1]
    time_interval = 10

    # Setting ground truth
    gt_center = np.zeros((2, time_steps))
    gt_center[:, 0] = [0, 0]

    gt_rotation = np.zeros((2, 2, time_steps))
    for t in range(time_steps):
        gt_rotation[:, :, t] = np.array([[np.cos(gt_orient[t]), -np.sin(gt_orient[t])], 
                                          [np.sin(gt_orient[t]), np.cos(gt_orient[t])]])
        if t > 0:
            gt_center[:, t] = gt_center[:, t-1] + gt_vel[:, t]*time_interval

    return gt_center, gt_rotation, gt_orient, gt_length, gt_vel, time_steps, time_interval

def measurement_update_3d(y, H, r, p, Cr, Cp, Ch, Cv):
    if len(y) == 0:
        return r, p, Cr, Cp
    y = y.reshape([-1, 3]) if y.ndim == 1 else y
    nk = y.shape[1] # number of measurements at time k
    
    for i in range(nk):
        CI, CII, M, F, Ftilde = get_auxiliary_variables_3d(p, Cp, Ch)
        
        yi = y[:, i].reshape([3,1])
        
        # calculate moments for the kinematic state update
        yibar = H @ r
        Cry = Cr @ H.T
        Cy = H @ Cr @ H.T + CI + CII + Cv
        
        # udpate kinematic estimate
        r = r + Cry @ np.linalg.inv(Cy) @ (yi - yibar)
        Cr = Cr - Cry @ np.linalg.inv(Cy) @ Cry.T
        
        # Enforce symmetry of the covariance   
        Cr = (Cr + Cr.T) / 2
        
        # construct pseudo-measurement for the shape update
        Yi = F @ np.kron(yi - yibar, yi - yibar)
        
        # calculate moments for the shape update 
        Yibar = F @ np.reshape(Cy, [9, 1])
        CpY = Cp @ M.T
        CY = F @ np.kron(Cy, Cy) @ (F + Ftilde).T
        
        # update shape 
        p = p + CpY @ np.linalg.inv(CY) @ (Yi - Yibar)
        Cp = Cp - CpY @ np.linalg.inv(CY) @ CpY.T
        
        # Enforce symmetry of the covariance
        Cp = (Cp + Cp.T) / 2
    
    return r, p, Cr, Cp


def get_auxiliary_variables_3d(p, Cp, Ch):
    psi,theta,fa=p[:3,0]
    l1,l2,l3=p[3:,0]
    S = np.dot(np.array([[np.cos(theta)*np.cos(fa),\
                        np.sin(psi)*np.sin(theta)*np.cos(fa)-np.cos(psi)*np.sin(fa),\
                        np.cos(psi)*np.sin(theta)*np.cos(fa)+np.sin(psi)*np.sin(fa)],\
                         [np.cos(theta)*np.sin(fa),\
                        np.sin(psi)*np.sin(theta)*np.sin(fa)+np.cos(psi)*np.cos(fa),\
                        np.cos(psi)*np.sin(theta)*np.sin(fa)-np.sin(psi)*np.cos(fa)],\
                         [-np.sin(theta),np.sin(psi)*np.cos(theta),np.cos(psi)*np.cos(theta)]]),
               np.diag([l1, l2, l3]))
    S1 = S[0, :]
    S2 = S[1, :]
    S3 = S[2, :]

    J1 = np.array([[0,\
                    -l1*np.sin(theta)*np.cos(fa),\
                    -l1*np.cos(theta)*np.sin(fa),\
                    np.cos(theta)*np.cos(fa),\
                    0,\
                    0,],\
                   [l2*(np.cos(psi)*np.sin(theta)*np.cos(fa)+np.sin(psi)*np.sin(fa)),\
                    l2*(np.sin(psi)*np.cos(theta)*np.cos(fa)),\
                    l2*(-np.sin(psi)*np.sin(theta)*np.sin(fa)-np.cos(psi)*np.cos(fa)),\
                    0,\
                    np.sin(psi)*np.sin(theta)*np.cos(fa)-np.cos(psi)*np.sin(fa),\
                    0],\
                    [l3*(-np.sin(psi)*np.sin(theta)*np.cos(fa)+np.cos(psi)*np.sin(fa)),\
                    l3*(np.cos(psi)*np.cos(theta)*np.cos(fa)),\
                    l3*(-np.cos(psi)*np.sin(theta)*np.sin(fa)+np.sin(psi)*np.cos(fa)),\
                    0,\
                    0,\
                    np.cos(psi)*np.sin(theta)*np.cos(fa)+np.sin(psi)*np.sin(fa)]])
    J2 = np.array([[0,\
                    -l1*np.sin(theta)*np.sin(fa),\
                    l1*np.cos(theta)*np.cos(fa),\
                    np.cos(theta)*np.sin(fa),\
                    0,\
                    0,],\
                   [l2*(np.cos(psi)*np.sin(theta)*np.sin(fa)-np.sin(psi)*np.cos(fa)),\
                    l2*(np.sin(psi)*np.cos(theta)*np.sin(fa)),\
                    l2*(np.sin(psi)*np.sin(theta)*np.cos(fa)-np.cos(psi)*np.sin(fa)),\
                    0,\
                    np.sin(psi)*np.sin(theta)*np.sin(fa)+np.cos(psi)*np.sin(fa),\
                    0],\
                    [l3*(-np.sin(psi)*np.sin(theta)*np.sin(fa)-np.cos(psi)*np.cos(fa)),\
                    l3*(np.cos(psi)*np.cos(theta)*np.sin(fa)),\
                    l3*(np.cos(psi)*np.sin(theta)*np.cos(fa)+np.sin(psi)*np.sin(fa)),\
                    0,\
                    0,\
                    np.cos(psi)*np.sin(theta)*np.sin(fa)-np.sin(psi)*np.cos(fa)]])
    J3 = np.array([[0,-l1*np.cos(theta),0,-np.sin(theta),0,0],\
                    [l2*np.cos(psi)*np.cos(theta),-l2*np.sin(psi)*np.sin(theta),0,0,np.cos(theta)*np.sin(psi),0],\
                    [-l3*np.cos(theta)*np.sin(psi),-l3*np.cos(psi)*np.sin(theta),0,0,0,np.cos(psi)*np.cos(theta)]])

    CI = np.dot(np.dot(S, Ch), S.T)
    CII = np.zeros((3, 3))
    CII[0, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J1)))
    CII[0, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J1)))
    CII[0, 2] = np.trace(np.dot(np.dot(Cp, J3.T), np.dot(Ch, J1)))

    CII[1, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J2)))
    CII[1, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J2)))
    CII[1, 2] = np.trace(np.dot(np.dot(Cp, J3.T), np.dot(Ch, J2)))

    CII[2, 0] = np.trace(np.dot(np.dot(Cp, J1.T), np.dot(Ch, J3)))
    CII[2, 1] = np.trace(np.dot(np.dot(Cp, J2.T), np.dot(Ch, J3)))
    CII[2, 2] = np.trace(np.dot(np.dot(Cp, J3.T), np.dot(Ch, J3)))

    M = np.vstack((2*np.dot(S1, np.dot(Ch, J1)),
                   2*np.dot(S2, np.dot(Ch, J2)),
                   2*np.dot(S3, np.dot(Ch, J3)),
                   np.dot(S1, np.dot(Ch, J2)) + np.dot(S2, np.dot(Ch, J1)),
                   np.dot(S1, np.dot(Ch, J3)) + np.dot(S3, np.dot(Ch, J1)),
                   np.dot(S2, np.dot(Ch, J3)) + np.dot(S3, np.dot(Ch, J2))))

    F = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0]])
    Ftilde = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0]])

    return CI, CII, M, F, Ftilde