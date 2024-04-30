from shapely.geometry import Point, Polygon
import numpy as np
from scipy.spatial import ConvexHull

def iou_of_two_dimensions(points1, points2, dim='xy'):
    map_dim = {'x': 0, 'y': 1, 'z': 2}

    if len(dim) != 2 or not all(d in map_dim for d in dim):
        raise ValueError('dim must be one of x, y, z')

    dims = np.array([map_dim[d] for d in dim])

    try:
        hull1 = ConvexHull(points1[:, dims])
        hull2 = ConvexHull(points2[:, dims])
    except Exception:
        return 0

    polygon1 = Polygon(points1[hull1.vertices][:, dims])
    polygon2 = Polygon(points2[hull2.vertices][:, dims])

    # Compute the intersection
    intersection_area = polygon1.intersection(polygon2).area

    # Compute the union
    union_area = polygon1.union(polygon2).area

    # Calculate IOU
    iou_ratio = intersection_area / union_area

    return iou_ratio

def iou_of_convex_hulls(points1, points2):
    """Calculates the IOU of the convex hulls constructed by the two sets of points.

    Args:
    points1: A set of 3D points.
    points2: A set of 3D points.
    n_ground: normal vector of the ground plane.
    d_ground: distance from the origin to the ground plane.
    """
    # Create Shapely MultiPoint objects from the point sets

    ret = []
    ret.append(iou_of_two_dimensions(points1, points2, 'xy'))
    ret.append(iou_of_two_dimensions(points1, points2, 'yz'))
    ret.append(iou_of_two_dimensions(points1, points2, 'xz'))

    return np.array(ret)


def difference_between_velocity(v: float, v_gt: float) -> float:
    """Calculates the difference between the predicted velocity and the ground truth velocity.

    Args:
    v: The predicted velocity.
    v_gt: The ground truth velocity.

    Returns:
    The difference between the predicted velocity and the ground truth velocity.
    """
    return np.abs(v) - np.abs(v_gt)