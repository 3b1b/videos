from manim_imports_ext import *

def sanitize_3D_vector(pt):
    """
    Attempts to format input as a NumPy array of size (N,3)
    """

    # Try to convert the input to a NumPy array of the right size
    pts = np.array(pt,ndmin=2)

    if len(pts.shape) > 2:
        raise ValueError("3D vectors cannot have depth more than 2.")

    if pts.shape[1] != 3:
        raise ValueError("3D vectors must have 3 components.")

    return pts


def sanitize_scalar(x):
    """
    Attemps to format input as a NumPy array of size (N,)
    """

    # Try to convert input to a NumPy array of the right size
    x_array = np.array(x,ndmin=1)

    if len(x_array.shape) > 1:
        raise ValueError("Scalars cannot have depth more than 2.")

    return x_array


def direction_field(pt, discontinuities="equator", epsilon=0.01):
    """
    Parameters
    ----------
    pt : NumPy array (N,3), parallelizable
        Point on the unit sphere

    discontinuities : str
        Specifies where the direction field can be discontinuous.
        Three options:
            "equator"
            "two"
            "one"

    epsilon : float
        Determines error used in dealing with discontinuities

    Returns
    -------
    vec : NumPy array (N,3), parallelizable
        Associated unit vector
    """

    # Sanitize inputs
    pts = sanitize_3D_vector(pt)

    # Get coordinates of the points
    (x,y,z) = np.transpose(pts)

    # Split operation based on the number of discontinuities

    if discontinuities == "equator":
        # Define mask depending on whether we are close to the equator
        mask = (abs(z) < 0.1)

        # Define behavior on equator
        equator = np.stack((-y,x,0*z))

        # Define behavior off equator
        non_equator = np.stack((0*x,z,-y))

        # Combine using the mask
        perp = mask*equator + np.logical_not(mask)*non_equator

    elif discontinuities == "two":
        # Define mask depending on whether we are near the poles
        mask = abs(abs(z)-1) < epsilon

        # Define behavior around poles
        poles = np.stack((0*x,z,-y))

        # Define behavior away from poles
        non_poles = np.stack((-y,x,0*z))

        # Combine using the mask
        perp = mask*poles + np.logical_not(mask)*non_poles

    elif discontinuities == "one":
        # Define mask depending on whether we are near the north pole
        mask = abs(z-1) < epsilon

        # Define behavior around north pole
        north_pole = np.stack((0*x,z,-y))

        # Define behavior away from north pole
        X = x**2 - y**2 - (z-1)**2
        Y = 2*x*y
        Z = 2*x*(z-1)

        non_north_pole = np.stack((X,Y,Z))

        # Combine using the mask
        perp = mask*north_pole + np.logical_not(mask)*non_north_pole

    else:
        raise NotImplementedError()

    # Determine number of vectors
    field = np.transpose(perp)
    num_pts = field.shape[0]

    # normalize vector field before returning
    norm = np.linalg.norm(field, axis=1)
    norm = np.reshape(norm,[num_pts,1])
    vec = field/norm

    return vec


def distension(pt, t):
    """
    Helper function for computing amount to distend homotopy.

    Parameters
    ----------
    pt : NumPy array (M,3), parallelizable
        Point on the unit sphere
    t : float or NumPy array (N,)
        Time elapsed (between 0 and 1)

    Returns
    -------
    rho_factor : NumPy array (N,M)
        Adjustment to distance from the origin
    """

    # Compute the scaling factor from time
    time_factor = np.sin(np.pi * t)

    # Compute the scaling factor from space
    (x,y,z) = np.transpose(pt)
    space_factor = y*z

    # Combine the factors
    rho_factor = np.tensordot(time_factor, space_factor, axes = 0)

    return rho_factor


def great_circle_map(pt, t, discontinuities="one", distend=0, epsilon=0.01):
    """
    Parameters
    ----------
    pt : NumPy array (M,3), parallelizable
        Point on the unit sphere

    t : float or NumPy array (N,)
        Time elapsed (between 0 and 1)

    discontinuities : str
        Specifies where the direction field can be discontinuous.
        Three options:
            "equator"
            "two"
            "one"

    distend : float or NumPy array (T,)
        Amount to distend from spherical surface

    epsilon : float
        Determines error used in dealing with discontinuities

    Returns
    -------
    new_pts : NumPy array (N,T,M,3), parallelizable
        Location of pt after time t
    """

    # Sanitize inputs
    times = sanitize_scalar(t)
    dist_factors = sanitize_scalar(distend)
    pts = sanitize_3D_vector(pt)

    # Calculate initial unit vectors
    units = direction_field(pts, discontinuities, epsilon)

    # Compute weights for the great circle map
    scaled_times = np.pi * times
    u1 = np.cos(scaled_times)
    u2 = np.sin(scaled_times)

    # Compute linear combination using the constructed weights
    base_pts = (np.tensordot(u1, pts,axes=0) + np.tensordot(u2, units,axes=0))

    # Compute distension factors
    rho_factors = distension(pts, times)
    full_factors = 1 + np.tensordot(rho_factors, dist_factors,axes=0)

    # Reshape the base points and the scaling factors so that they can be multiplied together cleanly
    base_pts_reshaped = np.expand_dims(base_pts, axis=2)
    factors_reshaped = np.expand_dims(full_factors, axis=-1)

    # Combine and then reshape the array for convenience
    new_pts = base_pts_reshaped * factors_reshaped
    new_pts = new_pts.transpose((0,2,1,3))

    return new_pts


### TEST FUNCTIONS

def test_direction_field(num_pts=1000, epsilon=0.001):
    """
    Function to test correctness of direction_field

    Parameters
    ----------
    num_pts : int
        Number of points on sphere to test. The default is 1000.
    epsilon : float
        Acceptable error. The default is 0.001.

    Returns
    -------
    None
    """

    points = fibonacci_sphere(num_pts)


    failed_counts = {"equator":0,"two":0,"one":0}

    for disc in ["equator","two","one"]:
        field = direction_field(points,discontinuities=disc,epsilon=epsilon)
        dots = (points*field).sum(axis=1)

        failures = (abs(dots) >= epsilon).sum(axis=0)
        failed_counts[disc] = failures

    print("Number of points where the direction field failed to be orthogonal to the sphere.")
    print("")

    for key in failed_counts.keys():
        value = failed_counts[key]

        print(f"{key}: {value} points")

    print("")
    print("Count completed.")
    return None


def test_great_circle_map(discontinuities="one"):
    """
    Tests correctness of great_circle_map

    Parameters
    ----------
    discontinuities : str
        Selects base vector field. The default is "one".

    Returns
    -------
    None.

    """

    # Define acceptable error
    epsilon = 0.000001

    # Initialize points on the sphere to try
    pts = fibonacci_sphere(15)

    # Initialize distension amounts
    distends = [0.1*x for x in range(11)]

    # Initialize first batch of points
    end_pts = great_circle_map(pts,[0,1],discontinuities=discontinuities,distend=distends)

    # Check whether dimensions match what they should be
    pts_dims = end_pts.shape

    if len(pts_dims) != 4:
        print("Warning! Array has an incorrect number of dimensions.")

    elif pts_dims != (2,len(distends),len(pts),3):
        print("Warning: Dimensions of array are incorrect.")

    else:
        print("Array has expected dimensions.")
        print("")

    # Separate points belonging to the beginning and end of the homotopy
    beginning = end_pts[0]
    ending = end_pts[1]

    identity_pass = True
    antipode_pass = True

    # Go through all copies of beginning; see if they match the identity
    for distend,pts_copy in zip(distends,beginning):
        if np.linalg.norm(pts-pts_copy) > epsilon:
            if identity_pass:
                print("Warning! At time t=0, non-identity map at distensions:")

            identity_pass = False

            print(distend)

    # If there have been no errors, print success
    if identity_pass:
        print("Homotopy correctly defaults to identity at time t=0")
        print("No dependence on distension")

    print("")

    # Go through all copies of ending; see if they match antipode
    for distend,pts_copy in zip(distends,ending):
        if np.linalg.norm(pts+pts_copy) > epsilon:
            if antipode_pass :
                print("Warning! At time t=1, non-antipode map at distensions:")

            antipode_pass = False

            print(distend)

    # If there have been no errors, print success
    if antipode_pass:
        print("Homotopy correctly defaults to antipode at time t=1")
        print("No dependence on distension")

    print("")

    # Initialize second batch of points
    halfway_pts = great_circle_map(pts,1/2,discontinuities=discontinuities)[0][0]

    # At zero distension, halfway points should be sqrt(2) away from where they started

    # Compute distance between halfway points and original
    distances = np.linalg.norm(halfway_pts-pts,axis=1)

    # Compute discrepancy away from sqrt(2) expected distance
    discrepancies = np.abs(distances - np.sqrt(2))

    # Check whether maximum discrepancy is larger than the tolerance
    if np.max(discrepancies) > epsilon:
        print("Warning! Points tested at the halfway point at zero distension are in the wrong position.")
    else:
        print("Points are halfway around the great circle at t=1/2 with distension 0.")

    print("")

    # Initialize third batch of points
    infinitesimal_pts = great_circle_map(pts,epsilon**2,discontinuities=discontinuities)[0][0]

    # At zero distension, moving points an infinitesimal amount should agree with the underlying vector field

    # Compute difference in positions to get velocity
    velocities_actual = infinitesimal_pts - beginning[0]

    # Compute underlying field and rescale
    units = direction_field(pts,discontinuities=discontinuities)
    velocities_expected = units * np.pi * (epsilon**2)

    # Compute differences between velocities
    velocity_discrepancy = np.linalg.norm(velocities_actual - velocities_expected,axis=1)

    if np.max(velocity_discrepancy) > epsilon:
        print("Warning! Points do not appear to move in the direction of the vector field.")
    else:
        print("Points move in the direction of the vector field.")

    print("")

    # Compute fourth batch of points
    # Choose a selection of random times between 0 and 1
    times=np.random.rand(5)
    distension_pts = great_circle_map(pts,times,discontinuities=discontinuities,distend=distends)

    # Pull out subarray with zero distension
    zero_distension_pts = distension_pts[0:,0:1,0:,0:]

    # Compute norms of all points and normalize
    norms = np.expand_dims(np.linalg.norm(distension_pts,axis=-1),axis=-1)
    normalized_pts = distension_pts/norms

    # Normalized points should match zero distension ones
    differences = normalized_pts - zero_distension_pts
    discrepancies = np.linalg.norm(differences,axis=-1)

    if np.max(discrepancies) > epsilon:
        print("Warning! Discrepancy is shifting the directions of points, not just radially.")
    else:
        print("Discrepancy moves points only radially.")

    print("")
    print("All tests completed.")

    return None


## Manim Scenes

def spherical_surface(theta, phi):
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)
    return np.array([[X, Y, Z]])


def spherical_eversion(theta, phi, t):
    pt = spherical_surface(theta, phi)
    new_pt = great_circle_map(pt, t, discontinuities="one", distend=0.5)
    return new_pt[0][0][0]
