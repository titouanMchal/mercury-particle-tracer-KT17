import numpy as np
from scipy.integrate import solve_ivp
import KT17
import functions as f
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


RM = f.RM
DIST_SUN = 0.314666
DISTURBANCE = 50
SHIFT = 479e3 / RM

lat_CRB = np.load("reversal_boundary.npy")
fi_ref = np.linspace(0, np.pi, 30)


def B_KT(
    coord: np.ndarray, distance_sun: float = DIST_SUN, disturbance: float = DISTURBANCE
):
    """
    KT17 model - Korth et al., 2017 - Matt James Implementation

    Inputs
    ----------
    coord : np.ndarray
        Position vector in simulation frame

    Returns
    -------
    np.ndarray
        The 3-component [Bx, By, Bz] in simulation frame
    """
    x, y, z = f.sim_frame_to_msm(coord / RM)
    Bx, By, Bz = KT17.ModelField(x, y, z, Rsun=distance_sun, DistIndex=disturbance)
    Bx = -Bx
    By = -By
    B = np.array([Bx, By, Bz])

    return B[:, 0] * 1e-9


def B_KT_MSM(x0: np.ndarray):
    """
    Computes magnetic field in MSM frame

    Inputs
    ----------
    x0 : np.ndarray
        Position vector in MSM frame

    Returns
    -------
    np.ndarray
        The 3-component [Bx, By, Bz] in MSM frame
    """
    Bx, By, Bz = KT17.ModelField(
        x0[0], x0[1], x0[2], Rsun=DIST_SUN, DistIndex=DISTURBANCE
    )
    B = np.array([Bx, By, Bz]) * 1e-9
    return B[:, 0]


def projection_KT(x0):
    """
    Trace a magnetic field line from the starting point x0 until it
    intersects the shifted sphere of radius rstop = 0.804 RM

    Parameters
    ----------
    x0 : np.ndarray
        Initial position vector in simulation frame (meters).

    Returns
    -------
    np.ndarray or None
        The 3-component position vector at the intersection point on the sphere in MSM coordinates,
        or None if the solver fails or the sphere is never reached.
    """
    rstop = 1 - 479e3 / RM
    x = x0.copy()

    x[0], x[1], x[2] = f.sim_frame_to_msm(x / RM)
    x[2] = np.abs(x[2])

    def fun(t, x):
        return B_KT_MSM(x)

    def hit_sphere(t, x):
        return np.linalg.norm(x) - rstop

    hit_sphere.terminal = True
    hit_sphere.direction = 0

    events = hit_sphere

    try:
        sol = solve_ivp(
            fun,
            (0, 1e15),
            x,
            method="RK45",
            max_step=1e13,
            events=hit_sphere,
        )
    except Exception:
        return None

    if sol.success and sol.t_events[0].size > 0:
        return sol.y_events[0][0]

    return None


def read_potential_KT(coord: np.ndarray, pdrop: float):
    """
    Computes electrostatic potential for a given position
    If you change the magnetic field settings (DIST_SUN, DISTURBANCE) and use the electric field,
    make sure you calculate the new reversal latitude with find_reversal() and update lat_CRB
    Inputs
    --------
    coord
        position vector in simulation frame
    pdrop
        half potential drop

    Returns
    ---------
    p
        local electrostatic potential value (V)
    """
    theta0 = np.deg2rad(1)

    x_proj_MSM = projection_KT(coord)

    if x_proj_MSM is None:
        return None

    x_proj = x_proj_MSM * RM
    x_proj[0] = -x_proj[0]
    x_proj[1] = -x_proj[1]

    x, y, z = x_proj[0], x_proj[1], x_proj[2]
    r, fi, lat = f.cartesian_to_spherical(x, y, z)

    lat_r = np.interp(np.abs(fi), fi_ref, lat_CRB)

    theta_r = np.pi / 2 - lat_r
    theta1 = theta_r - theta0 / 2
    theta2 = theta_r + theta0 / 2

    theta = np.pi / 2 - lat

    p1 = np.sin(theta) / np.sin(theta_r)
    p2 = 1 / p1**4
    p = p1

    if theta > theta2:
        p = p2

    if theta1 <= theta <= theta2:

        delta = theta - theta1
        pf = (
            10 * (delta / theta0) ** 3
            - 15 * (delta / theta0) ** 4
            + 6 * (delta / theta0) ** 5
        )
        p = p1 + pf * (p2 - p1)

    return pdrop * p * np.sin(fi)


def E_KT(coord, pdrop):
    """
    Computes electrostatic potential, E orthogonal components, E vector at point coord

    Inputs
    ------
    coord
        position vector
    pdrop
        half electrostatic potential drop (V)

    Outputs
    ------
    p0
        local potential
    e1, e2
        E orthogonal components (e1 is in the XZ plane)
    E
        electric field vector (V/m)
    """

    nB = np.linalg.norm(B_KT(coord))
    dist_neighbors = 5e-2 / nB

    u1, u2 = f.vec_ortho(B_KT(coord))
    v1, v2 = f.neigbors(u1, u2, coord, dist_vec=dist_neighbors)

    p0 = read_potential_KT(coord, pdrop=pdrop)
    p1 = read_potential_KT(v1, pdrop=pdrop)
    p2 = read_potential_KT(v2, pdrop=pdrop)

    if any(p is None for p in (p0, p1, p2)):
        return None

    e2 = (p0 - p2) / dist_neighbors
    e1 = (p0 - p1) / dist_neighbors

    E = e1 * u1 + e2 * u2
    E = np.array(E)

    return p0, e1, e2, E


def slice_B_KT(
    plot_magnitude=False,
    resolution=150,
    projection="XZ",
    di: float = DISTURBANCE,
    dist_sun: float = DIST_SUN,
    xmax: float = 5,
    ymax: float = 3,
    zmax: float = 3,
):

    fig, ax = plt.subplots()
    

    if plot_magnitude:
        from matplotlib.colors import LogNorm
        x = np.linspace(-1.6, xmax, resolution)
        y = np.zeros(resolution)
        z = np.linspace(-zmax, zmax, resolution)
        magn = np.zeros((resolution, resolution))

        nx, nz = len(x), len(z)
        for i in range(nz):
            for j in range(nx):

                Bx, By, Bz = B_KT(np.array([x[j], 0, z[i]])*RM)*1e9
                magn[i, j] = np.sqrt(Bx**2 + Bz**2 + By**2)

        magn = np.ma.masked_invalid(magn)
        cmap = plt.cm.viridis
        cmap.set_bad(color="white")
        im = ax.imshow(
            magn,
            extent=[x.min(), x.max(), z.min(), z.max()],
            origin="lower",
            cmap=cmap,
            norm=LogNorm(vmax=1e3),
            interpolation="bilinear",
            alpha=0.9,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            r"$\Vert \vec{B} \Vert$ (nT)", fontsize=13, fontname="DejaVu Serif"
        )

    ax.add_patch(Circle((0, 0), 1, fill=1, color="black", zorder=3))
    angle_start = 90
    angle_end = 270
    angles = np.linspace(np.radians(angle_start), np.radians(angle_end), 100)
    x_circle = np.cos(angles)
    y_circle = np.sin(angles)
    ax.fill_betweenx(y_circle, x_circle, 0, color="lightgray", zorder=4)

    if projection == "XZ":
        t = np.linspace(0.1 * np.pi, np.pi, 25)
        x0 = np.cos(t)
        z0 = np.sin(t)
        y0 = np.zeros(len(z0))
        T = KT17.TraceField(
            x0,
            y0,
            z0,
            Rsun=dist_sun,
            DistIndex=di,
            MPStop=True,
            EndSurface=2,
            TraceDir=0,
        )

        for i in range(T.n):
            line = T.GetTrace(i)
            x = line["x"]
            z = line["z"]
            x = -x
            z = z
            if plot_magnitude:
                col = "black"
            else:
                col = "gray"
            ax.plot(x, z + SHIFT, linewidth=0.8, color=col)
            ax.plot(x, -z + SHIFT, color=col, linewidth=0.8)
            ax.set_xlabel("X ($R_M$)", fontsize=13, fontname="DejaVu Serif")
            ax.set_ylabel("Z ($R_M$)", fontsize=13, fontname="DejaVu Serif")
            ax.set_ylim(-zmax, zmax)

    if projection == "XY":
        ax.set_xlabel("X ($R_M$)", fontsize=13, fontname="DejaVu Serif")
        ax.set_ylabel("Y ($R_M$)", fontsize=13, fontname="DejaVu Serif")
        ax.set_ylim(-ymax, ymax)

    fig.set_size_inches(10, 6)
    ax.set_xlim(-1.6, xmax)
    ax.set_aspect("equal")
    return fig, ax


def find_reversal_KT(save: bool = False, N_iter : int = 30):
    """
    Computes reversal boundary latitude in MSM frame for different local time values
    
    Parameters
    ----------
    save : bool, optional
        If True, saves the latitude array to 'reversal_boundary.npy'.
    N_iter : int, optional
        Number of iterations for the dichotomous search

    Returns
    -------
    None
    Prints the latitude values and shows diagnostic plots.
    """
    from matplotlib.ticker import AutoMinorLocator

    longitudes = fi_ref
    r_begin = 1 - SHIFT

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax.add_patch(Circle((0, SHIFT), 1 - SHIFT, color="grey"))
    ax.add_patch(Circle((0, 0), 1, color="lightgrey", alpha=0.8))
    ax2.add_patch(Circle((0, 0), 1 - SHIFT, color="grey"))
    ax2.add_patch(Circle((0, 0), 1, color="lightgrey", alpha=0.8))

    lat0 = 0
    lat1 = np.pi / 2

    boundaries = []

    for longitude in longitudes:
        for i in range(N_iter):

            lat = (lat1 + lat0) / 2
            x0, y0, z0 = f.spherical_to_cartesian(r=r_begin, fi=longitude, lat=lat)

            x0 = -x0
            y0 = -y0
            z0 = z0

            T = KT17.TraceField(
                x0,
                y0,
                z0,
                DistIndex=DISTURBANCE,
                Rsun=DIST_SUN,
                MPStop=False,
                EndSurface=2,
                TraceDir=-1,
                MaxLen=3000,
                TailX=-12,
            )
            line = T.GetTrace(0)
            x = -line["x"]
            y = -line["y"]
            z = line["z"] + SHIFT


            dist = np.sqrt(x[0] ** 2 + z[0] ** 2 + y[0] ** 2)

            if dist < 1 or x[0] < -5:
                lat0 = lat

            elif x[0] > 8:
                lat1 = lat

        ax.plot(x, z, linewidth=0.8, color="darkred", zorder=4.5)
        ax2.plot(x, y, linewidth=0.8, color="darkred", zorder=4.5)
        ax2.plot(x, -y, linewidth=0.8, color="darkred", zorder=4.5)
        ax2.scatter(x[-1], y[-1], zorder=6.1, color="black", s=3)
        ax2.scatter(x[-1], -y[-1], zorder=6.1, color="black", s=3)
        lat1 = np.pi / 2
        lat0 = 0
        boundaries.append(lat)
    
    boundaries = np.array(boundaries)
    ax.set_aspect("equal")
    ax2.set_aspect("equal")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel("X ($R_M$)", fontsize=13, fontname="DejaVu Serif")
    ax.set_ylabel("Z ($R_M$)", fontsize=13, fontname="DejaVu Serif")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 2)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel("X ($R_M$)", fontsize=13, fontname="DejaVu Serif")
    ax2.set_ylabel("Y ($R_M$)", fontsize=13, fontname="DejaVu Serif")
    ax.hlines(SHIFT, -1, 1, color="black", zorder=8.5, linewidth=0.5, linestyle="--")
    ax.add_patch(
        Circle(
            (0, SHIFT),
            1 - SHIFT,
            color="black",
            fill=False,
            zorder=5.5,
            linewidth=0.5,
            linestyle="-",
        )
    )
    ax.add_patch(
        Circle(
            (0, 0),
            0.832,
            color="black",
            fill=False,
            zorder=5.5,
            linewidth=0.1,
            linestyle="--",
        )
    )

    plt.show()
    print("reversal boundary (deg) : ", np.rad2deg(boundaries))

    if save:
        np.save("reversal_boundary.npy", boundaries)


if __name__ == "__main__":

    find_reversal_KT()

