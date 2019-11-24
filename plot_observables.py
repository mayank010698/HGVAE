def get_magnetisation(lattice):
    mag_x = np.mean(np.cos(2 * np.pi * lattice))
    mag_y = np.mean(np.sin(2 * np.pi * lattice))

    return (mag_x ** 2 + mag_y ** 2) ** 0.5


def get_energy(lattice):
    # print(lattice.shape)
    xp = np.zeros((8, 8))
    yp = np.zeros((8, 8))

    # print("row_wise")
    for i in range(lattice.shape[0]):
        xp[i, :] = np.cos(lattice[(i + 1) % lattice.shape[0], :] - lattice[i, :])

    # print("col_wise")
    for i in range(lattice.shape[1]):
        yp[:, i] = np.cos(lattice[:, (i + 1) % lattice.shape[1]] - lattice[:, i])

    return (np.sum(xp) + np.sum(yp)) / 64


