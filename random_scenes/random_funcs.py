def factorize(n):
    if n < 2:
        return []
    for k in range(2, int(n**0.5 + 1)):
        if n % k == 0:
            return factorize(k) + factorize(n // k)
    return [n]


def is_prime(n):
    if n < 2:
        return False
    for k in range(2, int(n**0.5 + 1)):
        if n % k == 0:
            return False
    return True


def to_base4(n):
    bits = "{0:b}".format(n)
    result = ""
    if len(bits) % 2 != 0:
        bits = "0" + bits
    for c1, c2 in zip(bits[-2::-2], bits[-1::-2]):
        new_char = str(2 * int(c1) + int(c2))
        result = new_char + result
    return int(result)


def lattice_points_in_R(R):
    N = int(np.ceil(R))
    R_squared = R**2
    result = 1  # (0, 0)
    result += 4 * N  # axis points
    result += 4 * int(N / np.sqrt(2))  # Diagonal points
    result += 8 * sum([
        int(x**2 + y**2 <= R_squared)
        for x in range(0, N + 1)
        for y in range(1, x)
    ])
    return result