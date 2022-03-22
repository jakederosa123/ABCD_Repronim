# first line: 28
def fast_svd(X, n_components, random_state=None):
    """ Automatically switch between randomized and lapack SVD (heuristic
        of scikit-learn).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data to decompose

    n_components : integer
        The order of the dimensionality of the truncated SVD

    random_state : int or RandomState, optional
        Pseudo number generator state used for random sampling.

    Returns
    -------

    U : array, shape (n_samples, n_components)
        The first matrix of the truncated svd

    S : array, shape (n_components)
        The second matric of the truncated svd

    V : array, shape (n_components, n_features)
        The last matric of the truncated svd

    """
    random_state = check_random_state(random_state)
    # Small problem, just call full PCA
    if max(X.shape) <= 500:
        svd_solver = 'full'
    elif n_components >= 1 and n_components < .8 * min(X.shape):
        svd_solver = 'randomized'
    # This is also the case of n_components in (0,1)
    else:
        svd_solver = 'full'

    # Call different fits for either full or truncated SVD
    if svd_solver == 'full':
        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)
        # The "copy" are there to free the reference on the non reduced
        # data, and hence clear memory early
        U = U[:, :n_components].copy()
        S = S[:n_components]
        V = V[:n_components].copy()
    else:
        n_iter = 'auto'

        U, S, V = randomized_svd(X, n_components=n_components,
                                 n_iter=n_iter,
                                 flip_sign=True,
                                 random_state=random_state)
    return U, S, V
