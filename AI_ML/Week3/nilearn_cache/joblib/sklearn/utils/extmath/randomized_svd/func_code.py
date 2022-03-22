# first line: 249
def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state="warn",
):
    """Computes a truncated randomized SVD.

    This method solves the fixed-rank approximation problem described in the
    Halko et al paper (problem (1.5), p5).

    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See Halko
        et al (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see Halko et al paper, page 9).

        .. versionchanged:: 0.18

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : bool or 'auto', default='auto'
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : bool, default=True
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, default='warn'
        The seed of the pseudo random number generator to use when
        shuffling the data, i.e. getting the random vectors to initialize
        the algorithm. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

        .. versionchanged:: 1.2
            The previous behavior (`random_state=0`) is deprecated, and
            from v1.2 the default value will be `random_state=None`. Set
            the value of `random_state` explicitly to suppress the deprecation
            warning.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions (Algorithm 4.3)
      Halko, et al., 2009 https://arxiv.org/abs/0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    if isinstance(M, (sparse.lil_matrix, sparse.dok_matrix)):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    if random_state == "warn":
        warnings.warn(
            "If 'random_state' is not supplied, the current default "
            "is to use 0 as a fixed seed. This will change to  "
            "None in version 1.2 leading to non-deterministic results "
            "that better reflect nature of the randomized_svd solver. "
            "If you want to silence this warning, set 'random_state' "
            "to an integer seed or to None explicitly depending "
            "if you want your code to be deterministic or not.",
            FutureWarning,
        )
        random_state = 0

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(
        M,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, Vt = linalg.svd(B, full_matrices=False)

    del B
    U = np.dot(Q, Uhat)

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]
