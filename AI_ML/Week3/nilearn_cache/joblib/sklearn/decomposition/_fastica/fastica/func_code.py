# first line: 150
def fastica(
    X,
    n_components=None,
    *,
    algorithm="parallel",
    whiten=True,
    fun="logcosh",
    fun_args=None,
    max_iter=200,
    tol=1e-04,
    w_init=None,
    random_state=None,
    return_X_mean=False,
    compute_sources=True,
    return_n_iter=False,
):
    """Perform Fast Independent Component Analysis.

    The implementation is based on [1]_.

    Read more in the :ref:`User Guide <ICA>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    n_components : int, default=None
        Number of components to extract. If None no dimension reduction
        is performed.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        Apply a parallel or deflational FASTICA algorithm.

    whiten : bool, default=True
        If True perform an initial whitening of the data.
        If False, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
        Otherwise you will get incorrect results.
        In this case the parameter n_components will be ignored.

    fun : {'logcosh', 'exp', 'cube'} or callable, default='logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. The derivative should be averaged along its last dimension.
        Example:

        def my_g(x):
            return x ** 3, np.mean(3 * x ** 2, axis=-1)

    fun_args : dict, default=None
        Arguments to send to the functional form.
        If empty or None and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}

    max_iter : int, default=200
        Maximum number of iterations to perform.

    tol : float, default=1e-04
        A positive scalar giving the tolerance at which the
        un-mixing matrix is considered to have converged.

    w_init : ndarray of shape (n_components, n_components), default=None
        Initial un-mixing array of dimension (n.comp,n.comp).
        If None (default) then an array of normal r.v.'s is used.

    random_state : int, RandomState instance or None, default=None
        Used to initialize ``w_init`` when not specified, with a
        normal distribution. Pass an int, for reproducible results
        across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_X_mean : bool, default=False
        If True, X_mean is returned too.

    compute_sources : bool, default=True
        If False, sources are not computed, but only the rotation matrix.
        This can save memory when working with big data. Defaults to True.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    K : ndarray of shape (n_components, n_features) or None
        If whiten is 'True', K is the pre-whitening matrix that projects data
        onto the first n_components principal components. If whiten is 'False',
        K is 'None'.

    W : ndarray of shape (n_components, n_components)
        The square matrix that unmixes the data after whitening.
        The mixing matrix is the pseudo-inverse of matrix ``W K``
        if K is not None, else it is the inverse of W.

    S : ndarray of shape (n_samples, n_components) or None
        Estimated source matrix

    X_mean : ndarray of shape (n_features,)
        The mean over features. Returned only if return_X_mean is True.

    n_iter : int
        If the algorithm is "deflation", n_iter is the
        maximum number of iterations run across all components. Else
        they are just the number of iterations taken to converge. This is
        returned only when return_n_iter is set to `True`.

    Notes
    -----
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = AS where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where ``S = W K X.``
    While FastICA was proposed to estimate as many sources
    as features, it is possible to estimate less by setting
    n_components < n_features. It this case K is not a square matrix
    and the estimated A is the pseudo-inverse of ``W K``.

    This implementation was originally made for data of shape
    [n_features, n_samples]. Now the input is transposed
    before the algorithm is applied. This makes it slightly
    faster for Fortran-ordered input.

    References
    ----------
    .. [1] A. Hyvarinen and E. Oja, "Fast Independent Component Analysis",
           Algorithms and Applications, Neural Networks, 13(4-5), 2000,
           pp. 411-430.
    """

    est = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        whiten=whiten,
        fun=fun,
        fun_args=fun_args,
        max_iter=max_iter,
        tol=tol,
        w_init=w_init,
        random_state=random_state,
    )
    sources = est._fit(X, compute_sources=compute_sources)

    if whiten:
        if return_X_mean:
            if return_n_iter:
                return (est.whitening_, est._unmixing, sources, est.mean_, est.n_iter_)
            else:
                return est.whitening_, est._unmixing, sources, est.mean_
        else:
            if return_n_iter:
                return est.whitening_, est._unmixing, sources, est.n_iter_
            else:
                return est.whitening_, est._unmixing, sources

    else:
        if return_X_mean:
            if return_n_iter:
                return None, est._unmixing, sources, None, est.n_iter_
            else:
                return None, est._unmixing, sources, None
        else:
            if return_n_iter:
                return None, est._unmixing, sources, est.n_iter_
            else:
                return None, est._unmixing, sources
