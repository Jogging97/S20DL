import numpy as np


def problem_a(A, B):
    return A + B


def problem_b(A, B, C):
    return A.dot(B) - C


def problem_c(A, B, C):
    return A * B + C.T


def problem_d(x, y):
    return np.inner(x, y)


def problem_e(A):
    return np.zeros(shape=A.shape)


def problem_f(A, x):
    return np.linalg.solve(A, x)


def problem_g(A, x):
    solution = np.linalg.solve(A.T, x.T)
    return solution.T


def problem_h(A, alpha):
    return A + alpha * np.eye(A.shape[0])


def problem_i(A, i, j):
    return A[i][j]


def problem_j(A, i):
    return np.sum(A[i])


def problem_k(A, c, d):
    A[A < c] = 0
    A[A > d] = 0
    return np.mean(A[np.nonzero(A)])


def problem_l(A, k):
    eigenvector = np.linalg.eig(A)[1]
    col = A.shape[1] - k
    return eigenvector[:, col:]


def problem_m(x, k, m, s):
    return np.random.multivariate_normal(x + m * (np.ones(len(x))), s * (np.eye(len(x))), (len(x), k))


def problem_n(A):
    return np.random.shuffle(A)


def linear_regression(X_tr, y_tr):
    w = np.linalg.solve(np.dot(X_tr.T, X_tr), np.dot(X_tr.T, y_tr))
    return w


def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    w_tr = linear_regression(X_tr, ytr)
    fmse_tr = np.square(np.subtract(np.dot(X_tr, w_tr), ytr)).mean() / 2
    return fmse_tr


def test_age_regressor():
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")
    w_te = linear_regression(X_te, yte)
    fmse_te = np.square(np.subtract(np.dot(X_te, w_te), yte)).mean() / 2
    return fmse_te


if __name__ == "__main__":
    fmse_tr = train_age_regressor()
    fmse_te = test_age_regressor()

    print("fMSE cost on the training data is:", fmse_tr)
    print("fMSE cost on the testing data is:", fmse_te)


    # test
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[1, 2], [3, 4]])
    C = np.array([[1, 2], [3, 4]])
    x = np.array([0, 1])
    y = np.array([1, 1])
    alpha = 2
    i = 1
    j = 1
    c = 1
    d = 2
    k = 1
    m = 1
    s = 2
    print("problem_a:", problem_a(A, B))
    print("problem_b:",problem_b(A, B, C))
    print("problem_c:",problem_c(A,B,C))
    print("problem_d:",problem_d(x,y))
    print("problem_e:",problem_e(A))
    print("problem_f:",problem_f(A, x))
    print("problem_g:",problem_g(A, x))
    print("problem_h:",problem_h(A, alpha))
    print("problem_i:",problem_i(A, i, j))
    print("problem_j:",problem_j(A, i))
    print("problem_k:",problem_k(A, c, d))
    print("problem_l:",problem_l(A, k))
    print("problem_m:",problem_m(x, k, m, s))
    print("problem_n:",problem_n(A))
