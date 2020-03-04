import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime

# initialise
# h = 0
# dh_du = 0
# dh_dv = 0

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        # TODO: IMPLEMENT ME

    def backward (self, y, h):
        # TODO: IMPLEMENT ME
        dh_du = 0
        dh_dv = 0
        for _ in range(self.numHidden):
            dj_dy = y - ys
            dy_dh = self.w.T
            dh_dz = np.diag(1 - (np.tanh(self.U*h + self.V*xs)**2))
            a = np.zeros((len(xs), len(xs)), float)
            a = np.fill_diagonal(a, h.T)
            dz_du = a + self.U * dh_du
            b = np.zeros((len(xs), len(xs)), float)
            b = np.fill_diagonal(b, xs.T)
            dh_dv = np.dot(dh_dz, b) + self.U * dh_dv
            dj_du = dj_dy * dy_dh * dh_dz * dz_du
            dj_dv = dj_dy * dy_dh * dh_dv
            dh_du = dh_dz * dz_du
            dj_dw = (y - ys) * h
            self.U -= dj_du
            self.V -= dj_dv
            self.w -= dj_dw
        pass

    def forward (self, x):
        # TODO: IMPLEMENT ME
        h = np.zeros((6, 1))
        loss = 0
        z = np.dot(self.U, h) + self.V*x
        h = np.tanh(z)
        print(h)
        y = np.dot(h.T, self.w.T)
        print(y)
        return loss, y, h
        pass

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    print (xs)
    print (ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    # initialise
    #h = 0
    rnn = RNN(numHidden, numInput, 1).forward(xs)
    #loss = RNN(numHidden, numInput, 1).backward(ys)
    print(rnn)
    # TODO: IMPLEMENT ME
