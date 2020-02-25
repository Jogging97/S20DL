import numpy as np


class RNN:
    def __init__(self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        # TODO: IMPLEMENT ME

    def backward(self, x, y):
        dj_dw = 0
        dj_dU = 0
        dj_dV = 0
        y_hat = []
        dz_dU = np.zeros((1, 6))
        dz_dV = x[0]
        h = np.zeros((51, 6))
        z = np.zeros((50, 6))

        for i in range(1, 51):
            z[i - 1, :] = np.dot(h[i - 1, :], self.U) + np.dot(x[i - 1], self.V).T[0]
            h[i, :] = np.tanh(z[i - 1, :])
            y_hat.append(np.dot(self.w, h[i, :]))

            dj_dw = dj_dw + np.dot((y_hat[i - 1] - y[i - 1]), h[i, :])

            dz_dU = h[i - 1, :] + np.dot(dz_dU, np.dot(self.U, (np.diag(1 - np.square(h[i - 1, :])))))
            dj_dh = np.dot(y_hat[i - 1] - y[i - 1], self.w)
            dh_dz = 1 - np.square(h[i, :])
            dj_dz = np.dot(dj_dh, dh_dz)

            dj_dU = dj_dU + np.dot(dj_dz, dz_dU[0])

            dz_dV = x[i - 1] + np.dot(dz_dV, np.dot(self.U, (np.diag(1 - np.square(h[i - 1, :])))))

            dj_dV = dj_dV + np.dot(dj_dz, dz_dV[0])

        return dj_dw, dj_dU, dj_dV, h, z

    def forward(self, x, y, h, z):
        y_hat = []
        for i in range(1, 51):
            y_hat.append(np.dot(h[i].T, self.w))
        loss = np.sum(0.5 * np.square(y_hat - y))
        return loss

    def SGD(self, x, y):

        loss = 1 # Give any value greater than 0.05
        while loss >= 0.05:
            dj_dw, dj_dU, dj_dV, h, z = self.backward(x, y)
            self.w = self.w - 1e-2 * dj_dw
            self.U = self.U - 1e-7 * dj_dU
            self.V = self.V - 1e-8 * dj_dV  # trail and error used for learning rates
            loss = self.forward(x, y, h, z)
            print(loss)


# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData():
    total_series_length = 50
    echo_step = 2  # 2-back task
    # batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)


if __name__ == "__main__":
    xs, ys = generateData()
    print(xs)
    print(ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)

    # Calling the SGD function that in turns calls all the other functions required

    rnn.SGD(np.array(xs), np.array(ys))