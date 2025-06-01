import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, dropout_rate=0.3):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.bn_means = []
        self.bn_vars = []
        self.bn_gamma = []
        self.bn_beta = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.random.randn(layer_sizes[i + 1]) * 0.01
            self.weights.append(w)
            self.biases.append(b)
            if i < self.num_layers - 2:  # Batch norm for hidden layers
                self.bn_means.append(np.zeros(layer_sizes[i + 1]))
                self.bn_vars.append(np.ones(layer_sizes[i + 1]))
                self.bn_gamma.append(np.ones(layer_sizes[i + 1]))
                self.bn_beta.append(np.zeros(layer_sizes[i + 1]))
        self.activations = []
        self.z_values = []
        self.bn_cache = []
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.m_bn_gamma = [np.zeros_like(g) for g in self.bn_gamma]
        self.v_bn_gamma = [np.zeros_like(g) for g in self.bn_gamma]
        self.m_bn_beta = [np.zeros_like(b) for b in self.bn_beta]
        self.v_bn_beta = [np.zeros_like(b) for b in self.bn_beta]
        self.t = 0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def dropout(self, x, training=True):
        if training:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            return x * mask / (1 - self.dropout_rate)
        return x

    def batch_norm_forward(self, x, mean, var, gamma, beta, training=True, epsilon=1e-5):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_norm = (x - batch_mean) / np.sqrt(batch_var + epsilon)
            out = gamma * x_norm + beta
            cache = (x, x_norm, batch_mean, batch_var, gamma, epsilon)
            return out, cache, batch_mean, batch_var
        else:
            x_norm = (x - mean) / np.sqrt(var + epsilon)
            return gamma * x_norm + beta, None, None, None

    def batch_norm_backward(self, delta, cache):
        x, x_norm, batch_mean, batch_var, gamma, epsilon = cache
        N = x.shape[0]
        dgamma = np.sum(delta * x_norm, axis=0)
        dbeta = np.sum(delta, axis=0)
        dx_norm = delta * gamma
        dvar = np.sum(dx_norm * (x - batch_mean) * -0.5 * (batch_var + epsilon) ** -1.5, axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(batch_var + epsilon), axis=0) + dvar * np.sum(-2 * (x - batch_mean), axis=0) / N
        dx = dx_norm / np.sqrt(batch_var + epsilon) + dvar * 2 * (x - batch_mean) / N + dmean / N
        return dx, dgamma, dbeta

    def forward(self, inputs, training=True):
        self.activations = [inputs]
        self.z_values = []
        self.bn_cache = []
        activation = inputs
        for i in range(self.num_layers - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i < self.num_layers - 2:
                z, cache, mean, var = self.batch_norm_forward(z, self.bn_means[i], self.bn_vars[i], self.bn_gamma[i], self.bn_beta[i], training)
                self.bn_cache.append(cache)
                if training:
                    self.bn_means[i] = 0.9 * self.bn_means[i] + 0.1 * mean
                    self.bn_vars[i] = 0.9 * self.bn_vars[i] + 0.1 * var
                activation = self.relu(z)
                activation = self.dropout(activation, training)
            else:
                activation = z  # Linear output
            self.activations.append(activation)
        return self.activations[-1]

    def clip_gradients(self, grads, max_norm=1.0):
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            return [g * scale for g in grads]
        return grads

    def adam_update(self, grads, m, v, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** self.t)
        v_hat = v / (1 - beta2 ** self.t)
        return lr * m_hat / (np.sqrt(v_hat) + epsilon), m, v

    def backpropagation(self, inputs, target, learning_rate=0.001):
        activations = self.forward(inputs, training=True)
        output = activations[-1]
        error = output - target
        delta = error
        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]
        bn_gamma_grads = [np.zeros_like(g) for g in self.bn_gamma]
        bn_beta_grads = [np.zeros_like(b) for b in self.bn_beta]
        for l in range(self.num_layers - 2, -1, -1):
            weight_grads[l] = np.outer(self.activations[l], delta)
            bias_grads[l] = delta
            if l > 0:
                error = np.dot(delta, self.weights[l].T)
                if l < self.num_layers - 2:
                    delta, dgamma, dbeta = self.batch_norm_backward(error, self.bn_cache[l - 1])
                    bn_gamma_grads[l - 1] = dgamma
                    bn_beta_grads[l - 1] = dbeta
                else:
                    delta = error
                delta = delta * self.relu_derivative(self.z_values[l - 1])
        weight_grads = self.clip_gradients(weight_grads)
        bias_grads = self.clip_gradients(bias_grads)
        bn_gamma_grads = self.clip_gradients(bn_gamma_grads)
        bn_beta_grads = self.clip_gradients(bn_beta_grads)
        for l in range(self.num_layers - 1):
            dw, self.m_w[l], self.v_w[l] = self.adam_update(weight_grads[l], self.m_w[l], self.v_w[l], lr=learning_rate)
            db, self.m_b[l], self.v_b[l] = self.adam_update(bias_grads[l], self.m_b[l], self.v_b[l], lr=learning_rate)
            self.weights[l] -= dw
            self.biases[l] -= db
            if l < self.num_layers - 2:
                dgamma, self.m_bn_gamma[l], self.v_bn_gamma[l] = self.adam_update(bn_gamma_grads[l], self.m_bn_gamma[l], self.v_bn_gamma[l], lr=learning_rate)
                dbeta, self.m_bn_beta[l], self.v_bn_beta[l] = self.adam_update(bn_beta_grads[l], self.m_bn_beta[l], self.v_bn_beta[l], lr=learning_rate)
                self.bn_gamma[l] -= dgamma
                self.bn_beta[l] -= dbeta

    def train_batch(self, inputs_batch, targets_batch, learning_rate=0.001):
        for inputs, targets in zip(inputs_batch, targets_batch):
            self.backpropagation(inputs, targets, learning_rate)

    def predict(self, inputs):
        return self.forward(inputs, training=False)

    def save(self, filename):
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'bn_means': self.bn_means,
            'bn_vars': self.bn_vars,
            'bn_gamma': self.bn_gamma,
            'bn_beta': self.bn_beta,
            'layer_sizes': self.layer_sizes
        }
        np.save(filename, data, allow_pickle=True)

    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.weights = data['weights']
        self.biases = data['biases']
        self.bn_means = data.get('bn_means', [np.zeros(s) for s in self.layer_sizes[1:-1]])
        self.bn_vars = data.get('bn_vars', [np.ones(s) for s in self.layer_sizes[1:-1]])
        self.bn_gamma = data.get('bn_gamma', [np.ones(s) for s in self.layer_sizes[1:-1]])
        self.bn_beta = data.get('bn_beta', [np.zeros(s) for s in self.layer_sizes[1:-1]])
        self.layer_sizes = data['layer_sizes']
        self.num_layers = len(self.layer_sizes)
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.m_bn_gamma = [np.zeros_like(g) for g in self.bn_gamma]
        self.v_bn_gamma = [np.zeros_like(g) for g in self.bn_gamma]
        self.m_bn_beta = [np.zeros_like(b) for b in self.bn_beta]
        self.v_bn_beta = [np.zeros_like(b) for b in self.bn_beta]
        self.t = 0