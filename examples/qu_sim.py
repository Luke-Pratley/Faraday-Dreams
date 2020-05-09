import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../Optimus-Primal')
import optimusprimal.prox_operators as prox_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.primal_dual as primal_dual
import faradaydreams.measurement_operator as measurement_operator
import faradaydreams.qu_fitting as qu_fitting
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy import signal

output_dir = "output/"

options = {'tol': 1e-4, 'iter': 5000, 'update_iter': 50, 'record_iters': False}

c = 2.92 * 10**8
m_size = 1024
down_sample = 8
ISNR = 20.
freq0 =  np.linspace(400, 800, m_size) * 10**6
dfreq = np.abs(freq0[1] - freq0[0])
lambda2 = (c/freq0)**2
lambda1 = np.sqrt(lambda2)
dlambda2 = (c/(freq0 - dfreq/2.))**2 - (c/(freq0 + dfreq/2.))**2
phi_max, phi_min, dphi = measurement_operator.phi_parameters(lambda2, dlambda2)



def component(lambda2, Q, U, rm):
    return Q * np.cos(2 * rm * lambda2) + 1j * U * np.sin(2 * rm * lambda2)

def model(lambda2, Q1, U1, rm1, Q2, U2, rm2):
    return component(lambda2, Q1, U1, rm1) + component(lambda2, Q2, U2, rm2)
def model_averaging(lambda2, Q1, U1, rm1, Q2, U2, rm2):
    return signal.decimate(model(lambda2, Q1, U1, rm1, Q2, U2, rm2), q=down_sample, n=4)
def model_no_averaging(lambda2, Q1, U1, rm1, Q2, U2, rm2):
    return signal.decimate(model(lambda2, Q1, U1, rm1, Q2, U2, rm2), q=down_sample, n=0)

model_choice =  model_no_averaging

true_params = (3, -4, 10, 5, 6, -200)

y0 = model_averaging(lambda2, *true_params)
sigma = 10**(-ISNR/20.)  * np.linalg.norm(y0) * 1./ np.sqrt(m_size * 2 * down_sample)
y = (y0 + np.random.normal(0, sigma, int(m_size/down_sample)) + 1j * np.random.normal(0, sigma, int(m_size/down_sample)))

estimate = np.array(true_params) * (1 + 0. * np.random.uniform(len(true_params)))
results = qu_fitting.least_squares(estimate, model_choice, lambda2, y, sigma)
print(results.x)
print(estimate)
print(true_params)

fig1, ax1 = plt.subplots(2, figsize=(10, 7), sharex=True)
ax1[0].plot(signal.decimate(lambda2, q=down_sample, n=0), np.real(model_averaging(lambda2, *true_params)))
ax1[0].plot(signal.decimate(lambda2, q=down_sample, n=0), np.imag(model_averaging(lambda2, *true_params)))
ax1[0].plot(lambda2, np.real(model(lambda2, *true_params)))
ax1[0].plot(lambda2, np.imag(model(lambda2, *true_params)))
ax1[0].legend(["Measured Q", "Measured U", "True Q", "True U"])
ax1[1].plot(lambda2, np.real(model(lambda2, *results.x)))
ax1[1].plot(lambda2, np.imag(model(lambda2, *results.x)))
ax1[1].plot(lambda2, np.real(model(lambda2, *true_params)))
ax1[1].plot(lambda2, np.imag(model(lambda2, *true_params)))
ax1[1].legend(["Q Fit", "U Fit"])
plt.show()
def constraint(P1, rm1, chi01, P2, rm2, chi02):
    return False


sampler = qu_fitting.mcmc_sample(model_choice, lambda2, y, sigma, results.x, constraint, 100, 5000)

labels = ["$Q_{01}$", "$U_{01}$", "$\phi_1$", "$Q_{01}$", "$U_{01}$", "$\phi_2$"]
fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()

flat_samples = sampler.get_chain(discard=1000, thin=40, flat=True)
print(flat_samples.shape)
fig = corner.corner(flat_samples, labels=labels, truths=[t for t in true_params], quantiles=[0.16, 0.5, 0.84])
plt.show()
