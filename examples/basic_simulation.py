import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../Optimus-Primal')
import optimusprimal.prox_operators as prox_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.primal_dual as primal_dual
import faradaydreams.measurement_operator as measurement_operator
import faradaydreams.models as models
import faradaydreams.plot_spectrum as plot_spectrum
import faradaydreams.convex_solvers as solvers
import numpy as np

output_dir = "output/"

options = {'tol': 1e-4, 'iter': 5000, 'update_iter': 50, 'record_iters': False}

c = 2.92 * 10**8
m_size = 512
ISNR = 20.
freq0 =  np.linspace(400, 800, m_size) * 10**6
dfreq = np.abs(freq0[1] - freq0[0])
lambda2 = (c/freq0)**2
lambda1 = np.sqrt(lambda2)
dlambda2 = (c/(freq0 - dfreq/2.))**2 - (c/(freq0 + dfreq/2.))**2
phi_max, phi_min, dphi = measurement_operator.phi_parameters(lambda2, dlambda2)
phi = np.arange(-phi_min, phi_min + 1, dphi)
x = models.deltas(phi, [1, 2, 4],[-40, 25, 0], [0, np.pi/5, -np.pi/3]) 
x += models.box(phi, 1, 10, 50, 0)
weights = np.ones(m_size)


m_op = measurement_operator.faraday_operator(lambda2, phi, weights, dlambda2)
m_op_true = measurement_operator.faraday_operator(lambda2, phi, weights)
y0 = m_op.dir_op(x) 
sigma = 10**(-ISNR/20.)  * np.linalg.norm(y0) * 1./ np.sqrt(m_size)
y = (y0 + np.random.normal(0, sigma, m_size) + 1j * np.random.normal(0, sigma, m_size)) * weights
y_true = m_op_true.dir_op(x)


nu, sol = linear_operators.power_method(m_op, x* 0 + 1, 1e-4)

#wav = ["dirac", "db2", "db4", "db6", "db8"]
#wav = ["db8"]
wav = ["dirac"]
levels = 4

z, diag = solvers.solver(solvers.algorithm.l1_constrained, weights * y, sigma, m_op, wav, levels, nu, 1e-2, options)
fname = output_dir+"sim_plot.png"
plot_spectrum.plot_simulation_spectrum(fname, lambda1, y_true, y, sigma, m_op, phi, x, z, 200)
