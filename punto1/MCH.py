import numpy as np
import matplotlib.pyplot as plt
data= np.loadtxt('datos_observacionales.dat')
t_obs = data[:,0]
x_obs = data[:,1]
y_obs = data[:,2]
z_obs = data[:,3]
sigma_total = np.ones(len(x_obs))
def model_x(y,sigma,t):
    for i in range(len(sigma)):
        x = y - np.exp(-sigma[i]*t)
    return x 
def model_y(x,z,rho,t):
    for i in range(len(rho)):
        y = x*(rho[i]-z)-np.exp(-t)
    return y    
def model_z(x,y,beta,t):
    for i in range(len(beta)):
        z = (1/beta)*(x*y-np.exp(-beta[i]*t))
    return z 
def loglikelihood(x_obs, y_obs, sigma_total, sigma,t_obs):
    d = y_obs -  model_x(y_obs, sigma,t_obs)
    d = d/sigma_total
    d = -0.5 * np.sum(d**2)
    return d
def loglikelihood_rho(x_obs, z_obs, sigma_total, rho,t_obs):
    d = y_obs -  model_y(y_obs,z_obs, rho,t_obs)
    d = d/sigma_total
    d = -0.5 * np.sum(d**2)
    return d
def loglikelihood_beta(x_obs, y_obs, sigma_total, beta, t_obs):
    d = y_obs -  model_z(x_obs, y_obs, beta ,t_obs)
    d = d/sigma_total
    d = -0.5 * np.sum(d**2)
    return d
def logprior(param):
    d = -0.5 * np.sum(param**2/(100.0)**2)
    return d
def divergence_loglikelihood(x_obs, y_obs, sigma_total, param, t_obs):
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood(x_obs, y_obs, sigma_total, param + delta_parameter,t_obs) 
        div[i] = div[i] - loglikelihood(x_obs, y_obs, sigma_total, param - delta_parameter,t_obs)
        div[i] = div[i]/(2.0 * delta)
    return div
def divergence_loglikelihood_y(x_obs, z_obs, sigma_total, param, t_obs):
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood_rho(x_obs, z_obs, sigma_total, param + delta_parameter,t_obs) 
        div[i] = div[i] - loglikelihood_rho(x_obs, z_obs, sigma_total, param - delta_parameter,t_obs)
        div[i] = div[i]/(2.0 * delta)
    return div
def divergence_loglikelihood_z(x_obs, y_obs, sigma_total, param, t_obs):
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood_beta(x_obs, y_obs, sigma_total, param + delta_parameter,t_obs) 
        div[i] = div[i] - loglikelihood_beta(x_obs, y_obs, sigma_total, param - delta_parameter,t_obs)
        div[i] = div[i]/(2.0 * delta)
    return div
def hamiltonian(x_obs, y_obs, sigma_total, param, param_momentum,t_obs):
    m = 100.0
    K = 0.5 * np.sum(param_momentum**2)/m
    V = -loglikelihood(x_obs, y_obs, sigma_total, param,t_obs)     
    return K + V
def hamiltonian_y(x_obs, z_obs, sigma_total, rho, rho_momentum,t_obs):
    m = 100.0
    K = 0.5 * np.sum(rho_momentum**2)/m
    V = -loglikelihood_rho(x_obs, z_obs, sigma_total, rho,t_obs)     
    return K + V
def hamiltonian_z(x_obs, y_obs, sigma_total, beta, beta_momentum,t_obs):
    m = 1.0
    K = 0.5 * np.sum(beta_momentum**2)/m
    V = -loglikelihood_beta(x_obs, y_obs, sigma_total, beta,t_obs)     
    return K + V
def leapfrog_proposal(x_obs, y_obs, sigma_total, param, param_momentum,t_obs):
    N_steps = 5
    delta_t = 1E-2
    m = 10.0
    new_param = param.copy()
    new_param_momentum = param_momentum.copy()
    for i in range(N_steps):
        new_param_momentum = new_param_momentum + divergence_loglikelihood(x_obs, y_obs, sigma_total, param,t_obs) * 0.5 * delta_t
        new_param = new_param + (new_param_momentum/m) * delta_t
        new_param_momentum = new_param_momentum + divergence_loglikelihood(x_obs, y_obs, sigma_total, param,t_obs) * 0.5 * delta_t
    new_param_momentum = -new_param_momentum
    return new_param, new_param_momentum
def leapfrog_proposal2(x_obs, z_obs, sigma_total, rho, rho_momentum,t_obs):
    N_steps = 5
    delta_t = 1E-2
    m = 100.0
    new_rho = rho.copy()
    new_rho_momentum = rho_momentum.copy()
    for i in range(N_steps):
        new_rho_momentum = new_rho_momentum + divergence_loglikelihood_y(x_obs, z_obs, sigma_total, rho,t_obs) * 0.5 * delta_t
        new_rho = new_rho + (new_rho_momentum/m) * delta_t
        new_rho_momentum = new_rho_momentum + divergence_loglikelihood_y(x_obs, z_obs, sigma_total, rho,t_obs) * 0.5 * delta_t
    new_rho_momentum = -new_rho_momentum
    return new_rho, new_rho_momentum
def leapfrog_proposal3(x_obs, y_obs, sigma_total, param, param_momentum,t_obs):
    N_steps = 5
    delta_t = 1E-2
    m = 100.0
    new_param = param.copy()
    new_param_momentum = param_momentum.copy()
    for i in range(N_steps):
        new_param_momentum = new_param_momentum + divergence_loglikelihood_z(x_obs, y_obs, sigma_total, param,t_obs) * 0.5 * delta_t
        new_param = new_param + (new_param_momentum/m) * delta_t
        new_param_momentum = new_param_momentum + divergence_loglikelihood_z(x_obs, y_obs, sigma_total, param,t_obs) * 0.5 * delta_t
    new_param_momentum = -new_param_momentum
    return new_param, new_param_momentum
def monte_carlo(x_obs, y_obs, z_obs,sigma_total,t_obs, N=5000):
    param = [np.random.random(1)]
    param_momentum = [np.random.normal(size=1)]
    rho = [np.random.random(1)]
    rho_momentum = [np.random.normal(size=1)]
    beta = [np.random.random(1)]
    beta_momentum = [np.random.normal(size=1)]
    
    for i in range(1,N):
        propuesta_param, propuesta_param_momentum = leapfrog_proposal(x_obs, y_obs, sigma_total, param[i-1], param_momentum[i-1],t_obs)
        energy_new = hamiltonian(x_obs, y_obs, sigma_total, propuesta_param, propuesta_param_momentum,t_obs)
        energy_old = hamiltonian(x_obs, y_obs, sigma_total, param[i-1], param_momentum[i-1],t_obs)
        
        propuesta_rho, propuesta_rho_momentum = leapfrog_proposal2(x_obs, z_obs, sigma_total, rho[i-1], rho_momentum[i-1],t_obs)
        energy_new2 = hamiltonian_y(x_obs, z_obs, sigma_total, propuesta_rho, propuesta_rho_momentum,t_obs)
        energy_old2 = hamiltonian_y(x_obs, z_obs, sigma_total, rho[i-1], rho_momentum[i-1],t_obs)
        
        propuesta_beta, propuesta_beta_momentum = leapfrog_proposal3(x_obs, y_obs, sigma_total, beta[i-1], beta_momentum[i-1],t_obs)
        energy_new3 = hamiltonian_z(x_obs, y_obs, sigma_total, propuesta_beta, propuesta_beta_momentum,t_obs)
        energy_old3 = hamiltonian_z(x_obs, y_obs, sigma_total, beta[i-1], beta_momentum[i-1],t_obs)
              
        r = min(1,np.exp(-(energy_new - energy_old)))
        r2 = min(1,np.exp(-(energy_new2 - energy_old2)))
        r3 = min(1,np.exp(-(energy_new3 - energy_old3)))
        
        alpha = np.random.random()
        if(alpha<r):
            param.append(propuesta_param)
           
        else:
            param.append(param[i-1])
        param_momentum.append(np.random.normal(size=1)) 
        if(alpha<r2):
            rho.append(propuesta_rho)
           
        else:
            rho.append(rho[i-1])
        rho_momentum.append(np.random.normal(size=1)) 
        if(alpha<r3):
            beta.append(propuesta_beta)
           
        else:
            beta.append(beta[i-1])
        beta_momentum.append(np.random.normal(size=1))     

    param = np.array(param)
    rho = np.array(rho)
    beta = np.array(beta)
    return param,rho,beta
param_chain = monte_carlo(x_obs, y_obs,z_obs, sigma_total,t_obs)
plt.hist(param_chain[0])
plt.title('Sigma')
plt.savefig('sigma.pdf')
plt.close()
plt.hist(param_chain[1])
plt.title('Rho')
plt.savefig('rho.pdf')
plt.close()
plt.hist(param_chain[2])
plt.title('Beta')
plt.savefig('beta.pdf')
plt.close()