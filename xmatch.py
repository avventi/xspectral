#!/usr/bin/env python
#
#       xmatch.py
#       
#       Copyright 2009 Enrico Avventi <avventi@kth.se>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

from numpy import *
import cvxopt.base as cb 
import cvxopt.solvers as solvers
from scipy.linalg.basic import hankel, toeplitz
from armalib import *

def cc_approx(cep,cov,alpha=1.0,beta=1.0,init_p=None,init_q=None,show_progress=False,
max_iter=100,mask_p=None,mask_q=None):
	""" Computes the power spactra approximatively matching given 
	cepstral and vovariance coefficients with weights alpha and 
	beta respectively."""

	m = len(cep)
	n = len(cov)
	if init_p is not None:
		i_p = iter2matrix(init_p/init_p[0])
	if init_q is not None:
		i_q = iter2matrix(init_q/init_p[0])
	
	if mask_p is None:
		# by default mask out the entropy
		rm = m-1
		Mp = cb.spmatrix([0.0], [rm-1], [m-1])
		Mp[rm::rm+1] = 1
	else:
		print "not supporting general masks yet"
		return None
	if mask_q is None:
		# no covariance lag is masked by default
		rn = n
		Mq = cb.spmatrix([0.0], [rn-1], [n-1])
		Mq[0::rn+1] = 1
	else:
		print "not supporting general masks yet"
		return None
		
	# for debugging purposes
	tmp_x = cb.matrix(1.0, (rm+rn+1,1))
	
	def F(x=None,z=None):
		if x is None:
			x = cb.matrix(0.0, (rm+rn+1,1))
			if init_p is None or init_q is None:
				x[rm] = 1
			else:
				x[:rm] = Mp*i_p
				x[rm:rm+rn] = Mq*i_q
			return 1, x
		xp = x[:rm]
		p = fromiter(Mp.T*xp, float)
		p[0] += 1
		xq = x[rm:rm+rn]
		q = fromiter(Mq*xq, float)
		num = pp2p(p)
		den = pp2p(q)
		if num is None or den is None:
			return None
		xcep = cb.matrix( arma2cep(num,den,m-1), (m,1))
		xcov = cb.matrix( arma2cov(num,den,n-1), (n,1))
		f = 0.5/alpha * xp.T*xp + 0.5/beta * xq.T*xq + xp.T*Mp*xcep -1 + xcep[0] - x[rm+rn]
		Df = cb.matrix(-1.0, (1,rm+rn+1))
		Df[0,0:rm] = (xp/alpha + Mp*xcep).T
		Df[0,rm:rm+rn]  = (xq/beta - Mq*xcov).T
		if z is None:
			return f, Df
		H = cb.matrix(0.0, (rm+rn+1,rm+rn+1))
		# evaluate hessian w.r.t p
		c_p = arma2cov(ones(1),num,2*m-2)
		H_pp = 0.5*toeplitz(c_p[:m]) + 0.5*hankel(c_p[:m], c_p[m-1:]) + eye(m)/alpha
		H[:rm,:rm] = Mp*cb.matrix(H_pp, (m,m))*Mp.T
		# evaluate hessian w.r.t q
		c_q = arma2cov(num,polymul(den,den),2*n-2)
		H_qq = 0.5*toeplitz(c_q[:n]) + 0.5*hankel(c_q[:n], c_q[n-1:]) + eye(n)/beta
		H[rm:rm+rn,rm:rm+rn] = Mq*cb.matrix(H_qq, (n,n))*Mq.T
		# evaluate the mixed part
		cc = arma2cov(ones(1),den,n+m-2)
		H_pq = -0.5*toeplitz(cc[:m], cc[:n]) - 0.5*hankel(cc[:m],cc[m-1:])
		H[:rm,rm:rm+rn] = Mp*cb.matrix(H_pq, (m,n))*Mq.T
		H[rm:rm+rn,:rm] = H[:rm,rm:rm+rn].T
		# save for debugging purposes
		for k in range(0,rm+rn+1):
			tmp_x[k] = x[k]
		return f, Df, z[0]*H
	
	solvers.options['maxiters'] = max_iter
	solvers.options['show_progress']= show_progress
	try:
		c = cb.matrix(1.0, (rm+rn+1,1))
		cep = iter2matrix(cep, (m,1))
		c[:rm] = -Mp*cep
		cov = iter2matrix(cov, (n,1))
		c[rm:rm+rn] = Mq*cov
		sol = solvers.cpl(c,F)
	except ArithmeticError: 
		p = fromiter(Mp.T*tmp_x[0:rm], float)
		p[0] += 1
		q = fromiter(Mq.T*tmp_x[rm:rm+rn], float)
		num = pp2p(p)
		den = pp2p(q)
		if show_progress:
			print tmp_x
			print abs(roots(num))
			print abs(roots(den))
			my_plot = plot_spectra()
			my_plot.add(p,q)
			my_plot.save("debug")
		raise
	if sol['status'] == 'unknown':
		return None, None, sol['status']
	x = sol['x'].T
	opt_p = fromiter(Mp.T*x[:rm], float)
	opt_p[0] += 1
	opt_q = fromiter(Mq.T*x[rm:rm+rn], float)
	return opt_p/opt_p[0], opt_q/opt_p[0], sol['status']
	
def ccx_iter(cep,cov,alpha=500,beta=500,max_iter=20):
	""" This function utilizes cc_approx with increasing values of weights 
	in order to converge faster. """
	# last convergent value
	act_p = zeros_like(cep)
	act_q = zeros_like(cov)
	act_p[0] = 1
	act_q[0] = 1
	#my_plot = plot_spectra()
	# next value to try
	dg = 1.0/max(alpha,beta)
	gamma = dg
	for k in range(0,max_iter):
		status = None
		try:
			(p, q, status) = cc_approx(cep,cov,gamma*alpha,gamma*beta,act_p,act_q,max_iter=20)
		except ArithmeticError:
			print k, gamma, 'err', gamma*alpha
			if gamma_lo is None:
				gamma = 0.1*gamma
			else:
				gamma = 0.5*gamma + 0.5*gamma_lo
		if status == 'optimal':
			print k, gamma, 'opt', gamma*alpha
			if gamma == 1:
				return act_p/act_p[0], act_q/act_p[0]
			gamma_lo = gamma
			if gamma > 1 - 10e-6:
				gamma = 1
			else:
				gamma = 0.5*gamma_lo + 0.5
			act_p = p
			act_q = q
			#my_plot.add(p,q)
		if status == 'unknown':
			print k, gamma, 'ukn', gamma*alpha
			if gamma_lo is None:
				gamma = 0.1*gamma
			else:
				gamma = 0.2*gamma + 0.8*gamma_lo
	#my_plot.save("all")
	return act_p/act_p[0], act_q/act_p[0]


j = complex(0,1)
z = array([.9,.9,.75,.75]) * exp(j*pi*array([.25,-.25,.5-.03,-.5+.03]))
num = real(poly(z))
z = array([.8,.8,.8,.8,.9,.9]) * exp(j*pi*array([.3,.2,-.3,-.2,.5,-.5]))
den = real(poly(z))
noise = random.randn(1000)
data = lfilter(num,den,noise) 
cep = estimate_cep(data,4)
cov = estimate_cov(data,6)
print 'est_cep', cep
print 'tru_cep', arma2cep(num,den,4) 
print 'est_cov', cov
print 'tru_cov', arma2cov(num,den,6)
p = p2pp(num)
q = p2pp(den)
(opt_p,opt_q) = ccx_iter(cep,cov,300,300,max_iter=60)	

#(opt_p,opt_q,status) = cc_approx(cep,cov,30,30,max_iter=100,show_progress=True)
opt_num = pp2p(opt_p)
opt_den = pp2p(opt_q)
print opt_p
print opt_q
print 'cep', cep 
print 'opt_cep', arma2cep(opt_num,opt_den,4)
print 'cov', cov
print 'opt_cov', arma2cov(opt_num,opt_den,6)

my_plot = plot_spectra()
my_plot.add(opt_p,opt_q)
my_plot.add(p,q)
my_plot.save("prova")
