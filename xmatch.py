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

def cc_approx(cep,cov,alpha=1,beta=1,init_p=None,init_q=None,show_progress=False,max_iter=100):
	""" Computes the power spactra approximatively matching given 
	cepstral and vovariance coefficients with weights alpha and 
	beta respectively."""
	
	m = len(cep)
	n = len(cov)
	# for debugging purposes
	tmp_x = cb.matrix(1.0, (m+n,1))
	
	def F(x=None,z=None):
		if x is None:
			x = cb.matrix(0.0, (m+n+1,1))
			if init_p is None or init_q is None:
				x[m] = 1
			else:
				x[:m] = init_p[:]
				x[0] -= 1
				x[m:m+n] = init_q[:]
			return 1, x
		p = zeros(m) 
		p[:] = x.T[0,0:m]
		p[0] += 1
		q = zeros(n)
		q[:] = x.T[0,m:m+n]
		num = pp2p(p)
		den = pp2p(q)
		p[0] -= 1
		if num is None or den is None:
			return None
		xcep = arma2cep(num,den,m-1)
		xcov = arma2cov(num,den,n-1)
		f = 0.5/alpha * dot(p,p) + 0.5/beta * dot(q,q) + dot(xcep,p) - p[0] -1 + xcep[0] - x[m+n]
		Df = cb.matrix(-1.0, (1,m+n+1))
		Df[0,0:m] = p/alpha + xcep
		Df[0,m:m+n]  = q/beta - xcov
		if z is None:
			return f, Df
		H = cb.matrix(0.0, (m+n+1,m+n+1))
		# evaluate hessian w.r.t p
		c_p = arma2cov(ones(1),num,2*m-2)
		H_pp = 0.5*toeplitz(c_p[:m]) + 0.5*hankel(c_p[:m], c_p[m-1:]) + eye(m)/alpha
		H[:m,:m] = H_pp
		# evaluate hessian w.r.t q
		c_q = arma2cov(num,polymul(den,den),2*n-2)
		H_qq = 0.5*toeplitz(c_q[:n]) + 0.5*hankel(c_q[:n], c_q[n-1:]) + eye(n)/beta
		H[m:m+n,m:m+n] = H_qq
		# evaluate the mixed part
		cc = arma2cov(ones(1),den,n+m-2)
		H_pq = -0.5*toeplitz(cc[:m], cc[:n]) - 0.5*hankel(cc[:m],cc[m-1:])
		H[m:m+n,:m] = H_pq.T
		H[:m,m:m+n] = H_pq
		# save for debugging purposes
		for k in range(0,m+n):
			tmp_x[k] = x[k]
		return f, Df, z[0]*H
	
	solvers.options['maxiters'] = max_iter
	solvers.options['show_progress']= show_progress
	try:
		c = cb.matrix(1.0, (m+n+1,1))
		c[:m] = -cep[:m]
		c[m:m+n] = cov[:n]
		sol = solvers.cpl(c,F)
	except ArithmeticError:
		p = zeros(m) 
		p[:] = tmp_x.T[0,0:m]
		p[0] += 1
		q = zeros(n)
		q[:] = tmp_x.T[0,m:]
		num = pp2p(p)
		den = pp2p(q)
		if show_progress:
			print tmp_x
			print abs(roots(num))
			print abs(roots(den))
		#my_plot = plot_spectra()
		#my_plot.add(p,q)
		#my_plot.save("debug")
		raise
	x = sol['x'].T
	opt_p = zeros(m)
	opt_q = zeros(n)
	opt_p[:] = x[0,:m]
	opt_p[0] += 1
	opt_q[:] = x[0,m:m+n]
	return opt_p/opt_p[0], opt_q/opt_p[0], sol['status']
	
def ccx_iter(cov,cep,alpha=500,beta=500,max_iter=10):
	""" This function utilizes cc_approx with increasing values of weights 
	in order to converge faster. """
	gamma_hi = 1
	gamma_lo = None
	act_p = zeros_like(cep)
	act_q = zeros_like(cov)
	act_p[0] = 1
	act_q[0] = 1
	for k in range(0,max_iter):
		status = None
		if gamma_lo is None:
			gamma = 0.5*gamma_hi
		else:
			gamma = 0.5*(gamma_lo + gamma_hi)
		try:
			(p, q, status) = cc_approx(cep,cov,gamma*alpha,gamma*beta,act_p,act_q,max_iter=50)
		except ArithmeticError:
			gamma_hi = gamma
			print k, gamma, 'err', gamma_lo, gamma_hi
		if status == 'optimal':
			gamma_lo = gamma
			act_p = p
			act_q = q
			print k, gamma, 'opt', gamma_lo, gamma_hi
		if status == 'unknown':
			gamma_hi = gamma
			print k, gamma, 'ukn', gamma_lo, gamma_hi
		
	return act_p/act_p[0], act_q/act_p[0] 
				
				
			 
	
	 

j = complex(0,1)
z = array([.9,.9,.75,.75]) * exp(j*pi*array([.25,-.25,.5-.03,-.5+.03]))
num = real(poly(z))
z = array([.8,.8,.8,.8,.9,.9]) * exp(j*pi*array([.3,.2,-.3,-.2,.5,-.5]))
den = real(poly(z))
noise = random.randn(1000)
data = lfilter(num,den,noise) 
cep = arma2cep(num,den,4)#estimate_cep(data,4)
cov = arma2cov(num,den,6)#estimate_cov(data,6)
p = p2pp(num)
q = p2pp(den)
(opt_p,opt_q) = ccx_iter(cep,cov,300,300)
(opt_p2,opt_q2,status) = cc_approx(cep,cov,300,300,p,q)
opt_num = pp2p(opt_p)
opt_den = pp2p(opt_q)
print opt_p
print opt_q
print 'cep', cep
print 'opt_cep', arma2cep(opt_num,opt_den,4)
print 'cov', cov
print 'opt_cov', arma2cov(opt_num,opt_den,6)

my_plot = plot_spectra()
my_plot.add(opt_p2,opt_q2)
my_plot.add(opt_p,opt_q)
my_plot.add(p,q)
my_plot.save("prova")
