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

def cc_approx(cep,cov,alpha=1,beta=1):
	""" Computes the power spactra approximatively matching given 
	cepstral and vovariance coefficients with weights alpha and 
	beta respectively."""
	
	m = len(cep)
	n = len(cov)
	
	#N = 10000
	#cos_mat = zeros((max(n,m),N))
	#for k in range(0,N):
		#for kk in range(0,max(n,m)):
			#cos_mat[kk,k] = cos(2*pi*k*kk/N)
	
	# x = [p_0-1, p_1, ... p_m, q_0, q_1, ... q_n]'
	def F(x=None,z=None):
		if x is None:
			x = cb.matrix(0.0, (m+n,1))
			x[m] = 1
			return 0, x
		p = zeros(m) 
		p[:] = x.T[0,0:m]
		p[0] += 1
		#p_min = min( dot(p,  cos_mat[0:m,:]) )
		#if p_min <= 0:
			#return None
		q = zeros(n)
		q[:] = x.T[0,m:]
		#q_min = min( dot(q,  cos_mat[0:n,:]) )
		#if q_min <= 0:
			#return None
		num = pp2p(p)
		den = pp2p(q)
		p[0] -= 1
		if num is None or den is None:
			return None
		xcep = arma2cep(num,den,m-1)
		xcov = arma2cov(num,den,n-1)
		f = dot(cov,q) - dot(cep,p) + 0.5/alpha * dot(p,p) +\
			0.5/beta * dot(q,q) + dot(xcep,p) - p[0] -1 + xcep[0]
		Df = cb.matrix(0.0, (1,m+n))
		Df[0,0:m] = xcep - cep + p/alpha
		Df[0,m:]  = cov - xcov + q/beta
		if z is None:
			return f, Df
		H = cb.matrix(0.0, (m+n,m+n))
		# evaluate hessian w.r.t p
		c_p = arma2cov(ones(1),num,2*m-2)
		H_pp = 0.5*toeplitz(c_p[:m]) + 0.5*hankel(c_p[:m], c_p[m-1:])\
			+ eye(m)/alpha
		H[:m,:m] = H_pp
		# evaluate hessian w.r.t q
		c_q = arma2cov(num,polymul(den,den),2*n-2)
		H_qq = 0.5*toeplitz(c_q[:n]) + 0.5*hankel(c_q[:n], c_q[n-1:])\
			+ eye(n)/beta
		H[m:,m:] = H_qq
		# evaluate the mixed part
		cc = arma2cov(ones(1),den,n+m-2)
		H_pq = -0.5*toeplitz(cc[:m], cc[:n]) - 0.5*hankel(cc[:m],cc[m-1:])
		H[m:,:m] = H_pq.T
		H[:m,m:] = H_pq
		return f, Df, H
	
	solvers.options['maxiters'] = 500
	sol = solvers.cp(F)
	x = sol['x'].T
	print size(x)
	opt_p = zeros(m)
	opt_q = zeros(n)
	opt_p[:] = x[0,:m]
	opt_p[0] += 1
	opt_q[:] = x[0,m:]
	return opt_p/opt_p[0], opt_q/opt_p[0]

z = array([ .9, .6, .3])
num = poly(z)
z = array([ .8, .4, .2])
den = poly(z)
noise = random.randn(1000)
data = lfilter(num,den,noise) 
cep = arma2cep(num,den,3)#estimate_cep(data,3)
cov = arma2cov(num,den,3)#estimate_cov(data,3)
(opt_p,opt_q) = cc_approx(cep,cov,100,100)
p = p2pp(num)
q = p2pp(den)
print p/p[0]
print q/q[0]
print opt_p
print opt_q
