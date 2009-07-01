#!/usr/bin/env python
#
#       armalib.py
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


def p2pp(p):
	pp = convolve(p[::-1],p)
	return pp[len(p)-1:]

	
def pp2p(q,acc=1e-6,max_iter=100):
	n = len(q)
	q_act = ones_like(q)
	theta = q / sqrt(q[0])
	unstable = False
	
	for iter in range(0,max_iter):
		p = zeros(n-1)
		p[0] = - theta[n-1]/theta[0]
		phi = zeros([n-1,n-1])
		phi[0,:] = theta[0:n-1]+p[0]*theta[n-1:0:-1]
  
  		for k in range(1,n-1):
  			p[k] = -phi[k-1,n-2]/phi[k-1,k-1]
 			phi[k,k:] = phi[k-1,k-1:n-2]+p[k]*phi[k-1,:k-1:-1]

  		if max(fabs(p)) > 1:
  			unstable=True
  			return None
  			
  		q_act = p2pp(theta)
  		qq = q+q_act
  
		y = zeros(n)  
		y[0] = 0.5*qq[0]/phi[n-2,n-2]
		y[n-1] = qq[n-1]/phi[0,0] 
  
  		for k in range(2,n):
  			y[n-k]=(qq[n-k]-dot(y[n+1-k:], phi[k-2::-1,k-1]))/phi[k-1,k-1]
		
		for k in range(1,n):
			y[0:k+1]=y[0:k+1]+p[n-k-1]*y[k::-1]
  		theta=y
  
		if max(fabs(q-q_act)) < acc:
			return theta


def arma2cep(num,den,N):
	n = len(den) -1
	m = len(num) -1
	# pad zeroes if necessary
	if m<N: 
		num = concatenate((num,zeros(N-m)))
	if n<N:
		den = concatenate((den,zeros(N-n)))
	cep = zeros(N+1)
	cep[0] = 2*log(abs(num[0])) - 2*log(abs(den[0]))	# entropy
	num = num/num[0]
	den = den/den[0]
	s_num = zeros(N)
	s_den = zeros(N)
	s_num[0] = -num[1]
	s_den[0] = -den[1]
	for k in range(1,N):
		s_num[k] = -(k+1)*num[k+1] - dot(num[k:0:-1], s_num[:k])
		s_den[k] = -(k+1)*den[k+1] - dot(den[k:0:-1], s_den[:k])
	cep[1:] = (s_den-s_num) / range(1,N+1)
	return cep

def arma2cov(num,den,N):
	n = len(den) -1
	m = len(num) -1
	# pad zeroes if necessary
	if m<n: 
		num = concatenate((num,zeros(n-m)))
	if n<m:
		den = concatenate((den,zeros(m-n)))
		n = m
	# Computing K, the reflection coefficients of the AR-part up to order n, 
	# using Levinson inverse algorithm.
	K = zeros(n+1)
	alpha = zeros([n+1,n+1])
	alpha[n,:] = den/den[0]
	for k in range(n-1,-1,-1):
		K[k] = -alpha[k+1,k+1]
		if abs(K[k]) == 1:
			return None
		alpha[k,1:k+1] = (alpha[k+1,1:k+1]+K[k]*alpha[k+1,k:0:-1])/(1-K[k]**2)
	# Computing g, the covariance coefficent of the AR-part up to order n,
	# applying Shur inverse algorithm to the reflection coefficents K.
	yu = zeros([n+1,n+1])
	yut = zeros([n+1,n+1])
	g = zeros(N+n+1)
	yu[0,0] = 1 / prod(1-K**2)
	yut[0,0] = yu[0,0]
	g[0] = yu[0,0]
	for k in range(0,n):
		for kk in range(k,-1,-1):
			yu[k+1,kk] = yu[k+1,kk+1] + K[kk]*yut[k-kk,kk]
			yut[k-kk,kk+1] = -K[kk]*yu[k+1,kk+1] + (1-K[kk]**2)*yut[k-kk,kk]
		yut[k+1,kk] = yu[k+1,kk]
		g[k+1] = yu[k+1,kk]
	# Comtinue the covariance sequence of the AR-part up to n+N order
	for k in range(n,n+N):
		g[k+1] = -dot(den[n:0:-1], g[k-n+1:k+1])/den[0]
	# Take into account the effect of the MA-part
	gg = concatenate((g[::-1],g[1:]))
	pp = convolve(num[::-1],num)
	cov = convolve(gg,pp)
	cov = cov[N+2*n:2*N+2*n+1]/den[0]**2
	return cov

z = array([ .9, .6, .3])
num = poly(z)
z = array([ .8, .4, .2])
den = poly(z)
print arma2cov(num,den,3)
