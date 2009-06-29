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
	
def pp2p(q,acc=1e-6):
	n = len(q)
	q_act = ones_like(q)
	theta = q / sqrt(q[0])
	unstable = False
	
	for iter in range(1,100):
		#print iter
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

z = array([ .9, .6, .3])
p = poly(z)
print absolute(roots(p))
pp = p2pp(p)
print p
print pp2p(pp)
