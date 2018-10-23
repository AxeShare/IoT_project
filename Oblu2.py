import numpy as np
import os
import matplotlib.pyplot as plt

#Predict Step
def predict(A, x_L, P, Q):
        _x = np.matmul(A, x_L)
        _P = np.matmul(A, np.matmul(P,A.transpose())) + Q
        return _x, _P

#Update Step
def update(H, _P, R, z, _x):
        S = np.matmul(H, (np.matmul(_P, H.transpose()))) + R
        V = z - np.matmul(H,  _x)
        K = np.matmul(_P, np.matmul(H.transpose(),  np.linalg.inv(S)))

        _x = _x + np.matmul(K, V)
        _P = _P - np.matmul(K, np.matmul(S, K.transpose()))
        return _x, _P

P = 2*np.identity(3)
Q = 2.5*np.identity(3)
H = np.identity(3)
plt.figure()


lineL = [line.rstrip('\n') for line in open('stepsL.txt')]
lineR = [line.rstrip('\n') for line in open('stepsR.txt')]
for i in range(len(lineL)):
	F = map(float, lineL[i].split(','))
	p = F[4]
	x_L = [F[1], F[2], F[3]]
	G = map(float, lineR[i].split(','))
	z = [G[1],G[2],G[3]]
	cov1, cov2, cov3, cov4, cov5, cov6 = G[4], G[5], G[6], G[7], G[8], G[9]

	theta = np.radians(p)
	c, s = np.cos(theta), np.sin(theta)
	A = np.array((((c,-s,0), (s, c, 0), (0,0,1))))

	R = np.zeros((3,3))
	R[0,:]= [cov1, cov2, cov3]
	R[1,:]= [cov2, cov4, cov5]
	R[2,:]= [cov3, cov5, cov6]

	_x, _P = predict(A, x_L, P, Q)
	x_final, P_final = update(H, _P, R, z, _x)
	print (x_final, P_final)
        #plt.subplot(2, 1, 1) 
        plt.plot(F[1], F[2], '-o', c = 'black')
	#plt.subplot(2, 1, 2)	
	plt.plot(G[1], G[2], '-o',c='red') 
	plt.plot(x_final[0], x_final[1], '-o',c='blue') 
	plt.ylim(-1,1)
	plt.xlim(-1,1)
	plt.show()



"""while True:
        while(os.stat("stepsL.txt").st_size == 0):
            pass
	f = open('stepsL.txt', 'r')
	F = map(float, f.read().split(','))
	p = F[4]
	x_L = [F[1], F[2], F[3]]

        while(os.stat("stepsR.txt").st_size == 0):
            pass
	g = open('stepsR.txt', 'r')
        G = map(float, g.read().split(','))
	z = [G[1],G[2],G[3]]
	cov1, cov2, cov3, cov4, cov5, cov6 = G[4],G[5],G[6],G[7],G[8],G[9]

	theta = np.radians(p)
	c, s = np.cos(theta), np.sin(theta)
	A = np.array((((c,-s,0), (s, c, 0), (0,0,1))))

	R = np.zeros((3,3))
	R[0,:]= [cov1, cov2, cov3]
	R[1,:]= [cov2, cov4, cov5]
	R[2,:]= [cov3, cov5, cov6]

	_x, _P = predict(A, x_L, P, Q)
	x_final, P_final = update(H, _P, R, z, _x)
	print (x_final, P_final)
        
	plt.subplot(2, 1, 1) 
        plt.plot(F[1], F[2], '-x', c = 'black')
	plt.subplot(2, 1, 2)	
	plt.plot(G[1], G[2], '-o',c='red') 
	plt.show()   
"""


