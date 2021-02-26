from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spdiag, mul, div, sparse 
from cvxopt import spmatrix, sqrt, base


def l1(P, q):

    """
    Returns the solution u of the ell-1 approximation problem

        (primal) minimize ||P*u - q||_1       
    
        (dual)   maximize    q'*w
                 subject to  P'*w = 0
                             ||w||_infty <= 1.
    """

    m, n = P.size

    # Solve equivalent LP 
    #
    #     minimize    [0; 1]' * [u; v]
    #     subject to  [P, -I; -P, -I] * [u; v] <= [q; -q]
    #
    #     maximize    -[q; -q]' * z 
    #     subject to  [P', -P']*z  = 0
    #                 [-I, -I]*z + 1 = 0 
    #                 z >= 0 
    
    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        if trans == 'N':
            # y := alpha * [P, -I; -P, -I] * x + beta*y
            u = P*x[:n]
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]

        else:
            # y := alpha * [P', -P'; -I, -I] * x + beta*y
            y[:n] =  alpha * P.T * (x[:m] - x[m:]) + beta*y[:n]
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]


    def Fkkt(W): 

        # Returns a function f(x, y, z) that solves
        #
        # [ 0  0  P'      -P'      ] [ x[:n] ]   [ bx[:n] ]
        # [ 0  0 -I       -I       ] [ x[n:] ]   [ bx[n:] ]
        # [ P -I -W1^2     0       ] [ z[:m] ] = [ bz[:m] ]
        # [-P -I  0       -W2      ] [ z[m:] ]   [ bz[m:] ]
        #
        # On entry bx, bz are stored in x, z.
        # On exit x, z contain the solution, with z scaled (W['di'] .* z is
        # returned instead of z). 

        d1, d2 = W['d'][:m], W['d'][m:]
        D = 4*(d1**2 + d2**2)**-1
        A = P.T * spdiag(D) * P
        lapack.potrf(A)

        def f(x, y, z):

            x[:n] += P.T * ( mul( div(d2**2 - d1**2, d1**2 + d2**2), x[n:]) 
                + mul( .5*D, z[:m]-z[m:] ) )
            lapack.potrs(A, x)

            u = P*x[:n]
            x[n:] =  div( x[n:] - div(z[:m], d1**2) - div(z[m:], d2**2) + 
                mul(d1**-2 - d2**-2, u), d1**-2 + d2**-2 )

            z[:m] = div(u-x[n:]-z[:m], d1)
            z[m:] = div(-u-x[n:]-z[m:], d2)

        return f


    # Initial primal and dual points from least-squares solution.

    # uls minimizes ||P*u-q||_2; rls is the LS residual.
    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q 

    # x0 = [ uls;  1.1*abs(rls) ];   s0 = [q;-q] - [P,-I; -P,-I] * x0
    x0 = matrix( [uls[:n],  1.1*abs(rls)] ) 
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1) 

    # z0 = [ (1+w)/2; (1-w)/2 ] where w = (.9/||rls||_inf) * rls  
    # if rls is nonzero and w = 0 otherwise.
    if max(abs(rls)) > 1e-10:  
        w = .9/max(abs(rls)) * rls
    else: 
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,  
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n]


