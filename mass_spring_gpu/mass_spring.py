import warp as wp
import numpy as np
import copy
from cmath import inf
import os
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import pygame       # pygame for visualization
pygame.init()
wp.init()

# I wanted to have q and a long 1D array that has 2N entries, 
# but I think Warp will work better using the wp.vec2 option 
#Box will be centered at (0,0) for this script

### Parameters ###
k_spring = 2.0 # Spring stiffness constant
initial_stretch = 1.4
h = .0004 # timestep in seconds
rho = 1000 # density
side_len = 1
n_seg = 4
seg_len = side_len / n_seg
dim = n_seg + 1
m = rho * side_len * side_len / (dim * dim) # calculate node mass evenly



### Create a lattice-like square mesh of points ###
def generate_warp_mesh(side_len, n_seg):
    # Generate grid points, q will be a 2D vector with x and y positions
    # pos = q = [(x1, y1), (x2, y2), (x3, y3), ..., (xn, yn)]
    q = []
    for i in range(dim):
        for j in range(dim):
            x = -side_len / 2 + i * seg_len
            y = -side_len / 2 + j * seg_len
            q.append((x, y)) 
    # Convert to Warp arrays, float32 works better on gpus!
    q_np = np.array(q, dtype=np.float32)
    q_wp = wp.array(q_np, dtype=wp.vec2, device="cuda")

    # Generate connectors
    connectors = []
    for i in range(dim):
        for j in range(dim):
            idx = i * dim + j
            # connect right
            if j < n_seg: 
                connectors.append((idx, idx + 1))
            # connect top
            if i < n_seg:  
                connectors.append((idx, idx + dim))
            # connect top-right
            if i < n_seg and j < n_seg:  
                connectors.append((idx, idx + dim + 1))
            # Connect top left (or down right)
            if i < n_seg and j > 0: 
                connectors.append((idx, idx + dim - 1))
    connectors_np = np.array(connectors, dtype=np.int32)  # connectors is a 2d array where each row is a pair of connected points
    connectors_wp = wp.array(connectors_np, dtype=wp.vec2i, device="cuda")

    # compute rest lengths
    rest_lens_np = np.linalg.norm(q_np[connectors_np[:, 0]] - q_np[connectors_np[:, 1]], axis=1).astype(np.float32)  # shape (num_edges,)
    rest_lens_wp = wp.array(rest_lens_np, dtype=wp.float32, device="cuda")

    return q_wp, q_np, connectors_wp, connectors_np, rest_lens_wp, rest_lens_np



### Calculate the Value, Gradient, and Hessian of Inertia for each point ###
@wp.kernel
def compute_val_inertia(q: wp.array(dtype=wp.vec2), 
                q_next: wp.array(dtype=wp.vec2),  
                val_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    diff = q[tid] - q_next[tid]
    wp.atomic_add(val_out, 0, 0.5*m[tid]*wp.dot(diff,diff))

@wp.kernel
def compute_grad_inertia(q: wp.array(dtype=wp.vec2), 
        q_next: wp.array(dtype=wp.vec2), 
        grad_out: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    grad_out[tid] = m[tid]*(q[tid] - q_next[tid])

def compute_hess_inertia(q, q_next):
    n = len(q)
    I = np.arange(n, dtype=np.int32)
    J = np.arange(n, dtype=np.int32)
    V = np.array(m).astype(np.float32)
    return [I, J, V]


### Calculate the value, gradient and hessian of the potential energy for each edge (connector) ###
@wp.kernel
def compute_val_potential(q: wp.array(dtype=wp.vec2),
                      connectors: wp.array(dtype=wp.vec2i),
                      rest_lengths: wp.array(dtype=wp.float32),
                      val_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    i1 = connectors[tid][0]
    i2 = connectors[tid][1]

    l2 = (rest_lengths[tid])^2
    diff = q[i1] - q[i2]
    dist2 = wp.dot(diff, diff)
    energy = 0.5 * l2 * k_spring * (dist2/l2 - 1)

    wp.atomic_add(val_out, 0, energy)

@wp.kernel
def compute_grad_potential(q: wp.array(dtype=wp.vec2),
                       connectors: wp.array(dtype=wp.vec2i),
                       rest_lengths: wp.array(dtype=wp.float32),
                       grad_out: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    i1 = connectors[tid][0]
    i2 = connectors[tid][1]

    rest_len = rest_lengths[tid]

    diff = q[i1] - q[i2]
    dist = wp.sqrt(wp.dot(diff, diff)) + 1e-8  # avoid div by zero

    force = k_spring * (dist - rest_len) / dist * diff

    wp.atomic_add(grad_out, i1, force)
    wp.atomic_add(grad_out, i2, -force)

def compute_hess_potential(q, connectors, rest_lengths):
    IJV = [[0] * (len(connectors) * 16), [0] * (len(connectors) * 16), np.array([0.0] * (len(connectors) * 16))]
    for i in range(0, len(connectors)):
        diff = q[connectors[i][0]] - q[connectors[i][1]]
        H_diff = 2 * k_spring / rest_lengths[i] * (2 * np.outer(diff, diff) + (diff.dot(diff) - rest_lengths[i]) * np.identity(2))
        H_local = make_SPD(np.block([[H_diff, -H_diff], [-H_diff, H_diff]]))
        # add to global matrix
        for nI in range(0, 2):
            for nJ in range(0, 2):
                indStart = i * 16 + (nI * 2 + nJ) * 4
                for r in range(0, 2):
                    for c in range(0, 2):
                        IJV[0][indStart + r * 2 + c] = connectors[i][nI] * 2 + r
                        IJV[1][indStart + r * 2 + c] = connectors[i][nJ] * 2 + c
                        IJV[2][indStart + r * 2 + c] = H_local[nI * 2 + r, nJ * 2 + c]
    return IJV

def make_SPD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))



### time integrator ###
def IP_val(q: wp.array(dtype=wp.vec2),
           connectors: wp.array(dtype=wp.int32),
           q_next: wp.array(dtype=wp.vec2),
           rest_lengths: wp.array(dtype=float)):
    
    n = q.shape[0]
    num_springs = rest_lengths.shape[0]

    # Allocate result scalars
    val_inertia = wp.zeros(1, dtype=float, device="cuda")
    val_potential = wp.zeros(1, dtype=float, device="cuda")

    wp.launch(kernel=compute_val_inertia, dim=n, inputs=[q, q_next, val_inertia])
    wp.launch(kernel=compute_val_potential, dim=num_springs, inputs=[q, connectors, rest_lengths, val_potential])

    # Combine the two values (IP energy = inertia + h² * potential)
    val = wp.zeros(1, dtype=float, device="cuda")

    @wp.kernel
    def combine(val_inertia: wp.array(dtype=float), val_potential: wp.array(dtype=float), h: float, out: wp.array(dtype=float)):
        out[0] = val_inertia[0] + h * h * val_potential[0]

    wp.launch(kernel=combine, dim=1, inputs=[val_inertia, val_potential, h, val])
    #print("val is: ", val)
    return val


def IP_grad(q: wp.array(dtype=wp.vec2),
            connectors: wp.array(dtype=wp.int32),
            q_next: wp.array(dtype=wp.vec2),
            rest_lengths: wp.array(dtype=float)):

    n = q.shape[0]
    num_springs = rest_lengths.shape[0]

    grad_inertia = wp.zeros(n, dtype=wp.vec2, device="cuda")
    grad_potential = wp.zeros(n, dtype=wp.vec2, device="cuda")
    grad_total = wp.zeros(n, dtype=wp.vec2, device="cuda")

    wp.launch(kernel=compute_grad_inertia, dim=n, inputs=[q, q_next, grad_inertia])
    wp.launch(kernel=compute_grad_potential, dim=num_springs, inputs=[q, connectors, rest_lengths, grad_potential])

    # Scale grad_potential by h^2 and add to grad_inertia
    @wp.kernel
    def combine_grad(grad_inertia: wp.array(dtype=wp.vec2),
                     grad_potential: wp.array(dtype=wp.vec2),
                     h: float,
                     out: wp.array(dtype=wp.vec2)):
        tid = wp.tid()
        out[tid] = grad_inertia[tid] + (h * h) * grad_potential[tid]

    wp.launch(kernel=combine_grad, dim=n, inputs=[grad_inertia, grad_potential, h, grad_total])

    #print("grad is: ", grad_total)
    return grad_total



def IP_hess(q, connectors, q_next, rest_lengths):
    IJV_In = compute_hess_inertia(q, q_next)
    IJV_MS = compute_hess_potential(q, connectors, rest_lengths)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV = np.append(IJV_In, IJV_MS, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(q) * 2, len(q) * 2)).tocsr()
    return H



def search_dir(q_wp, q_np, connectors_wp, connectors_np, q_next_wp, q_next_np, rest_lengths_wp, rest_lengths_np):
    projected_hess = IP_hess(q_np, connectors_np, q_next_np, rest_lengths_np)

    grad_wp = IP_grad(q_wp, connectors_wp, q_next_wp, rest_lengths_wp)  # returns wp array
    grad_np = grad_wp.numpy()  # copy back to CPU as NumPy array

    reshaped_grad = grad_np.reshape(len(q_np) * 2, 1)
    direction = spsolve(projected_hess, -reshaped_grad)

    return direction.reshape(len(q_np), 2)




def step_forward(q_wp, q_np, connectors_wp, connectors_np, v, rest_lengths_wp, rest_lengths_np, tol):
    q_next_np = q_np + v * h     # implicit Euler predictive position
    q_next_wp = wp.array(q_next_np, dtype=wp.vec2, device="cuda")

    q_n = copy.deepcopy(q_np)

    # Newton loop
    iter = 0
    max_iter = 10

    E_last_wp = IP_val(q_wp, connectors_wp, q_next_wp, rest_lengths_wp)
    E_last_np = E_last_wp.numpy()[0]
    p = search_dir(q_wp, q_np, connectors_wp, connectors_np, q_next_wp, q_next_np, rest_lengths_wp, rest_lengths_np)
   
    while LA.norm(p, np.inf) / h > tol and iter < max_iter:
    #while LA.norm(p, np.inf) / h > tol:
        print('residual =', LA.norm(p, inf) / h)
        print('Iteration', iter, ':')
        print("p is: ", p)
        
        alpha = 1
        while True: 
                q_trial_np = q_np + alpha * p
                q_trial_wp = wp.array(q_trial_np, dtype=wp.vec2, device="cuda")

                E_trial_wp = IP_val(q_trial_wp, connectors_wp, q_next_wp, rest_lengths_wp)
                E_trial_np = E_trial_wp.numpy()[0]

                if E_trial_np <= E_last_np:
                    break

                alpha /= 2
                print("E_trial is: ", E_trial_np )
                print("alpha is: ", alpha)

        print("we have officially exited the SECOND loop!")
        print("we are updating q_np by ", alpha*p)
        q_np += alpha * p
        q_wp = wp.array(q_np, dtype=wp.vec2, device="cuda")  # Sync with updated q_np

        E_last_wp = IP_val(q_wp, connectors_wp, q_next_wp, rest_lengths_wp)
        E_last_np = E_last_wp.numpy()[0]
        p = search_dir(q_wp, q_np, connectors_wp, connectors_np, q_next_wp, q_next_np, rest_lengths_wp, rest_lengths_np)
        iter += 1


    print("we have officially exited the FIRST loop!")
    v = (q_np - q_n) / h   # implicit Euler velocity update
    return [q_np, v]




# initialize simulation
[q_wp, q_np, connectors_wp, connectors_np, rest_lens_wp, rest_lens_np] = generate_warp_mesh(side_len, n_seg)  # node positions and edge node indices

v = np.array([[0.0, 0.0]] * len(q_wp))             # velocity

# apply initial stretch horizontally
for i in range(0, len(q_np)):
    q_np[i][0] *= initial_stretch

# simulation with visualization
resolution = np.array([900, 900])
offset = resolution / 2
scale = 200
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]



def write_to_file(frameNum, q_np, n_seg):
    # Check if 'output' directory exists; if not, create it
    if not os.path.exists('output'):
        os.makedirs('output')

    # create obj file
    filename = f"output/{frameNum}.obj"
    with open(filename, 'w') as f:
        # write vertex coordinates
        for row in q_np:
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n") 
        # write vertex indices for each triangle
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                # note: each cell is exported as 2 triangles for rendering
                f.write(f"f {i * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j+1 + 1}\n")
                f.write(f"f {i * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j+1 + 1} {i * (n_seg+1) + j+1 + 1}\n")



time_step = 0
write_to_file(time_step, q_np, n_seg)
screen = pygame.display.set_mode(resolution)
running = True
while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    print('### Time step', time_step, '###')

    # fill the background and draw the square
    screen.fill((255, 255, 255))
    #FIX this should be connectors np
    for cI in connectors_np:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(q_np[cI[0]]), screen_projection(q_np[cI[1]]))
    for qI in q_np:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(qI), 0.1 * side_len / n_seg * scale)

    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    [q_np, v] = step_forward(q_wp, q_np, connectors_wp, connectors_np, v, rest_lens_wp, rest_lens_np, 1e-2)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    write_to_file(time_step, q_np, n_seg)

pygame.quit()
