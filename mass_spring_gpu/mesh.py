import warp as wp
import numpy as np

# initialize warp
wp.init()

# I wanted to have q and a long 1D array that has 2N entries, 
# but I think Warp will work better using the wp.vec2 option 
#Box will be centered at (0,0) for this script

# Spring stiffness constant
k_spring = 2.0



def generate_warp_mesh():
    side_length = 2
    n_seg = 4
    seg_length = side_length / n_seg
    dim = n_seg + 1

    # Generate grid points, pos will be a 1D vector with x and y positions (basically q)
    # pos = q = [x1, y1, x2, y2, x3, y3, ..., xn, yn]
    q = []
    for i in range(dim):
        for j in range(dim):
            x = -side_length / 2 + i * seg_length
            y = -side_length / 2 + j * seg_length
            q.append((x, y))  # use 2D warp vectors (since this is a 2D simulation)

    # Convert to Warp arrays, float32 works better on gpus!
    q_np = np.array(q, dtype=np.float32)

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

    # compute rest lengths
    rest_lengths = np.linalg.norm(q_np[connectors_np[:, 0]] - q_np[connectors_np[:, 1]], axis=1).astype(np.float32)  # shape (num_edges,)


    # copy the arrays to cuda gpu using warp!
    q_wp = wp.array(q_np, dtype=wp.vec2, device="cuda")
    connectors_wp = wp.array(connectors_np, dtype=wp.vec2i, device="cuda")
    rest_lengths_wp = wp.array(rest_lengths, dtype=wp.float32, device="cuda")


    print(f"Created {len(q_np)} points and {len(connectors)} connectors")

    return q_wp, connectors_wp, rest_lengths_wp

# q and q_next are 1D arrays of size 2N







# USe warp!
# Intertia: value, gradietn and hessian
wp.init()
@wp.kernel
def compute_val_inertia(q: wp.array(dtype=wp.vec2), 
                q_next: wp.array(dtype=wp.vec2), 
                m: wp.array(dtype=wp.float32), 
                val_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    diff = q[tid] - q_next[tid]
    wp.atomic_add(val_out, 0, 0.5*m[tid]*wp.dot(diff,diff))
    #sum = 0.0
    #for i in range(len(q)):
        #E_I = 0.5 * m[i]* (q[i]-q_next[i])^2
        #sum += E_I
    #return sum

@wp.kernel
def compute_grad_inertia(q: wp.array(dtype=wp.vec2), 
        q_next: wp.array(dtype=wp.vec2), 
        m: wp.array(dtype=wp.float32), 
        grad_out: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    grad_out[tid] = m[tid]*(q[tid] - q_next[tid])


def compute_hess_inertia(q, q_next, m):
    n2 = len(q)
    I = np.arange(n2, dtype=np.int32)
    J = np.arange(n2, dtype=np.int32)
    V = m.astype(np.float32)
    return [I, J, V]






@wp.kernel
def compute_val_potential(q: wp.array(dtype=wp.vec2),
                      connectors: wp.array(dtype=wp.vec2i),
                      rest_lengths: wp.array(dtype=wp.float32),
                      val_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    i1 = connectors[tid][0]
    i2 = connectors[tid][1]

    rest_len = rest_lengths[tid]

    diff = q[i1] - q[i2]
    dist2 = wp.dot(diff, diff)
    dist = wp.sqrt(dist2)

    energy = 0.5 * k_spring * ((dist - rest_len) * (dist - rest_len))

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




def make_PSD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))


def compute_hess_potential(q, connectors, rest_lengths):
    IJV = [[0] * (len(connectors) * 16), [0] * (len(connectors) * 16), np.array([0.0] * (len(connectors) * 16))]
    for i in range(0, len(connectors)):
        diff = q[connectors[i][0]] - q[connectors[i][1]]
        H_diff = 2 * k_spring / rest_lengths[i] * (2 * np.outer(diff, diff) + (diff.dot(diff) - rest_lengths[i]) * np.identity(2))
        H_local = make_PSD(np.block([[H_diff, -H_diff], [-H_diff, H_diff]]))
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


















# running the warp kernels

def run_warp_ops(q_np, q_next_np, m_np, connectors_wp, rest_lengths_wp):
    n = q_np.shape[0]
    num_springs = rest_lengths_wp.shape[0]

    # Convert inputs to Warp arrays
    q_wp = wp.from_numpy(q_np.astype(np.float32), dtype=wp.vec2, device="cuda")
    q_next_wp = wp.from_numpy(q_next_np.astype(np.float32), dtype=wp.vec2, device="cuda")
    m_wp = wp.from_numpy(m_np.astype(np.float32), device="cuda")

    # connectors_wp and rest_lengths_wp are already Warp arrays, passed in
    grad_wp = wp.zeros(n, dtype=wp.vec2, device="cuda")
    val_wp = wp.zeros(1, dtype=wp.float32, device="cuda")

    # Launch kernels
    wp.launch(kernel=compute_val_inertia, dim=n, inputs=[q_wp, q_next_wp, m_wp, val_wp])
    wp.launch(kernel=compute_grad_inertia, dim=n, inputs=[q_wp, q_next_wp, m_wp, grad_wp])
    wp.launch(kernel=compute_val_potential, dim=num_springs, inputs=[q_wp, connectors_wp, rest_lengths_wp, val_wp])
    wp.launch(kernel=compute_grad_potential, dim=num_springs, inputs=[q_wp, connectors_wp, rest_lengths_wp, grad_wp])

    # Copy results back to CPU
    val_host = val_wp.numpy()[0]
    grad_host = grad_wp.numpy()
    hess = compute_hess_inertia(q_np, q_next_np, m_np)

    return val_host, grad_host, hess




# exmaple usage:
N = 4
q = np.random.rand(N, 2).astype(np.float32)
q_next = np.random.rand(N, 2).astype(np.float32)
m = np.ones(N, dtype=np.float32)
q_gpu, conn_gpu, rest_lens_gpu = generate_warp_mesh()

print(conn_gpu)

val, grad, hess = run_warp_ops(q, q_next, m, conn_gpu, rest_lens_gpu)
print("Value:", val)
print("Grad:", grad)
print("Hess IJV:", hess) 

# Call the function 









# for visualization, using matplotlib, this is a bit rudamentary, but will work for now 

import matplotlib.pyplot as plt

def visualize_mesh(q_np, connectors_np, title="Mesh Visualization"):
    """
    q_np: (N, 2) NumPy array of 2D positions
    connectors_np: (E, 2) NumPy array of index pairs
    """
    plt.figure(figsize=(6, 6))

    # Draw connectors
    for idx1, idx2 in connectors_np:
        x = [q_np[idx1, 0], q_np[idx2, 0]]
        y = [q_np[idx1, 1], q_np[idx2, 1]]
        plt.plot(x, y, 'k-', linewidth=1)

    # Draw particles
    plt.scatter(q_np[:, 0], q_np[:, 1], color='red', zorder=5)

    plt.axis("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()






q_gpu, conn_gpu, rest_gpu = generate_warp_mesh()

# Convert q and connectors back to CPU for plotting
q_np = q_gpu.numpy()
conn_np = conn_gpu.numpy()

visualize_mesh(q_np, conn_np)
