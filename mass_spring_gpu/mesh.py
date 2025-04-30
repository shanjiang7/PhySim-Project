import warp as wp
import numpy as np

# initialize warp
wp.init()

# I wanted to have q and a long 1D array that has 2N entries, 
# but I think Warp will work better using the wp.vec2 option 
#Box will be centered at (0,0) for this script
def generate_warp_mesh():
    side_length = 2
    n_seg = 3
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

    # Convert to Warp arrays, float32 works better on gpus!
    q_np = np.array(q, dtype=np.float32)
    connectors_np = np.array(connectors, dtype=np.int32).flatten()  # flatten to 1D array

    # copy the arrays to cuda gpu using warp!
    q_wp = wp.array(q_np, dtype=wp.vec2, device="cuda")
    connectors_wp = wp.array(connectors_np, dtype=wp.int32, device="cuda")

    print(f"Created {len(q_np)} points and {len(connectors)} connectors")

    return q_wp, connectors_wp, len(connectors)

# q and q_next are 1D arrays of size 2N
# USe warp!
wp.init()
@wp.kernel
def compute_val(q: wp.array(dtype=wp.vec2), 
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
def compute_grad(q: wp.array(dtype=wp.vec2), 
        q_next: wp.array(dtype=wp.vec2), 
        m: wp.array(dtype=wp.float32), 
        grad_out: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    grad_out[tid] = m[tid]*(q[tid] - q_next[tid])


def compute_hess(q, q_next, m):
    n2 = len(q)
    I = np.arange(n2, dtype=np.int32)
    J = np.arange(n2, dtype=np.int32)
    V = m.astype(np.float32)
    return [I, J, V]



# running the warp kernels

def run_warp_ops(q_np, q_next_np, m_np):
    n = q_np.shape[0]  # number of particles

    # np is for numpy (cpu) wp is for warp (gpu)
    q_wp = wp.from_numpy(q_np.astype(np.float32), dtype=wp.vec2, device="cuda")
    q_next_wp = wp.from_numpy(q_next_np.astype(np.float32), dtype=wp.vec2, device="cuda")
    m_wp = wp.from_numpy(m_np.astype(np.float32), device="cuda")
    grad_wp = wp.zeros(n, dtype=wp.vec2, device="cuda")
    val_wp = wp.zeros(1, dtype=wp.float32, device="cuda")

    # Launch kernels
    wp.launch(kernel=compute_val, dim=n, inputs=[q_wp, q_next_wp, m_wp, val_wp])
    wp.launch(kernel=compute_grad, dim=n, inputs=[q_wp, q_next_wp, m_wp, grad_wp])

    # copy back to cpu numpy
    val_host = val_wp.numpy()[0]
    grad_host = grad_wp.numpy()

    hess = compute_hess(q_np, q_next_np, m_np)

    return val_host, grad_host, hess

# exmaple usage:
N = 4
q = np.random.rand(N, 2).astype(np.float32)
q_next = np.random.rand(N, 2).astype(np.float32)
m = np.ones(N, dtype=np.float32)

val, grad, hess = run_warp_ops(q, q_next, m)
print("Value:", val)
print("Grad:", grad)
print("Hess IJV:", hess) 

# Call the function 
q_gpu, conn_gpu, num_connectors = generate_warp_mesh()








# for visualization, using matplotlib, this is a bit rudamentary, but will work for now 

import matplotlib.pyplot as plt

def visualize_mesh(q_np, connectors_np, title="Mesh Visualization"):
    """
    q_np: (N, 2) NumPy array of 2D positions
    connectors_np: flat NumPy array of shape (2*num_connectors,) with index pairs
    """
    plt.figure(figsize=(6, 6))
    q = q_np.reshape(-1, 2)

    # Draw connectors
    for i in range(0, len(connectors_np), 2):
        idx1 = connectors_np[i]
        idx2 = connectors_np[i + 1]
        x = [q[idx1, 0], q[idx2, 0]]
        y = [q[idx1, 1], q[idx2, 1]]
        plt.plot(x, y, 'k-', linewidth=1)

    # Draw particles
    plt.scatter(q[:, 0], q[:, 1], color='red', zorder=5)

    plt.axis("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

q_gpu, conn_gpu, num_connectors = generate_warp_mesh()

# Convert q and connectors back to CPU for plotting
q_np = q_gpu.numpy()
conn_np = conn_gpu.numpy()

visualize_mesh(q_np, conn_np)
