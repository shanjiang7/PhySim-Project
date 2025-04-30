import warp as wp
import numpy as np

# initialize warp
wp.init()

#THIS IS FOR HAVING POINTS AND CONNECTORS AS 1D VECTORS
#Box will be centered at (0,0) for this script
def generate_warp_mesh():
    side_length = 2
    n_seg = 2
    seg_length = side_length / n_seg

    dim = n_seg + 1

    # Generate grid points, pos will be a 1D vector with x and y positions (basically q)
    # pos = q = [x1, y1, x2, y2, x3, y3, ..., xn, yn]
    pos = []
    for i in range(dim):
        for j in range(dim):
            x = -side_length / 2 + i * seg_length
            y = -side_length / 2 + j * seg_length
            pos.append((x, y, 0.0))  # Warp prefers 3D vectors (vec3)

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
    pos_np = np.array(pos, dtype=np.float32)
    connectors_np = np.array(connectors, dtype=np.int32).flatten()  # flatten to 1D array

    # copy the arrays to cuda gpu using warp!
    pos_wp = wp.array(pos_np, dtype=wp.vec3, device="cuda")
    connectors_wp = wp.array(connectors_np, dtype=wp.int32, device="cuda")

    print(f"Created {len(pos_np)} points and {len(connectors)} connectors")

    return pos_wp, connectors_wp, len(connectors)

# Call the function 
pos_gpu, conn_gpu, num_connectors = generate_warp_mesh()
