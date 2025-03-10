from mpi4py import MPI
import numpy as np
import sys
import os
from math import ceil, sqrt
from PIL import Image
import pickle

import vtkmodules.all as vtk

import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
## mpiexec -n 32 python script.py 32 Isabel_1000x1000x200_float32.raw 2 2 8 1 opacity_TF.txt color_TF.txt


def print_flush(message):
    """Print message and flush the output."""
    print(message)
    sys.stdout.flush()


def read_file_dimensions(filename):
    """Extract dimensions from filename."""
    base = os.path.basename(filename)
    parts = base.split('_')
    dims_part = parts[-2]
    dimensions = tuple(map(int, dims_part.split('x')))
    print_flush(f"Extracted dimensions from filename: {dimensions}")
    return dimensions


def read_raw_file(filename, dims):
    """Read raw file into numpy array with specified dimensions and float32 type."""
    print_flush("Reading data from file...")
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    x_size, y_size, z_size = dims
    data = data.reshape((z_size, x_size, y_size))
    data = data[:, ::-1, ::-1]
    print_flush("Data read successfully.")
    return data





def partition_data(data,  x_num, y_num, z_num):
    """Partition data based on the specified partition type."""
    z_size, x_size, y_size = data.shape
    sub_data = []

    # Decompose along both x and y directions
    x_chunk_size = ceil(x_size / x_num)
    y_chunk_size = ceil(y_size / y_num)
    z_chunk_size = ceil(z_size / z_num)
    for k in range(z_num):
        for i in range(x_num):
            for j in range(y_num):
                    x_start = i * x_chunk_size
                    x_end = min((i + 1) * x_chunk_size, x_size)
                    y_start = j * y_chunk_size
                    y_end = min((j + 1) * y_chunk_size, y_size)
                    z_start = k * z_chunk_size
                    z_end = min((k + 1) * z_chunk_size, z_size)

                    sub_data.append(data[z_start:z_end, x_start:x_end, y_start:y_end])
    #print_flush(sub_data)
    return sub_data




def ray_casting(data_chunk, step_size, opacity_tf , triplets_list):
    z_size, x_size, y_size = data_chunk.shape
    output_image = np.zeros((x_size, y_size, 4), dtype=np.float32)  # 3 channels for RGB
    r_tf = vtk.vtkPiecewiseFunction()
    g_tf = vtk.vtkPiecewiseFunction()
    b_tf = vtk.vtkPiecewiseFunction()

    def calc_transfer_functions():
        """Transfer function to map input value to color"""
        for val, r, g, b in triplets_list:
            scalar_value = float(val)
            r_float = float(r)
            g_float = float(g)
            b_float = float(b)
            r_tf.AddPoint(scalar_value, r_float)
            g_tf.AddPoint(scalar_value, g_float)
            b_tf.AddPoint(scalar_value, b_float)

    calc_transfer_functions()

    def opacity_ttf(x, opacitytf):
        return opacitytf.GetValue(x)

    
    for x in range(x_size):
        for y in range(y_size):
            color_accum = np.zeros(3)
            opacity_accum = 0

            z = 0
            while z < z_size:
                if step_size == 1:
                    value = data_chunk[int(z), x, y]
                else:
                    z1 = int(z)
                    z2 = min(z1 + 1, z_size - 1)
                    alpha = z - z1
                    value = (1 - alpha) * data_chunk[z1, x, y] + alpha * data_chunk[z2, x, y]

                # Apply transfer functions
                r = r_tf.GetValue(value)
                g = g_tf.GetValue(value)
                b = b_tf.GetValue(value)
                color = r, g, b
                opacity = opacity_ttf(value, opacity_tf)  # Assuming opacity is the first channel

                # Composite color and opacity
                color = np.array(color)
                color_accum += (1 - opacity_accum) * color * opacity
                opacity_accum += opacity * (1 - opacity_accum)

                if opacity_accum>=1:
                    break
                

                z += step_size

            output_image[x, y, 0] = color_accum[0] if color_accum[0] <= 1 else 1
            output_image[x, y, 1] = color_accum[1] if color_accum[1] <= 1 else 1
            output_image[x, y, 2] = color_accum[2] if color_accum[2] <= 1 else 1
            output_image[x, y, 3] = opacity_accum if opacity_accum <= 1 else 1

    return output_image


def process_data(sub_data_chunk, step_size, opacity_tf, triplets_list):
    """Process the chunk of data for ray casting with transfer functions."""
    return ray_casting(sub_data_chunk, step_size, opacity_tf, triplets_list)


def read_file_to_pairs(filename):
    pairs_list = []
    with open(filename, 'r') as file:
        values = file.read().replace(',', '').split()

    for i in range(0, len(values), 2):
        pair = (values[i], values[i + 1])
        pairs_list.append(pair)

    return pairs_list

def process_colour_tf_file(colour_tf_file):
    triplets_list = []
    with open('color_TF.txt', 'r') as file:
        values = file.read().replace(',', '').split()

    for i in range(0, len(values), 4):
        triplet = (values[i], values[i + 1], values[i + 2], values[i + 3])
        triplets_list.append(triplet)
    return triplets_list

def tree_based_gather(data, root ,ch):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    step = 1
    gathered_data = np.array(data)  # Start with the data of the current process
    rank = rank
    # Tree-based gather in log2(size) steps
    while step < size:
        partner = (rank ^ (step*ch))  # XOR to find the partner node
        
        # Ensure only processes within range communicate
        if partner < size:
            if rank < partner:
                # Receive data from partner and combine it
                # print("received" , rank , partner)
                received_data = comm.recv(source=partner, tag=step)
                B = np.ones_like(data[:, :, 3])
                mask0 = gathered_data[:, :, 0] < 1
                mask1 = gathered_data[:, :, 1] < 1
                mask2 = gathered_data[:, :, 2] < 1
                mask3 = gathered_data[:, :, 3] < 1
                gathered_data[:, :, 0][mask0] += np.multiply((B-gathered_data[:, :, 3]) , received_data[:, :, 0])[mask0]
                gathered_data[:, :, 1][mask1] += np.multiply((B-gathered_data[:, :, 3]) , received_data[:, :, 1])[mask1]
                gathered_data[:, :, 2][mask2] += np.multiply((B-gathered_data[:, :, 3]) , received_data[:, :, 2])[mask2]
                gathered_data[:, :, 3][mask3] += np.multiply((B-gathered_data[:, :, 3]) , received_data[:, :, 3])[mask3]
                gathered_data = np.clip(gathered_data, 0, 1)
            else:
                # Send current data to partner
                #print("sending" , rank , partner)
                a=time.time()
                comm.send(gathered_data, dest=partner, tag=step)
                #print(time.time()-a, rank , partner)
                
                break  # Once sent, this process stops participating

        step <<= 1  # Double the step size to reach higher levels
        
    # Only root will have the complete gathered data
    return gathered_data if rank == root else None

def tree_based_gather_after(data, root ,ch):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    step = 1
    
    gathered_data = [data]  # Start with the data of the current process

    # Tree-based gather in log2(size) steps
    while step < ch:
        partner = rank ^ step  # XOR to find the partner node
        
        # Ensure only processes within range communicate
        if partner < size:
            if rank < partner:
                # Receive data from partner and combine it
                #print("received" , rank , partner)
                a=time.time()
                received_data = comm.recv(source=partner, tag=step)
                #print(time.time()-a, "111")
                gathered_data.extend(received_data)
                
                
            else:
                # Send current data to partner
                comm.send(gathered_data, dest=partner, tag=step)
                break  # Once sent, this process stops participating

        step <<= 1  # Double the step size to reach higher levels

    # Only root will have the complete gathered data
    return gathered_data if rank == root else None

def main():
    rank = comm.Get_rank()
    if rank == 0:
        print_flush("starting")
        start_time = time.time()
        #print_flush(sys.argv)
        if len(sys.argv) != 9:
            print_flush("Usage: python script.py <num_proc> <filename> <px> <py> <pz> <step_size> <opacity_tf_file> <colour_tf_file>")
            sys.exit(1)

        num_proc = int(sys.argv[1])
        filename = sys.argv[2]
        num_proc_x= int(sys.argv[3])
        num_proc_y= int(sys.argv[4])
        num_proc_z= int(sys.argv[5])
        step_size = float(sys.argv[6])
        opacity_tf_file = sys.argv[7]
        colour_tf_file = sys.argv[8]
        dims = read_file_dimensions(filename)
        data = read_raw_file(filename, dims)
        print_flush("data read")
        ch = num_proc_x * num_proc_y
        # Partition the data
        sub_data_chunks = partition_data(data,  num_proc_x, num_proc_y, num_proc_z)
        chunk_shapes = [chunk.shape for chunk in sub_data_chunks]  # Get shapes for each chunk
        n = num_proc
        # Prepare counts and displacements for scatterv
        counts = [chunk.size for chunk in sub_data_chunks]
        displacements = [sum(counts[:i]) for i in range(size)]

        # Flatten all data
        flattened_data = np.concatenate([chunk.flatten() for chunk in sub_data_chunks])
        triplets_list = process_colour_tf_file(colour_tf_file)
        opacity_tf_pairs = read_file_to_pairs(opacity_tf_file)
        
    else:
        # On non-root ranks, initialize variables as None to receive the broadcast/scatter
        sub_data_chunks = None
        chunk_shapes = None
        step_size = None
        flattened_data = None
        counts = None
        displacements = None
        triplets_list = None
        opacity_tf_pairs = None
        ch = None
        n = None
    
    n = comm.bcast(n, root=0)
    step_size = comm.bcast(step_size, root=0)
    # Broadcast chunk shapes to all ranks
    ch = comm.bcast(ch, root=0)
    chunk_shapes = comm.bcast(chunk_shapes, root=0)
    
    result = np.empty(chunk_shapes[rank%ch])
    # Broadcast transfer functions to all ranks
    transfer_functions = comm.bcast(None, root=0)
    triplets_list = comm.bcast(triplets_list, root=0)
    opacity_tf_pairs = comm.bcast(opacity_tf_pairs, root=0)
    # Broadcast counts and displacements for scatterv
    counts = comm.bcast(counts, root=0)
    displacements = comm.bcast(displacements, root=0)

    # Scatter the data
    recv_count = counts[rank]
    recv_buffer = np.empty(recv_count, dtype=np.float32)

    comm.Scatterv([flattened_data, counts, displacements, MPI.FLOAT], recv_buffer, root=0)

    # Reshape the received data on each rank using the appropriate shape for the current rank
    sub_data_chunk = recv_buffer.reshape(chunk_shapes[rank])

    # Process data (ray casting)
    opacity_tf = vtk.vtkPiecewiseFunction()
    for pair in opacity_tf_pairs:
        opacity_tf.AddPoint(float(pair[0]), float(pair[1]))
    
    local_image  = process_data(sub_data_chunk, step_size, opacity_tf, triplets_list)
    #print(local_image.shape, "l")
    
     # Gather all local images to the root processor
    b=time.time()
    '''
    gathered_images = tree_based_gather(local_image , rank % ch , ch)
    #print(time.time() - a , rank)
    #print(gathered_images.shape)
    c=time.time()
    a=time.time()
    comm.Barrier()
    if rank < ch:
        #print(gathered_images.shape)
        gathered_images = gathered_images[: ,: ,:3]
        #img = Image.fromarray((gathered_images * 255 / gathered_images.max()).astype('uint8'))
        #img.show()
        #img.save(f'{rank}.png')
        #gathered_images += np.full_like(gathered_images, 0.1)
        gathered_images = tree_based_gather_after(gathered_images ,0 ,ch)
        
        
    
    '''
    
    gathered_images = comm.gather(local_image, root=0)
    
    #print(c-b , time.time() - a , rank)
    #print("gather done" , rank)
    if rank == 0:
        print_flush("after gather")
        gathered_images = np.array(gathered_images)
        print(gathered_images.shape)
        # Combine all sub-images into the final image
        print("Stitching images.")
        z_size, x_size, y_size = data.shape
        final_image = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.float32)
        opac = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
        z_size, x_size, y_size = data.shape

        # Decompose along both x and y directions
        x_chunk_size = ceil(x_size / num_proc_x)
        y_chunk_size = ceil(y_size / num_proc_y)
        z_chunk_size = ceil(z_size / num_proc_z)
        h=0
        for k in range(num_proc_z):
            for i in range(num_proc_x):
                for j in range(num_proc_y):
                        x_start = i * x_chunk_size
                        x_end = min((i + 1) * x_chunk_size, x_size)
                        y_start = j * y_chunk_size
                        y_end = min((j + 1) * y_chunk_size, y_size)
                        
                        B = np.ones_like(opac[x_start:x_end, y_start:y_end])
                        
                        

                        final_image[x_start:x_end, y_start:y_end ,0] += np.multiply((B-opac[x_start:x_end, y_start:y_end ]) , gathered_images[h, :, :, 0])
                        final_image[x_start:x_end, y_start:y_end ,1] += np.multiply((B-opac[x_start:x_end, y_start:y_end ]) , gathered_images[h, :, :, 1])
                        final_image[x_start:x_end, y_start:y_end ,2 ] += np.multiply((B-opac[x_start:x_end, y_start:y_end ]) , gathered_images[h, :, :, 2])
                        opac [x_start:x_end, y_start:y_end ] += np.multiply((B-opac[x_start:x_end, y_start:y_end ]) , gathered_images[h, :, :, 3])
                        final_image = np.clip(final_image, 0, 1)
                        
                        
                        
                        h+=1


        
        img = Image.fromarray((final_image * 255 / final_image.max()).astype('uint8'))
        
        #img.show()
        img.save('rendered_image1.png')

        print_flush("Final image displayed.")
    
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time taken: ", total_time)


if __name__ == "__main__":
    main()