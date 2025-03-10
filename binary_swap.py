from mpi4py import MPI
import numpy as np
import sys
import os
from math import ceil, sqrt
import numba
from PIL import Image
import traceback
import time
import socket

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
    print(f"Extracted dimensions from filename: {dimensions}")
    return dimensions


def read_raw_file(filename, dims):
    """Read raw file into numpy array with specified dimensions and float32 type."""
    print("Reading data from file...")
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    x_size, y_size, z_size = dims
    data = data.reshape((z_size, x_size, y_size))
    data = data[:, ::-1, ::-1]
    print("Data read successfully.")
    return data




@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def binary_search_transfer_function(value, transfer_function):
    """Perform binary search on the transfer function values."""
    left = 0
    right = len(transfer_function) - 1

    while left <= right:
        mid = (left + right) // 2
        if transfer_function[mid][0] == value:
            return transfer_function[mid][1]
        elif transfer_function[mid][0] < value:
            left = mid + 1
        else:
            right = mid - 1

    # If the value is not found, interpolate between the nearest values
    if left >= len(transfer_function):
        return transfer_function[-1][1]
    elif right < 0:
        return transfer_function[0][1]
    else:
        lower_value, lower_tf = transfer_function[right]
        upper_value, upper_tf = transfer_function[left]
        alpha = (value - lower_value) / (upper_value - lower_value)
        return lower_tf + alpha * (upper_tf - lower_tf)
@numba.jit(nopython=True)
def ray_casting(data_chunk, step_size, opacity_tf , r_tf , g_tf, b_tf):
    z_size, x_size, y_size = data_chunk.shape
    output_image = np.zeros((x_size, y_size, 4), dtype=np.float32)  # 3 channels for RGB

    
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
                r = binary_search_transfer_function(value, r_tf)
                g = binary_search_transfer_function(value, g_tf)
                b = binary_search_transfer_function(value, b_tf)
                color = r, g, b
                opacity = binary_search_transfer_function(value, opacity_tf)  # Assuming opacity is the first channel

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

@numba.jit(nopython=True)
def process_data(sub_data_chunk, step_size, opacity_tf,  r_tf , g_tf, b_tf):
    """Process the chunk of data for ray casting with transfer functions."""
    return ray_casting(sub_data_chunk, step_size, opacity_tf,  r_tf , g_tf, b_tf)


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

def tree_based_gather(data, root ,ch, rank , size):
    
    comm = MPI.COMM_WORLD
    step = 1
    isnotfull = True
    gathered_data = np.array(data)  # Start with the data of the current process
    rank = rank
    full = np.ones_like(data[:, :, 3])
    # Tree-based gather in log2(size) steps
    while step < size:
        partner = (rank ^ (step*ch))  # XOR to find the partner node
        
        # Ensure only processes within range communicate
        if partner < size:
            if rank < partner and isnotfull:
                # Receive data from partner and combine it
                #print("received" , rank , partner)
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
                if np.array_equal(gathered_data , full):
                    print("terminated early")
                    isfull = False
                
            else:
                # Send current data to partner
                #print("sending" , rank , partner)
                
                comm.send(gathered_data, dest=partner, tag=step)
                #print(time.time()-a, rank , partner)
                
                break  # Once sent, this process stops participating

        step <<= 1  # Double the step size to reach higher levels
        
    # Only root will have the complete gathered data
    return gathered_data if rank == root else None

def binary_swap(image_data, width ,ch, size):
    current_width = width
    current_image = image_data.copy()
    start = 0
    # Iteratively reduce and blend portions
    step = 1
    while step < size:
        partner = (rank ^ (step*ch))
        if partner < size:
            # Define which portion to keep based on rank and partner
            if rank < partner:
                # Keep the left half, send the right half
                
                to_send = current_image[:, current_width//2:, :]
                to_keep = current_image[:, :current_width//2, :]
                comm.send(to_send, dest=partner, tag=rank)
                received = comm.recv(source=partner, tag=partner)
            else:
                # Keep the right half, send the left half
                
                to_send = current_image[:, :current_width//2, :]
                to_keep = current_image[:, current_width//2:, :]
                received = comm.recv(source=partner, tag=partner)
                comm.send(to_send, dest=partner, tag=rank)

            # Send and receive portions
            
            
            
            
            
            # Blend received portion with the kept portion
            if rank < partner:
                B = np.ones_like(to_keep[:, :, 3])
                mask0 = to_keep[:, :, 0] < 1
                mask1 = to_keep[:, :, 1] < 1
                mask2 = to_keep[:, :, 2] < 1
                mask3 = to_keep[:, :, 3] < 1
                to_keep[:, :, 0][mask0] += np.multiply((B-to_keep[:, :, 3]) , received[:, :, 0])[mask0]
                to_keep[:, :, 1][mask1] += np.multiply((B-to_keep[:, :, 3]) , received[:, :, 1])[mask1]
                to_keep[:, :, 2][mask2] += np.multiply((B-to_keep[:, :, 3]) , received[:, :, 2])[mask2]
                to_keep[:, :, 3][mask3] += np.multiply((B-to_keep[:, :, 3]) , received[:, :, 3])[mask3]
                to_keep = np.clip(to_keep, 0, 1)
                current_image = to_keep
            else:
                B = np.ones_like(received[:, :, 3])
                mask0 = received[:, :, 0] < 1
                mask1 = received[:, :, 1] < 1
                mask2 = received[:, :, 2] < 1
                mask3 = received[:, :, 3] < 1
                received[:, :, 0][mask0] += np.multiply((B-received[:, :, 3]) , to_keep[:, :, 0])[mask0]
                received[:, :, 1][mask1] += np.multiply((B-received[:, :, 3]) , to_keep[:, :, 1])[mask1]
                received[:, :, 2][mask2] += np.multiply((B-received[:, :, 3]) , to_keep[:, :, 2])[mask2]
                received[:, :, 3][mask3] += np.multiply((B-received[:, :, 3]) , to_keep[:, :, 3])[mask3]
                received = np.clip(received, 0, 1)
                current_image = received

            

            
            current_width //= 2  # Reduce width by half for the next stage

        step *= 2  # Move to the next level in binary swap
    return  np.array(current_image)



def tree_based_gather_after1(data, root ,ch, rank , size):
    comm = MPI.COMM_WORLD
    
    
    step = 1
    gathered_data = [data]  # Start with the data of the current process

    # Tree-based gather in log2(size) steps
    while step <= ch:
        partner = (rank ^ (step*ch))  # XOR to find the partner node
        
        # Ensure only processes within range communicate
        if partner < size:
            if rank < partner:
                # Receive data from partner and combine it
                #print("received" , rank , partner)
                a=time.time()
                #print(gathered_data.shape)
                received_data = comm.recv(source=partner, tag=step)
                #print(received_data.shape)
                #print(time.time()-a, "111")
                gathered_data.extend(received_data)

                
                
            else:
                # Send current data to partner
                comm.send(gathered_data, dest=partner, tag=step)
                break  # Once sent, this process stops participating

        step <<= 1  # Double the step size to reach higher levels

    # Only root will have the complete gathered data
    return gathered_data if rank == root else None


def tree_based_gather_after(data, root ,ch, rank , size):
    comm = MPI.COMM_WORLD
    
    
    step = 1
    gathered_data = [data]  # Start with the data of the current process

    # Tree-based gather in log2(size) steps
    while step < ch:
        partner = (rank ^ (step))  # XOR to find the partner node
        
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
    return np.array(gathered_data) if rank == root else None

def main():
    rank = comm.Get_rank()
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    #print_flush(f"Process {rank} is running on host {hostname} with IP {ip_address}")
    rank = comm.Get_rank()
    if rank == 0:
        print("starting")
        sys.stdout.flush()
        start_time = time.time()
        #print_flush(sys.argv)
        if len(sys.argv) != 9:
            print("Usage: python script.py <num_proc> <filename> <px> <py> <pz> <step_size> <opacity_tf_file> <colour_tf_file>")
            sys.stdout.flush()
            sys.exit(1)

        num_proc = int(sys.argv[1])
        filename = sys.argv[2]
        num_proc_x= int(sys.argv[3])
        num_proc_y= int(sys.argv[4])
        num_proc_z= int(sys.argv[5])
        step_size = float(sys.argv[6])
        opacity_tf_file = sys.argv[7]
        n = num_proc
        colour_tf_file = sys.argv[8]
        dims = read_file_dimensions(filename)
        data = read_raw_file(filename, dims)
        z_size, x_size, y_size = data.shape
        print("data read")
        sys.stdout.flush()
        ch = num_proc_x * num_proc_y
        # Partition the data
        data = partition_data(data,  num_proc_x, num_proc_y, num_proc_z)
        

        sub_data_chunk = data[0]
        for i in range(1, num_proc):
            comm.send(data[i] , dest = i , tag = 0)
        
        sub_data_chunks = sub_data_chunk

        # Flatten all data
        
        triplets_list = process_colour_tf_file(colour_tf_file)
        opacity_tf_pairs = read_file_to_pairs(opacity_tf_file)
        
    else:
        # On non-root ranks, initialize variables as None to receive the broadcast/scatter
        step_size = None
        
        triplets_list = None
        opacity_tf_pairs = None
        ch = None
        n = None
        # print_flush(f"{rank} recieving chunk")
        sub_data_chunks = comm.recv(source = 0 , tag =0)
        # print_flush(f"{rank} receveid chunk")
    
    n = comm.bcast(n, root=0)
    step_size = comm.bcast(step_size, root=0)
    # Broadcast chunk shapes to all ranks
    ch = comm.bcast(ch, root=0)
    
    
    # Broadcast transfer functions to all ranks
    
    triplets_list = comm.bcast(triplets_list, root=0)
    opacity_tf_pairs = comm.bcast(opacity_tf_pairs, root=0)
    # Broadcast counts and displacements for scatterv
    


    # Reshape the received data on each rank using the appropriate shape for the current rank
    

    # Process data (ray casting)
    opacity_tf = []
    for pair in opacity_tf_pairs:
        opacity_tf.append((float(pair[0]), float(pair[1])))
    r_tf =[]
    g_tf =[]
    b_tf =[]
    for triplets in triplets_list:
        a,b,c,d = triplets
        r_tf.append((float(a), float(b)))
        g_tf.append((float(a), float(c)))
        b_tf.append((float(a), float(d)))

    ray_casting_pp_st = time.time()
    local_image = process_data(sub_data_chunks, step_size, opacity_tf , r_tf , g_tf, b_tf)
    ray_casting_pp_et = time.time()
    #print(local_image.shape, "l")

    gathered_raycasting_times = comm.gather(ray_casting_pp_et-ray_casting_pp_st, root=0)
     # Gather all local images to the root processor
    b=time.time()
    '''
    tree_gather_st = time.time()
    gathered_images = tree_based_gather(local_image , rank % ch , ch , rank , n)
    tree_gather_et = time.time()
    gathered_times = comm.gather(tree_gather_et - tree_gather_st, root=0)
    #print(time.time() - a , rank)
    #print(gathered_images.shape)
    c=time.time()
    a=time.time()
    comm.Barrier()'''
    
    width = sub_data_chunks.shape[2]
    #print(width)
    tree_gather_st = time.time()
    gathered_images = binary_swap(local_image , width , ch , n)
    
    comm.Barrier()
    gathered_images = tree_based_gather_after1(gathered_images , rank % ch , ch , rank , n)
    if rank<ch:
        #print(len(gathered_images))
        a= np.array(gathered_images[0]).shape
        #print(a)
        even_res = np.zeros(a)
        num_proc_z = n//ch
        if num_proc_z>2:
            
            for i in range(0,num_proc_z,2):
                if(i>=num_proc_z/2):
                    break
                if i==0:
                    even_res = even_res + gathered_images[i]
                else:
                    even_res = np.concatenate((even_res , gathered_images[i]) , axis = 1)
                
                even_res = np.concatenate((even_res ,gathered_images[i+int(num_proc_z/2)]) , axis = 1)
            b= np.array(gathered_images[1]).shape
            odd_res=np.zeros(b)
            for i in range(1, num_proc_z,2):
                if(i>=num_proc_z/2):
                    break
                if i==1:
                    odd_res = odd_res + gathered_images[i]
                else:
                    odd_res = np.concatenate((odd_res , gathered_images[i]) , axis = 1)
                odd_res = np.concatenate((odd_res ,gathered_images[i+int(num_proc_z/2)]) , axis = 1)
            

            even_res = np.concatenate((even_res , odd_res) , axis = 1)
        else:
            even_res = np.array(gathered_images[0])
            even_res = np.concatenate((even_res , gathered_images[1]) , axis=1)

        gathered_images = np.array(even_res)
        

    tree_gather_et = time.time()
    gathered_times = comm.gather(tree_gather_et - tree_gather_st, root=0)
    new_comm = comm.Split(color=0 if rank < ch else MPI.UNDEFINED, key=rank)
    if rank < ch:
        gathered_images = np.array(gathered_images)
        #print(gathered_images.shape)
        gathered_images = gathered_images[: ,: ,:3]
        #img = Image.fromarray((gathered_images * 255 / gathered_images.max()).astype('uint8'))
        #img.show()
        #img.save(f'{rank}.png')
        #gathered_images += np.full_like(gathered_images, 0.1)
        tree_based_after_st = time.time()
        
        gathered_images = tree_based_gather_after(gathered_images ,0 ,ch, rank , n)
        tree_based_after_et = time.time()
        gathered_after_times = new_comm.gather(tree_based_after_et - tree_based_after_st, root=0)
        
        
    comm.Barrier()
    '''
    for i in range(n):
        comm.Reduce(local_image, result, op=MPI.SUM, root=rank%ch)
    for i in ch:'''
    #gathered_images = comm.Gather(local_image, root=0)
    
    #print(c-b , time.time() - a , rank)
    #print("gather done" , rank)
    if rank == 0:
        #print("after gather")
        gathered_images = np.array(gathered_images)
        #print(gathered_images.shape)
        # Combine all sub-images into the final image
        print("Stitching images.")
        
        final_image = np.zeros((x_size, y_size, 3), dtype=np.float32)

        # Decompose along both x and y directions
        x_chunk_size = ceil(x_size / num_proc_x)
        y_chunk_size = ceil(y_size / num_proc_y)
        z_chunk_size = ceil(z_size / num_proc_z)
        h=0
        for i in range(num_proc_x):
            for j in range(num_proc_y):
                    x_start = i * x_chunk_size
                    x_end = min((i + 1) * x_chunk_size, x_size)
                    y_start = j * y_chunk_size
                    y_end = min((j + 1) * y_chunk_size, y_size)

                    final_image[x_start:x_end, y_start:y_end] = gathered_images[h]
                    h+=1


        
        img = Image.fromarray((final_image * 255 / final_image.max()).astype('uint8'))
        
        #img.show()
        img.save('rendered_image1.png')

        print("Final image displayed.")
    
        end_time = time.time()
        total_time = end_time - start_time
        total_gather_time = max(gathered_times)
        # print_flush(f"total tree gather time: {total_gather_time}")

        total_gather_raycasting_time = max(gathered_raycasting_times)
        print_flush(f"Total raycasting time: {total_gather_raycasting_time}")

        total_gather_after_time = max(gathered_after_times)
        # print_flush(f"total tree after gather time: {total_gather_after_time}")
        print_flush(f"Total communication time: {total_gather_time+total_gather_after_time}")

        print_flush(f"Total computation time: {total_time - (total_gather_time+total_gather_after_time)}")
        print("Total time taken: ", total_time)


if __name__ == "__main__":
    main()
