from mpi4py import MPI
import numpy as np
import sys
import os
from math import ceil, sqrt
from PIL import Image

import vtkmodules.all as vtk

import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

triplets_list = []
with open('color_TF.txt', 'r') as file:
    values = file.read().replace(',', '').split()

for i in range(0, len(values), 4):
    triplet = (values[i], values[i + 1], values[i + 2], values[i + 3])
    triplets_list.append(triplet)


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


def trim_data(data, x_min, x_max, y_min, y_max):
    """Trim the numpy array according to xy bounds."""
    print_flush(f"Trimming data to x: [{x_min}, {x_max}] and y: [{y_min}, {y_max}]")
    trimmed_data = data[:, x_min:x_max + 1, y_min:y_max + 1]
    #trimmed_data = trimmed_data[:, ::-1, ::-1]
    print_flush("Data trimmed successfully.")
    return trimmed_data


# def partition_data(data, num_procs, partition_type):
#     """Partition data based on the specified partition type."""
#     z_size, x_size, y_size = data.shape
#     sub_data = []

#     if partition_type == 1:
#         # Decompose along x direction only
#         print_flush("Partitioning data along x direction...")
#         chunk_size_x = ceil(x_size / num_procs)
#         for i in range(num_procs):
#             x_start = i * chunk_size_x
#             x_end = min((i + 1) * chunk_size_x, x_size)
#             sub_data.append(data[:, x_start:x_end, :])
#         print_flush("Data partitioned along x direction.")

#     else:
#         # Decompose along both x and y directions
#         print_flush("Partitioning data along both x and y directions...")
#         for i in range(int(sqrt(num_procs)), 0, -1):
#             if num_procs % i == 0:
#                 y_num = i
#                 x_num = num_procs // i
#                 break
#         x_chunk_size = ceil(x_size / x_num)
#         y_chunk_size = ceil(y_size / y_num)
#         for i in range(x_num):
#             for j in range(y_num):
#                 x_start = i * x_chunk_size
#                 x_end = min((i + 1) * x_chunk_size, x_size)
#                 y_start = j * y_chunk_size
#                 y_end = min((j + 1) * y_chunk_size, y_size)
#                 sub_data.append(data[:, x_start:x_end, y_start:y_end])
#         print_flush("Data partitioned along both x and y directions.")

#     return sub_data

from math import ceil

def partition_data(data, x_partitions, y_partitions, z_partitions):
    """Partition data based on specified x, y, and z partitions."""
    z_size, x_size, y_size = data.shape
    sub_data = []

    # Calculate chunk sizes for each direction
    x_chunk_size = ceil(x_size / x_partitions)
    y_chunk_size = ceil(y_size / y_partitions)
    z_chunk_size = ceil(z_size / z_partitions)

    print_flush("Partitioning data along x, y, and z directions...")
    for i in range(x_partitions):
        for j in range(y_partitions):
            for k in range(z_partitions):
                x_start = i * x_chunk_size
                x_end = min((i + 1) * x_chunk_size, x_size)
                y_start = j * y_chunk_size
                y_end = min((j + 1) * y_chunk_size, y_size)
                z_start = k * z_chunk_size
                z_end = min((k + 1) * z_chunk_size, z_size)
                
                # Append the partitioned data slice
                sub_data.append(data[z_start:z_end, x_start:x_end, y_start:y_end])

    print_flush("Data partitioned along x, y, and z directions.")
    return sub_data


def linear_interpolation(pairs_list, x):
    for i in range(len(pairs_list) - 1):
        x1, y1 = float(pairs_list[i][0]), float(pairs_list[i][1])
        x2, y2 = float(pairs_list[i + 1][0]), float(pairs_list[i + 1][1])

        if x1 <= x <= x2:
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            return y

    return 0.0


def ray_casting(data_chunk, step_size, opacity_tf):
    """Ray casting on a decomposed 3D data chunk with image compositing."""
    z_size, x_size, y_size = data_chunk.shape
    rays_terminated = 0
    partial_image = np.zeros((x_size, y_size, 3), dtype=np.float32)  # RGB channels
    r_tf = vtk.vtkPiecewiseFunction()
    g_tf = vtk.vtkPiecewiseFunction()
    b_tf = vtk.vtkPiecewiseFunction()

    def calc_transfer_functions():
        """Set up transfer functions to map input values to colors."""
        for val, r, g, b in triplets_list:
            scalar_value = float(val)
            r_tf.AddPoint(scalar_value, float(r))
            g_tf.AddPoint(scalar_value, float(g))
            b_tf.AddPoint(scalar_value, float(b))

    calc_transfer_functions()

    def opacity_ttf(x):
        """Get opacity from transfer function based on scalar value x."""
        return opacity_tf.GetValue(x)

    # Perform ray casting for each pixel in the x-y plane of the data chunk
    for x in range(x_size):
        for y in range(y_size):
            color_accum = np.zeros(3)
            opacity_accum = 0

            z = 0
            while z < z_size:
                # Sample the scalar value in the z direction (interpolated if step_size > 1)
                if step_size == 1:
                    value = data_chunk[int(z), x, y]
                else:
                    z1 = int(z)
                    z2 = min(z1 + 1, z_size - 1)
                    alpha = z - z1
                    value = (1 - alpha) * data_chunk[z1, x, y] + alpha * data_chunk[z2, x, y]

                # Map scalar value to color using transfer functions
                r = r_tf.GetValue(value)
                g = g_tf.GetValue(value)
                b = b_tf.GetValue(value)
                color = np.array([r, g, b])
                opacity = opacity_ttf(value)

                # Composite color and opacity
                color_accum += (1 - opacity_accum) * color * opacity
                opacity_accum += opacity * (1 - opacity_accum)

                # Early ray termination if opacity accumulates to 1.0
                if opacity_accum >= 1.0:
                    rays_terminated += 1
                    break

                z += step_size

            # Store the accumulated color in the partial image
            partial_image[x, y] = np.clip(color_accum, 0, 1)

    return partial_image, rays_terminated


def process_data(sub_data_chunk, step_size, opacity_tf):
    """Process the chunk of data for ray casting with transfer functions."""
    return ray_casting(sub_data_chunk, step_size, opacity_tf)


def read_file_to_pairs(filename):
    pairs_list = []
    with open(filename, 'r') as file:
        values = file.read().replace(',', '').split()

    for i in range(0, len(values), 2):
        pair = (values[i], values[i + 1])
        pairs_list.append(pair)

    return pairs_list


def read_file_to_triplets(filename):
    triplets_list = []
    with open(filename, 'r') as file:
        values = file.read().replace(',', '').split()

    for i in range(0, len(values), 4):
        triplet = (values[i], values[i + 1], values[i + 2], values[i + 3])
        triplets_list.append(triplet)

    return triplets_list


def composite_pixelwise(image1, image2):
    """
    Perform pixel-wise compositing of two images using the "Over" operator.
    
    Parameters:
    - image1, image2: Two images to composite (RGB or RGBA).
    
    Returns:
    - Composited image.
    """
    # Add an alpha channel if not present
    if image1.shape[2] == 3:
        alpha_channel1 = np.ones((image1.shape[0], image1.shape[1], 1))
        image1 = np.concatenate((image1, alpha_channel1), axis=2)

    if image2.shape[2] == 3:
        alpha_channel2 = np.ones((image2.shape[0], image2.shape[1], 1))
        image2 = np.concatenate((image2, alpha_channel2), axis=2)

    opacity1 = image1[..., 3]  # Alpha channel for image1
    opacity2 = image2[..., 3]  # Alpha channel for image2

    # Composite RGB channels using the "Over" operator
    composite_rgb = (image1[..., :3] * opacity1[:, :, None] +
                     image2[..., :3] * opacity2[:, :, None] * (1 - opacity1[:, :, None]))
    composite_alpha = opacity1 + opacity2 * (1 - opacity1)

    # Stack RGB and composited alpha into a single image
    composited_image = np.zeros_like(image1)
    composited_image[..., :3] = composite_rgb
    composited_image[..., 3] = composite_alpha
    return composited_image



def main():
    if rank == 0:
        # start_time = time.time()
        if len(sys.argv) != 8:
            print_flush("Usage: python script.py <filename> <partition> <step_size> <x_min> <x_max> <y_min> <y_max>")
            sys.exit(1)

        filename = sys.argv[1]
        # partition_type = int(sys.argv[2])
        x, y, z = map(int, sys.argv[2:5])
        step_size = float(sys.argv[5])
        
        print_flush(f"x: {x}, y: {y}, z: {z}, step size: {step_size}")
        

        # num_rays = (y_max - y_min + 1) * (x_max - x_min + 1)
        dims = read_file_dimensions(filename)
        data = read_raw_file(filename, dims)

    #     # Trim the data based on x and y bounds
    #     # trimmed_data = trim_data(data, x_min, x_max, y_min, y_max)
    #     num_rays = trimmed_data[0].size 

        # Partition the data
        sub_data_chunks = partition_data(data, x, y, z)
        chunk_shapes = [chunk.shape for chunk in sub_data_chunks]  # Get shapes for each chunk

        # Prepare counts and displacements for scatterv
        counts = [chunk.size for chunk in sub_data_chunks]
        displacements = [sum(counts[:i]) for i in range(size)]

        # Flatten all data
        flattened_data = np.concatenate([chunk.flatten() for chunk in sub_data_chunks])
    else:
        # On non-root ranks, initialize variables as None to receive the broadcast/scatter
        sub_data_chunks = None
        chunk_shapes = None
        step_size = None
        flattened_data = None
        counts = None
        displacements = None
    if rank == 0:
        step_size = float(sys.argv[5])
    step_size = comm.bcast(step_size, root=0)
    # Broadcast chunk shapes to all ranks
    chunk_shapes = comm.bcast(chunk_shapes, root=0)

    # Broadcast transfer functions to all ranks
    transfer_functions = comm.bcast(None, root=0)

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
    opacity_tf_pairs = read_file_to_pairs("opacity_TF.txt")
    opacity_tf = vtk.vtkPiecewiseFunction()
    for pair in opacity_tf_pairs:
        opacity_tf.AddPoint(float(pair[0]), float(pair[1]))

    partial_image , rays_terminated = process_data(sub_data_chunk, step_size, opacity_tf)
    print_flush(f"ray casting completed by process: {rank}")

    image_shape = partial_image.shape
    final_image = np.zeros_like(partial_image) if rank==0 else None

    step = 1
    while(step < size):
        partner_rank = rank ^ step

        if(partner_rank<size):
            recv_image = np.empty_like(partial_image)

            comm.Sendrecv(partial_image, dest=partner_rank, recvbuf=recv_image, source=partner_rank)
            partial_image = composite_pixelwise(partial_image, recv_image)

        step *= 2
    if rank == 0:
        all_images = np.empty((size, *image_shape), dtype=partial_image.dtype)
    else:
        all_images = None

    comm.Gather(partial_image, all_images, root=0)

    if rank == 0:
        final_image = all_images[0]
        for img in all_images[1:]:
            final_image = composite_pixelwise(final_image, img)



     # Gather all local images to the root processor
    # gathered_images = comm.gather(local_image, root=0)
    # array = comm.gather(rays_terminated ,root=0)
    # if rank == 0:
    #     # Combine all sub-images into the final image
    #     print_flush("Stitching images.")
    #     z_size, x_size, y_size = data.shape
    #     final_image = np.zeros((x_size, y_size, 3), dtype=np.float32)

    #     # Calculate chunk sizes based on partitions
    #     x_chunk_size = ceil(x_size / x)
    #     y_chunk_size = ceil(y_size / y)
    #     part = 0
    #     for i in range(x):
    #         for j in range(y):
    #             for k in range(z):
    #                 # Extract sub-image and place it in the correct location within the final image
    #                 sub_image = gathered_images[part]
    #                 part += 1

    #                 x_start = i * x_chunk_size
    #                 x_end = x_start + sub_image.shape[0]
                
    #                 y_start = j * y_chunk_size
    #                 y_end = y_start + sub_image.shape[1]

    #                 # Add the sub-image to the final image
    #                 final_image[x_start:x_end, y_start:y_end] = sub_image

        img = Image.fromarray((final_image * 255 / final_image.max()).astype('uint8'))
        img.save('rendered_image1.png')
        print_flush("Final image displayed.")
        
        # Display the final image using PIL
        #print(final_image)

        # img = Image.fromarray((final_image * 255 / final_image.max()).astype('uint8'))
        # #img.show()
        # img.save('rendered_image1.png')

        # print_flush("Final image displayed.")
        # early_ray_termination_count = 0
        # for i in range(size):
        #     early_ray_termination_count = early_ray_termination_count + array[i]
        #print(array)
        # print("Number of early terminated rays are: ", early_ray_termination_count) 
        # print("Percentage of early terminated rays are: ", (early_ray_termination_count / num_rays) *100) 
        # end_time = time.time()
        # total_time = end_time - start_time
        # print("Total time taken: ", total_time)


if __name__ == "__main__":
    main()