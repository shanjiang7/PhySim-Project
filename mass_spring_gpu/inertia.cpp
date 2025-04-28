#include <iostream>
#include <vector>
#include <cassert>
#include <muda/muda.h>
#include <muda/container.h>
using namespace muda;

//DeviceBuffer is "A lightweight std::vector-like cuda device memory container."
// I am replacing "std::vector" with "DeviceBuffer" to put the arrays on GPUs
double val(const DeviceBuffer<double>& points, 
           const DeviceBuffer<double>& points_prev, 
           const DeviceBuffer<double>& masses)
{
    int N = points.size();
    DeviceBuffer<double> device_val(N); // temporary buffer to store local energy per entry

    //launch GPU loop
    // "MUDA Viewers provide safe inner-kernel memory access, which checks all input to ensure access does not go out of range and does not dereference a null pointer. If something goes wrong, they report the debug information as much as possible and trap the kernel to prevent further errors."
    // cviewer gives a read-only view to GPU memory
    // viewer gives a writeable view
    // parrallelFor(256) launches a GPU kernel with blovk size 256
    // inside the __device__ kernel it compute the enrgy E_I for each entry
    // device sum summs it up
    ParallelFor(256)
        .apply(N, 
            [device_val = device_val.viewer(), 
            points = points.cviewer(), 
            points_prev = points_prev.cviewer(), 
            masses = masses.cviewer()]__device__(int i) mutable
            {
                // perform the calcualtions
                double diff = points(i) - points_prev(i);
                device_val(i) = 0.5 * masses(i)*diff*diff;
            })
        // "Further, you can use muda::Debug::set_sync_callback() to retrieve the output once wait() is called, as:"
        .wait();
    return devicesum(device_val)
}






//example usage
int main()
{
    // h_points are on CPU
    // d_points are on GPU

    std::vector<double> h_points = {2.0, 2.0, 2.0};
    std::vector<double> h_points_prev = {1.0, 1.0, 1.0};
    std::vector<double> H_masses = {1.0, 1.0, 1.0};

    DeviceBuffer<double> d_points(h_points.size());
    DeviceBuffer<double> d_points_prev(h_points_prev.size());
    DeviceBuffer<double> d_masses(h_masses.size());

    d_points.copy_from(h_points);
    d_points_prev.copy_from(h_points_prev);
    d_masses.copy_from(h_masses);


    double energy = val(d_points, d_points_prev, d_masses);
    std::cout << "Inertia; energy: " << energy << std::endl;

    return 0;
}