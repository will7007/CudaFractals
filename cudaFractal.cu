#include <iostream>
#include <fstream> // for files manipulation
#include <cstdlib>
#include <thrust/complex.h>
using namespace std;

struct vec3 { //dim3's are only for ints, so this one is for floats
    float x = 0;
    float y = 0;
    float z = 0;
};

__device__ vec3 fractalCalculator(thrust::complex<float> z,thrust::complex<float> c, int iterLimit) {
    int iter = 0;
    while(thrust::abs(z) < 2.0 && iter <= iterLimit) {
        z = z*z + c;
        iter++;
    }
    vec3 returnValue;
    returnValue.x = iter;
    returnValue.y = thrust::abs(z);
    returnValue.z = thrust::arg(z);
    return returnValue;
}

__device__ float dims(int pos, int dim, float imDim, float imEdge) {
    return static_cast<float>(pos)/static_cast<float>(dim)*imDim + imEdge;
}

__global__ void fractal(int *red, int *green, int *blue, dim3 dimens, vec3 imDims, vec3 imEdges, int iterLimit) {
//    int x = (blockIdx.x*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
//    int y = (blockIdx.y*gridDim.y + blockIdx.y)*blockDim.y + threadIdx.y;
    int x = blockIdx.x*blockDim.x + threadIdx.x; //you do NOT need the grid size to calculate the thread pos within the grid!!!
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= dimens.x || y >= dimens.y) { return; } //fixed: don't forget this, extra thread will do naughty things!
    // But you only hear about it once you try to copy the memory back to the host
    thrust::complex<float> c(dims(x,dimens.x,imDims.x,imEdges.x), //returns INT_MAX
                             dims(y,dimens.y,imDims.y,imEdges.y)); //returns 0
    //linearly maps x values to [-1.5,-0.5] and y to [-0.5,-0.5]
    thrust::complex<float> z(0, 0); //must declare precision of complex #
    vec3 result = fractalCalculator(z,c, iterLimit);
    int absScaled = static_cast<int>(result.y/5.0*255.0);
    int iterScaled = static_cast<int>(result.x/static_cast<float>(iterLimit)*255.0);
    red[y*dimens.y + x] = 0; //flip x and y, and height and width, to rotate image
    green[y*dimens.y + x] = (absScaled <= 255) ? absScaled : 255;
    blue[y*dimens.y + x] = (iterScaled <= 255) ? iterScaled : 255;
}

void ppmWriter(int width, int height, int *red, int *green, int *blue) {
    ofstream my_Image ("mandelbrot.ppm");
    if(!my_Image.is_open()) { return; }
    my_Image << "P3\n" << width << " " << height << " 255\n";
    for(int i = 0; i < height*width; i++) {
        my_Image << red[i] << ' ' << green[i] << ' ' << blue[i] << "\n";
    }
    my_Image.close();
}

void sayError() {
    cudaError_t cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) {
        std::cout << cudaGetErrorString(cudaResult) << std::endl;
    } else {
        std::cout << "No CUDA errors found" << std::endl;
    }
}

int main (int argc, char *argv[])  {
    int width =  800;
    int height = 800;
    float imWidth = 1;
    float imHeight = 1;
    float imEdgeX = -1.5;
    float imEdgeY = -0.5;
    int iterLimit = 40;
    switch(argc) {
        default:
            cout << "No args given, using defaults" << endl;
            break;
        case 8:
            iterLimit = atof(argv[7]);
        case 7:
            width = atof(argv[5]);
            height = atof(argv[6]);
        case 5:
            imEdgeX = atof(argv[3]);
            imEdgeY = atof(argv[4]);
        case 3:
            imWidth = atof(argv[1]);
            imHeight = atof(argv[2]);
            break;
	}

    int *d_red, *d_green, *d_blue;
	cudaMallocManaged(&d_red,width*height*sizeof(int));
    cudaMallocManaged(&d_green,width*height*sizeof(int));
    cudaMallocManaged(&d_blue,width*height*sizeof(int));

    const int threadDim = 32;
    const dim3 blockSize(threadDim, threadDim, 1);
    const dim3 gridSize(width/threadDim + 1, height/threadDim + 1, 1);
    const dim3 dims = dim3(width,height);
    const vec3 imDim = vec3{imWidth,imHeight}; //fixed: dim3's rouding down 1 (0.999) causing div by 0 in GPU
    const vec3 imEdge = vec3{imEdgeX,imEdgeY};
    fractal<<<gridSize,blockSize>>>(d_red,d_green,d_blue,dims,imDim,imEdge,iterLimit);

    int *h_red = new int[width*height];
    int *h_green = new int[width*height];
    int *h_blue = new int[width*height];
    cudaMemcpy(h_red,d_red,width*height*sizeof(int),cudaMemcpyDefault);
    cudaMemcpy(h_green,d_green,width*height*sizeof(int),cudaMemcpyDefault);
    cudaMemcpy(h_blue,d_blue,width*height*sizeof(int),cudaMemcpyDefault);

    ppmWriter(width,height,h_red,h_green,h_blue);
    sayError();
    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;

    return 0;
}

