#include <iostream>
//#include <fstream> // for files manipulation
#include <cstdlib>
#include <thrust/complex.h>
#include <GL/glut.h>
using namespace std;

int width =  1920;
int height = 1920;
float imWidth = 1;
float imHeight = 1;
float imEdgeX = -1.5;
float imEdgeY = -0.5;
int iterLimit = 40;

struct vec3 { //dim3's are only for ints, so this one is for floats
    float x = 0;
    float y = 0;
    float z = 0;
};

const int threadDim = 32;
const dim3 blockSize(threadDim, threadDim, 1);
const dim3 gridSize(width/threadDim + 1, height/threadDim + 1, 1);
dim3 dimsVar = dim3(width,height);
vec3 imDim = vec3{imWidth,imHeight}; //fixed: dim3's rouding down 1 (0.999) causing div by 0 in GPU
vec3 imEdge = vec3{imEdgeX,imEdgeY};
int *d_red, *d_green, *d_blue;
int *h_red, *h_green, *h_blue;



__device__ vec3 fractalCalculator(thrust::complex<float> z,thrust::complex<float> c, int iterLimit) {
    int iter = 0;
    while(thrust::abs(z) < 2.0 && iter <= iterLimit) { //4 for ship
        z = z*z + c; //mandlebrot
//        float zReal = z.real(), zImag = z.imag();
//        thrust::complex<float> zShip(zReal < 0 ? -1*zReal : zReal,
//                                     zImag < 0 ? -1*zImag : zImag);
//        z = zShip*zShip + c;
        iter++;
    }
    vec3 returnValue;
    returnValue.x = iter;
    returnValue.y = thrust::abs(z);
    returnValue.z = thrust::arg(z);
    return returnValue;
}

__device__ float dims(int pos, int dim, float imDim, float imEdge) {
    //linearly maps x values to [-1.5,-0.5] and y to [-0.5,-0.5]
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
    thrust::complex<float> z(0, 0); //must declare precision of complex #
    vec3 result = fractalCalculator(z,c, iterLimit);
    int absScaled = static_cast<int>(result.y/5.0*INT_MAX);
    int iterScaled = static_cast<int>(result.x/static_cast<float>(iterLimit))*INT_MAX;
    red[y*dimens.y + x] = 0; //flip x and y, and height and width, to rotate image
    green[y*dimens.y + x] = absScaled;
    blue[y*dimens.y + x] = iterScaled;
}

void sayError() {
    cudaError_t cudaResult = cudaGetLastError();
    if (cudaResult != cudaSuccess) {
        std::cout << cudaGetErrorString(cudaResult) << std::endl;
    } else {
        std::cout << "No CUDA errors found" << std::endl;
    }
}

void renderLoop() {
    glClear(GL_DEPTH_BUFFER_BIT);
    glDrawPixels(width,height,GL_RED,GL_INT,h_green);
    glutSwapBuffers();
}

void runKernel() {
    cout << "Running kernel..." << endl;
    dimsVar = dim3(width,height);
    imDim = vec3{imWidth,imHeight}; //fixed: dim3's rouding down 1 (0.999) causing div by 0 in GPU
    imEdge = vec3{imEdgeX,imEdgeY};
    fractal<<<gridSize,blockSize>>>(d_red,d_green,d_blue,dimsVar,imDim,imEdge,iterLimit);
    cudaMemcpy(h_red,d_red,width*height*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green,d_green,width*height*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue,d_blue,width*height*sizeof(int),cudaMemcpyDeviceToHost);
    sayError();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 97:
            imEdgeX -= imWidth*.01;
            cout << "Left: imEdgeX is now " << imEdgeX << endl;
            break;
        case 100:
            imEdgeX += imWidth*.01;
            cout << "Right: imEdgeX is now " << imEdgeX << endl;
            break;
        case 119:
            imEdgeY += imHeight*.01;
            cout << "Up: imEdgeY is now " << imEdgeY << endl;
            break;
        case 115:
            imEdgeY -= imHeight*.01;
            cout << "Down: imEdgeY is now " << imEdgeY << endl;
            break;
        case 65:
            imEdgeX -= imWidth*.1;
            cout << "Super left: imEdgeX is now " << imEdgeX << endl;
            break;
        case 68:
            imEdgeX += imWidth*.1;
            cout << "Super right: imEdgeX is now " << imEdgeX << endl;
            break;
        case 87:
            imEdgeY += imHeight*.1;
            cout << "Super up: imEdgeY is now " << imEdgeY << endl;
            break;
        case 83:
            imEdgeY -= imHeight*.1;
            cout << "Super down: imEdgeY is now " << imEdgeY << endl;
            break;
        case 102: //f
            imWidth /= 1.2;
            imHeight /= 1.2;
            cout << "Focusing in: imWidth is now " << imWidth << endl;
            break;
        case 118: //v
            imWidth *= 1.2;
            imHeight *= 1.2;
            cout << "Focusing out: imWidth is now " << imWidth << endl;
            break;
        case 70: //f
            imWidth /= 2;
            imHeight /= 2;
            cout << "Super focusing in: imWidth is now " << imWidth << endl;
            break;
        case 86: //v
            imWidth *= 2;
            imHeight *= 2;
            cout << "Super focusing out: imWidth is now " << imWidth << endl;
            break;
        case 113: //press q to quit
            glutDestroyWindow(glutGetWindow());
            return;
            break;
        default:
            cout << "Default key case" << endl;
            break;
    }
    runKernel();
    glutPostRedisplay();
}

int main (int argc, char *argv[])  {
    glutInit(&argc, argv);
    glutInitWindowPosition(-1,-1);
    glutInitWindowSize(800,800);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("CUDA Fractal Viewer");


    switch(argc) {
        default:
            cout << "No args given, using defaults" << endl;
            break;
        case 4:
            iterLimit = atof(argv[3]);
        case 3:
            width = atof(argv[1]);
            height = atof(argv[2]);
            break;
	}


	cudaMalloc(&d_red,width*height*sizeof(int));
    cudaMalloc(&d_green,width*height*sizeof(int));
    cudaMalloc(&d_blue,width*height*sizeof(int));
    h_red = new int[width*height];
    h_green = new int[width*height];
    h_blue = new int[width*height];

    runKernel();
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(renderLoop);
    glutMainLoop();

    delete[] h_red;
    delete[] h_green;
    delete[] h_blue;
    return 0;
}

