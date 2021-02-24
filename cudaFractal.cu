#include <iostream>
//#include <fstream> // for files manipulation
#include <cstdlib>
#include <thrust/complex.h>
#include <GL/glut.h>
using namespace std;

int width =  1920;
int height = 1080;
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
vec3 *d_output;
vec3 *h_output;

__device__ vec3 fractalCalculator(thrust::complex<float> z,thrust::complex<float> c, int iterLimit);
__device__ float dims(int pos, int dim, float imDim, float imEdge);
__global__ void fractal(int *red, int *green, int *blue, dim3 dimens, vec3 imDims, vec3 imEdges, int iterLimit);
void sayError();
void renderLoop();
void runKernel();
void keyboard(unsigned char key, int x, int y);

__device__ vec3 mandlebrot(thrust::complex<float> c, int iterLimit) {
    thrust::complex<float> z(0, 0);
    int i = 0;
    if((256*powf(thrust::abs(c),4)-96*powf(thrust::abs(c),2)+32*c.real()-3 >= 0)
        && (16*powf(thrust::abs(c+1),2)-1 >= 0)) { //the given value of 4 was too big for me
        //detect and skip mandlebrot manlets
        //https://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
        //https://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
        do { z = z * z + c; } while (thrust::abs(z) < 2.0 && ++i <= iterLimit);
    } else { i = iterLimit; }
    vec3 returnValue;
    returnValue.x = static_cast<float>(i);
    returnValue.y = thrust::abs(z);
    returnValue.z = thrust::arg(z);
    return returnValue;
}

__device__ vec3 julia(thrust::complex<float> z, int iterLimit) {
    thrust::complex<float> c(0.26f, 0.0015f);
    int i = 0;
    do { z = z*z + c; } while(z.real() < 2 && z.real() > -2
                                && z.imag() < 2 && z.imag() > -2
                                && ++i <= iterLimit);
    vec3 returnValue;
    returnValue.x = static_cast<float>(i);
    returnValue.y = thrust::abs(z);
    returnValue.z = thrust::arg(z);
    return returnValue;
}

__device__ float dims(int pos, int dim, float imDim, float imEdge) {
    //normalize position out of max value for that position,
    //then multiply by zoomed-in dimension to turn original position into new one
    //then add offset away from the origin
    return static_cast<float>(pos)/static_cast<float>(dim)*imDim + imEdge;
}

__global__ void fractal(vec3 *output, dim3 dimens, vec3 imDims, vec3 imEdges, int iterLimit) {
    int x = blockIdx.x*blockDim.x + threadIdx.x; //you do NOT need the grid size to calculate the thread pos within the grid!!!
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x >= dimens.x || y >= dimens.y) {
//        output[y*dimens.y + x].x = -5; //not a good check: pixels not on the last row will print -5 inside next row's elements
        return;
    } //fixed: don't forget this, extra thread will do naughty things!
    // But you only hear about it once you try to copy the memory back to the host
    thrust::complex<float> pixel(dims(x,dimens.x,imDims.x,imEdges.x), //returns INT_MAX
                             dims(y,dimens.y,imDims.y,imEdges.y)); //returns 0
//    vec3 result = mandlebrot(pixel, iterLimit);
    vec3 result = julia(pixel, iterLimit);

//    if(result.x < iterLimit) { //if we are not in the converged region then give the tentacles nice colors
//        uint8_t temp = result.y/5*7;
//        output[y*dimens.x + x].x = (temp & 0x1) ? result.y/5.0f : 0.0f;
//        output[y * dimens.x + x].y = (temp & 0x4) ? result.y/5.0f : 0.0f;
//        output[y * dimens.x + x].z = (temp & 0x2) ? result.y/5.0f : 0.0f;
//    } else {
//        output[y*dimens.x + x].x = 0.0f; //fixed from y*dimens.y to y*dimens.x (row width, not column height)
//        output[y * dimens.x + x].y = 0.0f;
//        output[y * dimens.x + x].z = 0.0f;
//    }
    output[y*dimens.x + x].x = result.x / static_cast<float>(iterLimit);
//    output[y * dimens.x + x].y = result.y / 5.0f; //for green tentacles
    output[y * dimens.x + x].y = 0.0f;
    output[y * dimens.x + x].z = 0.0f;
    //(result.x == iterLimit) ? 1.0f : 0.0f; //ghostly blue outline
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
    glDrawPixels(width,height,GL_RGB,GL_FLOAT,h_output); //it didn't like it when I just used an array: I needed to wrap pixels in a struct
    glutSwapBuffers();
}

void runKernel() {
    dim3 blockSize(threadDim, threadDim, 1);
    dim3 gridSize(width/threadDim + 1, height/threadDim + 1, 1);
    dim3 dimsVar = dim3(width,height);
    vec3 imDim = vec3{imWidth,imHeight}; //fixed: dim3's rouding down 1 (0.999) causing div by 0 in GPU
    vec3 imEdge = vec3{imEdgeX,imEdgeY};
    fractal<<<gridSize,blockSize>>>(d_output,dimsVar,imDim,imEdge,iterLimit);
    cudaMemcpy(h_output,d_output,width*height*sizeof(vec3),cudaMemcpyDeviceToHost);
//    sayError();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 97:
            imEdgeX -= imWidth*.01;
            break;
        case 100:
            imEdgeX += imWidth*.01;
            break;
        case 119:
            imEdgeY += imHeight*.01;
            break;
        case 115:
            imEdgeY -= imHeight*.01;
            break;
        case 65:
            imEdgeX -= imWidth*.1;
            break;
        case 68:
            imEdgeX += imWidth*.1;
            break;
        case 87:
            imEdgeY += imHeight*.1;
            break;
        case 83:
            imEdgeY -= imHeight*.1;
            break;
        case 102: //f
            imWidth /= 1.2;
            imHeight /= 1.2;
            break;
        case 118: //v
            imWidth *= 1.2;
            imHeight *= 1.2;
            break;
        case 70: //f
            imWidth /= 2;
            imHeight /= 2;
            break;
        case 86: //v
            imWidth *= 2;
            imHeight *= 2;
            break;
        case 113: //press q to quit
            glutDestroyWindow(glutGetWindow());
            return;
        default:
            break;
    }
    runKernel();
    glutPostRedisplay();
}

int main (int argc, char *argv[])  {
    switch(argc) {
        default:
            cout << "No args given, using defaults" << endl;
            break;
        case 4:
            iterLimit = atof(argv[3]);
        case 3:
            //Problem: non-square width produces weird image artifacts (out of bounds in array?)
            //10x9 only goes up to array bound 81
            //Is 1 pixel is missing off of the end of each line?
            //No, array seems to match up to bounds and yet the image is still skewed.
            //Solution:
            //I checked the array with x/fw and saw that having a bounds check of >= appeared to be too constrictive,
            //so I took it off...but upon further inspection, I realized that that would make no sense as it would allow
            //invalid values to go into the array index. I realized it must have been something up with the array indexing
            //converer (2d to 1d) and I found that I had [y*dim.y + x] instead of [y*dim.x + x]! I think that the dim.y
            //was left over from when I had [x*dim.y + y] originally, which displays the mandlebrot upright.
            width = atof(argv[1]);
            height = atof(argv[2]);
            break;
    }

    glutInit(&argc, argv);
    glutInitWindowPosition(-1,-1);
    glutInitWindowSize(width,height);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("CUDA Fractal Viewer");

    cudaMalloc(&d_output,width*height*sizeof(vec3));
    h_output = new vec3[width*height];

    runKernel();
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(renderLoop);
    glutMainLoop();

    cudaFree(d_output);
    delete[] h_output;
    return 0;
}
