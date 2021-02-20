#include <iostream>
#include <fstream> // for files manipulation
#include <complex> // for complex numbers
#include <cstdlib>
using namespace std;

int width =  800;
int height = 800;
float imWidth = 1;
float imHeight = 1;
float imEdgeX = -1.5;
float imEdgeY = -0.5;

struct vec3 {
	int x = 0;
	int y = 0;
	int z = 0;
};

vec3 fractal(complex<float> &z,complex<float> &c) {
	int iter = 0;
	while(abs(z) < 2 && iter <= 40) { //check for boundedness
		//2 is just a low enough bounds check, a quick detector for if
		//the number is going to blow up or not
		z = z*z + c;
		iter++;
	}
	vec3 returnValue;
	returnValue.x = iter;
	returnValue.y = abs(z);
	returnValue.z = arg(z);
	return returnValue;
}

float dims(int pos, int dim, float imDim, float imEdge) {
	return static_cast<float>(pos)/static_cast<float>(dim)*imDim + imEdge;
}

bool ppmWriter(int length, int width, int *red, int *green, int *blue) {
    ofstream my_Image ("mandelbrot.ppm");
    if(!my_Image.is_open()) { return false; }
    my_Image << "P3\n" << width << " " << height << " 255\n";
    for(int i = 0; i < length*width; i++) {
        my_Image << red[i] << ' ' << green[i] << ' ' << blue[i] << "\n";
    }
    my_Image.close();
}

int main (int argc, char *argv[])  {
	switch(argc) {
	default:
		cout << "No args given, using defaults" << endl;
		break;
	case 5:
		imEdgeX = atof(argv[3]);
		imEdgeY = atof(argv[4]);
	case 3:
		imWidth = atof(argv[1]);
		imHeight = atof(argv[2]);
		break;
	}

	int red[width*height],
        green[width*height],
        blue[width*height];
    for (int x = 0; x < width; x++) {
         for (int y = 0; y < height; y++)  {
            vec3 pixel;
            pixel.z = 0;
            //complex<float> c((float)x/width-1.5, (float)y/height-0.5);
            complex<float> c(dims(x,width,imWidth,imEdgeX),\
                            dims(y,height,imHeight,imEdgeY));
            //linearly maps x values to [-1.5,-0.5] and y to [-0.5,-0.5]
            complex<float> z(0, 0); //must declare precision of complex #
            vec3 result = fractal(z,c);
            pixel.y = result.y/5.0*255;
            float mapping = result.x/40.0*255.0;
            //pixel.x = (abs(result.z)/90 <= 255) ? abs(result.z)/90 : 255;
            pixel.z = (mapping <= 255) ? mapping : 255; //iterations until inf
            pixel.y = (pixel.y <= 255) ? pixel.y : 255; //convergent value
            red[y*height + x] = 17;
            green[y*height + x] = pixel.y;
            blue[y*height + x] = pixel.z; //flip x and y, and height and width, to rotate image
         }
    }
    ppmWriter(width,height,red,green,blue);
    return 0;
}

