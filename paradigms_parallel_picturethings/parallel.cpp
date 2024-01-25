//#pragma error(disable : 4996)

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "lodepng.h"
//#include "cl.hpp";
#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace lodepng;
//using namespace cl;

const string CHOOSE_PICTURE = "Choose the picture:\n1 - 300x300 px\n2 - 400x400 px\n3 - 500x500 px\n4 - 600x600 px\n5 - 950x950 px\n6 - 2400x2400 px\nNote that entering number other than 1-6 will result in an error!\n";
const string CHOOSE_FILTER = "Choose the filter:\nNC - Negative consequentially\nNOMP - Negative using OMP\nNV - Negative using vectorization\nNOCL - Negative using OCL(deprecated)\nGC - Gaussian Blur consequentially\nGOMP - Gaussian Blur using OMP\nGV - Gaussian Blur using vectorization\nGOCL - Gaussian Blur using OCL(deprecated)\n";
const string CHOOSE_AMOUNT_OF_ITERATIONS = "Now enter the amount of iterations you need. Please only enter integer numbers\n";
const double PI = 3.1415926535;

vector<unsigned char> convertToThreeChannels(vector<unsigned char> input, int width, int height) {
    const int numPixels = width * height;
    const int numChannels = 4;
    vector<unsigned char> output(numPixels * 3);
    for (int i = 0, j = 0; i < numPixels * numChannels; i += numChannels, j += 3) {
        output[j] = input[i];
        output[j + 1] = input[i + 1];
        output[j + 2] = input[i + 2];
    }
    return output;
}

void applyNegativeFilterConsequentially(vector<unsigned char>& image) {
    for (int i = 0; i < image.size(); i += 4) {
        image[i] = 255 - image[i];
        image[i + 1] = 255 - image[i + 1];
        image[i + 2] = 255 - image[i + 2];
    }
}

void applyNegativeFilterUsingOMP(vector<unsigned char>& image) {
    omp_set_num_threads(omp_get_max_threads());
    int i;
#pragma omp parallel for shared(image) private(i)
    for (i = 0; i < image.size(); i += 4) {
        image[i] = 255 - image[i];
        image[i + 1] = 255 - image[i + 1];
        image[i + 2] = 255 - image[i + 2];
    }
}

void applyNegativeFilterUsingVectorizaion(vector<unsigned char>& image) {
    const __m128i pixelLimit = _mm_set1_epi8(255);
    for (int i = 0; i < image.size(); i += 16) {
        __m128i pixel = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&image[i]));
        pixel = _mm_sub_epi8(pixelLimit, pixel);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&image[i]), pixel);
    }
}

//void applyNegativeFilterUsingOCL(vector<unsigned char>& image) {
//    const char* saxpy_kernel =
//        "__kernel                                           \n"
//        "void saxpy_kernel(                                 \n"
//        "                   __global int VECTOR_SIZE,       \n"
//        "                   __global unsigned char* vector) \n"
//        "{                                                  \n"
//        "   int index = get_global_id(0);                   \n"
//        "   if (index >= VECTOR_SIZE) return;               \n"
//        "   input[index] = 255 - input[index];              \n"
//        "}                                                  \n";
//    vector<Platform> platforms;
//    Platform::get(&platforms);
//    vector<Device> devices = vector<Device>();
//    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
//    Device device = devices[0];
//    vector<Device> contextDevices;
//    contextDevices.push_back(device);
//    Context context(contextDevices);
//    CommandQueue queue(context, device);
//    Buffer clmInputImage = Buffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image.size()*sizeof(unsigned char), &image);
//    string sourceCode(saxpy_kernel);
//    Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length()+1));
//    Program program = Program(context, source);
//    program.build(contextDevices);
//    Kernel kernel(program, "saxpy_kernel");
//    int iArg = 0;
//    kernel.setArg(iArg++, image.size());
//    kernel.setArg(iArg++, clmInputImage);
//    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(image.size()), NDRange(64));
//    queue.finish();
//    queue.enqueueReadBuffer(clmInputImage, CL_TRUE, 0, image.size() * sizeof(unsigned int), &image);
//} 

vector<float> setKernel(int radius, float sigma) {
    vector<float> kernel(2 * radius + 1);
    float sigma_sq = sigma * sigma;
    float sum = 0;
    for (int i = -radius; i <= radius; i++) {
        float r = sqrt(2 * PI) * sigma;
        float num = exp(-(i * i) / (2 * sigma_sq));
        kernel[i + radius] = num / r;
        sum += kernel[i + radius];
    }
    for (int i = 0; i < kernel.size(); i++) {
        kernel[i] /= sum;
    }
    return kernel;
}

void applyGaussianBlurConsequentially(std::vector<unsigned char>& image, unsigned int width, unsigned int height, float sigma, int radius) {
    vector<float> kernel = setKernel(radius, sigma);
    vector<unsigned char> temp_image(image.size());
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0, g = 0, b = 0;
            for (int i = -radius; i <= radius; i++) {
                int index = 4 * (y * width + min(max(x + i, 0), int(width - 1)));
                r += kernel[i + radius] * image[index];
                g += kernel[i + radius] * image[index + 1];
                b += kernel[i + radius] * image[index + 2];
            }
            int index = 4 * (y * width + x);
            temp_image[index] = r;
            temp_image[index + 1] = g;
            temp_image[index + 2] = b;
            temp_image[index + 3] = image[index + 3];
        }
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0, g = 0, b = 0;
            for (int i = -radius; i <= radius; i++) {
                int index = 4 * (min(max(y + i, 0), int(height - 1)) * width + x);
                r += kernel[i + radius] * temp_image[index];
                g += kernel[i + radius] * temp_image[index + 1];
                b += kernel[i + radius] * temp_image[index + 2];
            }
            int index = 4 * (y * width + x);
            image[index] = r;
            image[index + 1] = g;
            image[index + 2] = b;
            image[index + 3] = temp_image[index + 3];
        }
    }
}

void applyGaussianBlurUsingOMP(vector<unsigned char>& image, int width, int height, float sigma, int radius) {
    vector<float> kernel = setKernel(radius, sigma);
    vector<unsigned char> temp_image(image.size());
    int y, x, i, index;
    float r, g, b;
    omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for shared (kernel, image, temp_image, width, height) private (y, x, i, index, r, g, b)
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            r = 0, g = 0, b = 0;
            for (i = -radius; i <= radius; i++) {
                index = 4 * (y * width + min(max(x + i, 0), int(width - 1)));
                r += kernel[i + radius] * image[index];
                g += kernel[i + radius] * image[index + 1];
                b += kernel[i + radius] * image[index + 2];
            }
            index = 4 * (y * width + x);
            temp_image[index] = r;
            temp_image[index + 1] = g;
            temp_image[index + 2] = b;
            temp_image[index + 3] = image[index + 3];
        }
    }
#pragma omp parallel for shared (kernel, image, temp_image, width, height) private (y, x, i, index, r, g, b)
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            r = 0, g = 0, b = 0;
            for (i = -radius; i <= radius; i++) {
                index = 4 * (min(max(y + i, 0), int(height - 1)) * width + x);
                r += kernel[i + radius] * temp_image[index];
                g += kernel[i + radius] * temp_image[index + 1];
                b += kernel[i + radius] * temp_image[index + 2];
            }
            index = 4 * (y * width + x);
            image[index] = r;
            image[index + 1] = g;
            image[index + 2] = b;
            image[index + 3] = temp_image[index + 3];
        }
    }
}

void applyGaussianBlurUsingVectorization(vector<unsigned char>& image, int width, int height, float sigma, int radius) {
    vector<float> kernel = setKernel(radius, sigma);
    vector<unsigned char> temp_image(image.size());

    for (int i = 0; i < image.size(); i += 8) {
        int y1 = (i / 4) / width;
        int x1 = (i / 4) % width;
        int y2 = ((i + 4) / 4) / width;
        int x2 = ((i + 4) / 4) % width;
        __m256 twoPixels = _mm256_setr_ps(0, 0, 0, image[i + 3], 0, 0, 0, image[i + 7]);
        for (int j = -radius; j <= radius; ++j) {
            int index1 = 4 * (y1 * width + min(max(x1 + j, 0), width - 1));
            int index2 = 4 * (y2 * width + min(max(x2 + j, 0), width - 1));
            __m256 kernels = _mm256_set1_ps(kernel[j + radius]);
            __m256 surroundingPixels = _mm256_setr_ps(image[index1], image[index1 + 1], image[index1 + 2], 0, image[index2], image[index2 + 1], image[index2 + 2], 0);
            twoPixels = _mm256_add_ps(twoPixels, _mm256_mul_ps(kernels, surroundingPixels));
        }
        float* colors = (float*)&twoPixels;
        for (int j = 0; j < 8; ++j) {
            temp_image[i + j] = (unsigned char)colors[j];
        }
        //конечно, быстрее было бы использовать _mm_storeu, но запись в изображение осуществляется корректно только при использовании
        //формата epi8 (char), а epi8 в свою очередь не имеет функции перемножения векторов (т.е. сильно замедляет работу предыдущего цикла)
        //данная проблема бы решилась конвертацией, например, epi32 в epi8, т.к. и такая конвертация есть, и перемножаться epi32 может, 
        //но для правильной конвертации необходимо использовать __m512i epi32 (т.к. только на 512 он содержит 16 элементов, как epi8),
        //а уже 512 в свою очередь у меня не проходит в связи с техническими ограничениями ноутбука :(.
        //в общем, всецело понимаю, что костыль, но, к сожалению, при его починке появляется необходимость ставить костыль в другом месте.
    }
    for (int i = 0; i < image.size(); i += 8) {
        int y1 = (i / 4) / width;
        int x1 = (i / 4) % width;
        int y2 = ((i + 4) / 4) / width;
        int x2 = ((i + 4) / 4) % width;
        __m256 twoPixels = _mm256_setr_ps(0, 0, 0, image[i + 3], 0, 0, 0, image[i + 7]);
        for (int j = -radius; j <= radius; j++) {
            int index1 = 4 * (min(max(y1 + j, 0), int(height - 1)) * width + x1);
            int index2 = 4 * (min(max(y2 + j, 0), int(height - 1)) * width + x2);
            __m256 kernels = _mm256_set1_ps(kernel[j + radius]);
            __m256 surroundingPixels = _mm256_setr_ps(temp_image[index1], temp_image[index1 + 1], temp_image[index1 + 2], 0, temp_image[index2], temp_image[index2 + 1], temp_image[index2 + 2], 0);
            twoPixels = _mm256_add_ps(twoPixels, _mm256_mul_ps(kernels, surroundingPixels));
        }
        float* colors = (float*)&twoPixels;
        for (int j = 0; j < 8; ++j) {
            image[i + j] = (unsigned char)colors[j];
        }
    }
}

//void applyGaussianBlurUsingOCL(vector<unsigned char>& image, int width, int height, float sigma, int radius) {
//    vector<float> kernels = setKernel(radius, sigma);
//    const char* saxpy_kernel =
//        "__kernel void gaussianBlur(__global int SIZE, __global unsigned char* newImage, __global float* gaussKernel, __global int KERNELSIZE) {         \n"
//        "   int gid = get_global_id(0);                                                                                                                  \n"
//        "   int width = sqrt(SIZE/2);                                                                                                                    \n"
//        "   int height = sqrt(SIZE/2);                                                                                                                   \n"
//        "   float pixelR = 0, pixelG = 0, pixelB = 0;                                                                                                    \n"
//        "   int y = gid / width;                                                                                                                         \n"
//        "   int x = gid % width;                                                                                                                         \n"
//        "   int halfOfKernelSize = KERNELSIZE / 2;                                                                                                       \n"
//        "   for (int i = 0; i < KERNELSIZE; i++) {                                                                                                       \n"
//        "       for (int j = 0; j < KERNELSIZE; j++) {                                                                                                   \n"
//        "           int pixelX = x + i - halfOfKernelSize;                                                                                               \n"
//        "           int pixelY = y + j - halfOfKernelSize;                                                                                               \n"
//        "           if (pixelX >= 0 && pixelX < width && pixelY >= 0 && pixelY < height) {                                                               \n"
//        "               int ind = (pixelY * width + pixelX) * 3;                                                                                         \n"
//        "               pixelR += gaussKernel[i * KERNELSIZE + j] * newImage[ind];                                                                       \n"
//        "               pixelG += gaussKernel[i * KERNELSIZE + j] * newImage[ind + 1];                                                                   \n"
//        "               pixelB += gaussKernel[i * KERNELSIZE + j] * newImage[ind + 2];                                                                   \n"
//        "           }                                                                                                                                    \n"
//        "       }                                                                                                                                        \n"
//        "   }                                                                                                                                            \n"
//        "   int index = (y * width + x) * 3;                                                                                                             \n"
//        "   newImage[index] = pixelR;                                                                                                                    \n"
//        "   newImage[index + 1] = pixelG;                                                                                                                \n"
//        "   newImage[index + 2] = pixelB;                                                                                                                \n"
//        "}";
//    vector<Platform> platforms;
//    Platform::get(&platforms);
//    vector<Device> devices = vector<Device>();
//    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
//    Device device = devices[0];
//    vector<Device> contextDevices;
//    contextDevices.push_back(device);
//    Context context(contextDevices);
//    CommandQueue queue(context, device);
//    Buffer clmInputImage = Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image.size() * sizeof(unsigned char), &image);
//    Buffer clmInputKernels = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernels.size() * sizeof(float), &kernels);
//    string sourceCode(saxpy_kernel);
//    Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
//    Program program = Program(context, source);
//    program.build(contextDevices);
//    Kernel kernel(program, "saxpy_kernel");
//    int iArg = 0;
//    kernel.setArg(iArg++, image.size());
//    kernel.setArg(iArg++, clmInputImage);
//    kernel.setArg(iArg++, clmInputKernels);
//    kernel.setArg(iArg++, kernels.size());
//    queue.enqueueNDRangeKernel(kernel, NullRange, NDRange(image.size()), NDRange(64));
//    queue.finish();
//    queue.enqueueReadBuffer(clmInputImage, CL_TRUE, 0, image.size() * sizeof(unsigned int), &image);
//}

//сигнатура не совсем честная, хотя функция называется "приложить фильтр", помимо этого она ещё возвращает время наложения этого фильтра
double applyFilter(vector<unsigned char> input, unsigned int width, unsigned int height, string filterType, string filterSpecification, string updateOutput) {
    vector<unsigned char> imageCopy = input;
    auto start = chrono::high_resolution_clock::now();
    if (filterSpecification == "NC") {
        applyNegativeFilterConsequentially(imageCopy);
    }
    else if (filterSpecification == "NOMP") {
        applyNegativeFilterUsingOMP(imageCopy);
    }
    else if (filterSpecification == "NV") {
        applyNegativeFilterUsingVectorizaion(imageCopy);
    }
    /*else if (filterSpecification == "NOCL") {
        applyNegativeFilterUsingOCL(imageCopy);
    }*/
    else if (filterSpecification == "GC") {
        float sigma = 7.2;
        int radius = 15;
        applyGaussianBlurConsequentially(imageCopy, width, height, sigma, radius);
    }
    else if (filterSpecification == "GOMP") {
        float sigma = 7.2;
        int radius = 15;
        applyGaussianBlurUsingOMP(imageCopy, width, height, sigma, radius);
    }
    else if (filterSpecification == "GV") {
        float sigma = 7.2;
        int radius = 15;
        applyGaussianBlurUsingVectorization(imageCopy, width, height, sigma, radius);
    }
    /*else if (filterSpecification == "GOCL") {
        float sigma = 7.2;
        int radius = 15;
        applyGaussianBlurUsingOCL(imageCopy, width, height, sigma, radius);
    }*/
    else {
        return -1;
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);
    if (updateOutput == "YES") {
        vector<unsigned char> output;

        unsigned error;
        if (filterSpecification == "NV") {
            error = encode(output, imageCopy, width, height, LCT_RGB);
        }
        else {
            error = encode(output, imageCopy, width, height);
        }
        if (error) {
            cout << "Ошибка при загрузке изображения: " << lodepng_error_text(error) << std::endl;
            return -1;
        }
        ofstream file("outputPictures/" + filterType + "Output.png", ios::binary);
        file.write((char*)output.data(), output.size());
        file.close();
    }
    return duration.count();
}

double getAverageTime(vector<unsigned char> input, int width, int height, string filterType, string filterSpecification, int amountOfIterations) {
    //для применения векторизации с негативным фильтром легче использовать изображения в трёхканальном формате (чтобы просто игнорировать альфа-канал),
    //поэтому здесь перед обработкой изображения мы его переводим непосредственно в трёхканальный вид
    if (filterSpecification == "NV") {
        input = convertToThreeChannels(input, width, height);
    }
    double summ = applyFilter(input, width, height, filterType, filterSpecification, "YES");
    for (int i = 1; i < amountOfIterations; ++i) {
        summ += applyFilter(input, width, height, filterType, filterSpecification, "NO");
    }
    return summ / amountOfIterations;
}

int main()
{
    vector<unsigned char> image;
    cout << CHOOSE_PICTURE;
    string pictureNumber;
    cin >> pictureNumber;
    cout << CHOOSE_FILTER;
    string filterType, filterSpecification;
    cin >> filterSpecification;
    if (filterSpecification[0] == 'N') {
        filterType = "negative";
    }
    else if (filterSpecification[0] == 'G') {
        filterType = "gaussian";
    }
    cout << CHOOSE_AMOUNT_OF_ITERATIONS;
    int amountOfIterations;
    cin >> amountOfIterations;
    unsigned int width, height;
    unsigned error = decode(image, width, height, "inputPictures/" + pictureNumber + ".png");
    cout << "The application of the " + filterType + " filter lasted " << getAverageTime(image, width, height, filterType, filterSpecification, amountOfIterations)
        << " seconds on average of " << amountOfIterations << " iterations.";
}