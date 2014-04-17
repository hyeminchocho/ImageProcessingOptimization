#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{


    //44-45 GFLOPS --- HAVE NOT HANDLED WEIRD NUMBERS YET ---

    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    int pad_size_X = data_size_X + 2;
    int pad_size_Y = data_size_Y + 2;
    int i, j, x, y, k;
    int toppad = kern_cent_Y*(kern_cent_X*2+data_size_X)+kern_cent_X;
    int xRemainder = data_size_X % 4;

    // Inverting the kernel
    float flipKernel[KERNX * KERNY];
    for (k = 0; k < (KERNX * KERNY); k++){
	flipKernel[k] = kernel[(KERNX * KERNY) - k - 1];
    }


    __m128 kern1 = _mm_load1_ps(flipKernel);
    __m128 kern2 = _mm_load1_ps(flipKernel+1);
    __m128 kern3 = _mm_load1_ps(flipKernel+2);
    __m128 kern4 = _mm_load1_ps(flipKernel+3);
    __m128 kern5 = _mm_load1_ps(flipKernel+4);
    __m128 kern6 = _mm_load1_ps(flipKernel+5);
    __m128 kern7 = _mm_load1_ps(flipKernel+6);
    __m128 kern8 = _mm_load1_ps(flipKernel+7);
    __m128 kern9 = _mm_load1_ps(flipKernel+8);
    __m128 resVect;
 
    int padlen = (data_size_X+kern_cent_X*2) * (data_size_Y+kern_cent_Y*2);
    float *padin = malloc(pad_size_X*pad_size_Y*sizeof(float));
    //assign 0.0 to each index of the padded array
    #pragma omp parallel num_threads(8)
    {
    #pragma omp for
    for (i = 0; i < (pad_size_X * pad_size_Y)/4; i++) {
        _mm_storeu_ps((padin + i*4), _mm_setzero_ps());
    }
    for (i = (pad_size_X * pad_size_Y)/4 * 4; i < pad_size_X * pad_size_Y; i++){
            padin[i] = 0.0;
    }

    } // closes pragma parallel

// Copy elements from "in" the "padin"
    #pragma omp parallel firstprivate(x) num_threads(8)
    {
    #pragma omp for
    for(y = 0; y < data_size_Y; y++){
	   for(x = 0; x < data_size_X/4; x++){
	        _mm_storeu_ps(padin+toppad+y*pad_size_X+x*4, _mm_loadu_ps(in+data_size_X*y+x*4));
	   }
	   for(x = (data_size_X/4) * 4; x < data_size_X; x++){
		padin[pad_size_X+1+y*pad_size_X+x] = in[data_size_X * y + x];
	   }
    }

    } // closes pragma parallel 
    
    // main convolution loop
	#pragma omp parallel firstprivate(x, resVect, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, kern9) num_threads(8)
	{
	#pragma omp for 
	for(y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
	   for(x = 0; x < data_size_X/4; x++){ // the x coordinate of the output location we're focusing on


			resVect = _mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + (y-1) * pad_size_X + x * 4 -1), kern1);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + (y-1) * pad_size_X + x * 4), kern2), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + (y-1) * pad_size_X + x * 4 +1), kern3), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + y * pad_size_X + x * 4 -1), kern4), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + y * pad_size_X + x * 4), kern5), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + y * pad_size_X + x * 4+1), kern6), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + (y+1) * pad_size_X + x * 4 -1), kern7), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + (y+1) * pad_size_X + x * 4), kern8), resVect);

			resVect = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(padin+(pad_size_X + 1) + (y+1) * pad_size_X + x * 4+1), kern9), resVect);
			
            _mm_storeu_ps(out + y * data_size_X + x * 4, resVect);

		}
     for (x = (data_size_X/4) * 4; x < data_size_X; x++) {
            out[x+y*data_size_X] += flipKernel[0] * padin[pad_size_X+1+(x-1) + (y-1)*pad_size_X] + 
                                    flipKernel[1] * padin[pad_size_X+1+x+ (y-1)*pad_size_X] +
                                    flipKernel[2] * padin[pad_size_X+1+(x+1) + (y-1)*pad_size_X] +
                                    flipKernel[3] * padin[pad_size_X+1+(x-1) +y*pad_size_X] +
                                    flipKernel[4] * padin[pad_size_X+1+x+ y*pad_size_X] +
                                    flipKernel[5] * padin[pad_size_X+1+(x+1) + y*pad_size_X] +
                                    flipKernel[6] * padin[pad_size_X+1+(x-1) + (y+1)*pad_size_X] +
                                    flipKernel[7] * padin[pad_size_X+1+x + (y+1)*pad_size_X] +
                                    flipKernel[8] * padin[pad_size_X+1+(x+1) + (y+1)*pad_size_X];
        }
	}

    } // closes pragma parallel loop


	free(padin);

	return 1;
}
