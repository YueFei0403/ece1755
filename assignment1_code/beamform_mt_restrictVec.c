// 3D Ultrasound beamforming baseline code for EECS 570 
// Created by: Richard Sampson, Amlan Nayak, Thomas F. Wenisch
// Revision 1.0 - 11/15/16
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <pthread.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <stdalign.h>

#ifdef __SSE2__
  #include <emmintrin.h>
#else
  #warning SSE2 support is not available. Code will not compile
#endif

#define handle_error_en(en, msg) \
               do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

/*=========  Global Constant Variables for transducer geometry ==========*/
int TRANS_X = 32; // Transducers in x dim
int TRANS_Y = 32; // Transducers in y dim

float RX_Z = 0; // Receive transducer z position

int DATA_LEN = 12308; // Number for pre-processed data values per channel


float TX_X = 0; // Transmit transducer x position
float TX_Y = 0; // Transmit transducer y position
float TX_Z = -0.001; // Transmit transducer z position

int PTS_R = 1560; // Radial points along scanline

const float IDX_CONST = 0.000009625; // Speed of sound and sampling rate, converts dist to index
const int FILTER_DELAY = 140; // Constant added to index to account filter delay (off by 1 from MATLAB)
//=======================================================================

typedef struct thread_info {
    pthread_t curr_thread;
    int thread_ID;
	int num_iter_tx;
	int curr_tx_point; // Starting index into image space
	float * restrict tx_point_x;
	float * restrict tx_point_y;
	float * restrict tx_point_z;
	float * restrict rx_point_x;
	float * restrict rx_point_y;
	float * restrict rx_point_iter;
	float * restrict rx_data;
	float * image;  // Pointer to full image (accumulated so far)
	int sls_theta;
	int sls_phi;
} thread_info;


void *compute_txrx_dist( thread_info *curr_thread_info )
{
	/* Variables for image space points */
	/* --------------------------- COMPUTATION ------------------------------ */
	/* First compute transmit distance */
	int point_start;
	point_start = curr_thread_info->curr_tx_point / 4;
	int offset; // Offset into rx_data
	int index;    // Index into transducer data

	// Iterate over entire image space
	/* Now compute reflected distance, find index values, add to image */
	__m128 curr_txdist_simd;

	// Casting pointer into vector-type
	__m128 TX_X_simd = _mm_set1_ps(TX_X);
	__m128 TX_Y_simd = _mm_set1_ps(TX_Y);
	__m128 TX_Z_simd = _mm_set1_ps(TX_Z);
	
	__m128* restrict tx_point_x_simd = (__m128 *) curr_thread_info->tx_point_x;
	__m128* restrict tx_point_y_simd = (__m128 *) curr_thread_info->tx_point_y;
	__m128* restrict tx_point_z_simd = (__m128 *) curr_thread_info->tx_point_z;


	__m128 x_comp_simd, y_comp_simd,z_comp_simd;
	__m128 tx_sum_simd;
	float x_comp,y_comp,z_comp;
	int num_txloop = (curr_thread_info->num_iter_tx * curr_thread_info->sls_phi * PTS_R)/4;
	int num_rxloop = TRANS_X * TRANS_Y;
	int idx_tx,idx_rx,idx_float,it_rx;

	float* image_rx = curr_thread_info->image;
	float* image_tx = curr_thread_info->image;
	float txrx_dist,tmp_sum;
	for (idx_tx = 0; idx_tx < num_txloop; idx_tx++) {
		x_comp_simd = _mm_sub_ps(TX_X_simd,tx_point_x_simd[point_start]);
		x_comp_simd = _mm_mul_ps(x_comp_simd,x_comp_simd);

		y_comp_simd = _mm_sub_ps(TX_Y_simd,tx_point_y_simd[point_start]);
		y_comp_simd = _mm_mul_ps(y_comp_simd,y_comp_simd);

		z_comp_simd = _mm_sub_ps(TX_Z_simd,tx_point_z_simd[point_start]);
		z_comp_simd = _mm_mul_ps(z_comp_simd,z_comp_simd);

		tx_sum_simd = _mm_add_ps(x_comp_simd,y_comp_simd);
		tx_sum_simd = _mm_add_ps(tx_sum_simd,z_comp_simd);
		curr_txdist_simd = _mm_sqrt_ps(tx_sum_simd);

		
		float* restrict tx_x = _mm_malloc(4*sizeof(float),16);
		float* restrict tx_y = _mm_malloc(4*sizeof(float),16);
		float* restrict tx_z = _mm_malloc(4*sizeof(float),16);
		float* restrict tx_dist = _mm_malloc(4*sizeof(float),16);
		_mm_store_ps(tx_x,tx_point_x_simd[point_start]);
		_mm_store_ps(tx_x,tx_point_x_simd[point_start]);
		_mm_store_ps(tx_y,tx_point_y_simd[point_start]);
		_mm_store_ps(tx_z,tx_point_z_simd[point_start]);
		_mm_store_ps(tx_dist,curr_txdist_simd);
		image_rx = image_tx;
		for (idx_float=0; idx_float<4; idx_float++) {
			offset = 0; tmp_sum = 0;
			for (it_rx = 0; it_rx < TRANS_X * TRANS_Y; it_rx++) {
				x_comp = curr_thread_info->rx_point_x[it_rx] - tx_x[idx_float];
				x_comp = x_comp * x_comp;
				y_comp = curr_thread_info->rx_point_y[it_rx] - tx_y[idx_float];
				y_comp = y_comp * y_comp;
				z_comp = RX_Z - tx_z[idx_float];
				z_comp = z_comp * z_comp;

				txrx_dist = tx_dist[idx_float] + (float)sqrt(x_comp + y_comp + z_comp);
				index = (int)(txrx_dist/IDX_CONST + FILTER_DELAY + 0.5);
				tmp_sum += curr_thread_info->rx_data[index+offset];
				offset += DATA_LEN;
			}
			*image_rx = tmp_sum;
			image_rx++;
		}
		image_tx += 4;
		point_start++;
	}
}


int main (int argc, char **argv) {

    int num_threads = atoi(argv[1]);
    int size = atoi(argv[2]);

	float* restrict point_x; // Point x position
	float* restrict point_y; // Point y position
	float* restrict point_z; // Point z position

	float* restrict rx_x; // Receive transducer x position
	float* restrict rx_y; // Receive transducer y position

	int sls_t = size; // Number of scanlines in theta
	int sls_p = size; // Number of scanlines in phi

	float* image;  // Pointer to full image (accumulated so far)
	float* restrict rx_data; // Pointer to pre-processed receive channel data

	float* restrict dist_tx; // Transmit distance (ie first leg only)
	FILE* input;
	FILE* output;

	/* Allocate space for data */
	rx_x = (float*) _mm_malloc(TRANS_X * TRANS_Y * sizeof(float),16);
	if (rx_x == NULL) fprintf(stderr, "Bad malloc on rx_x\n");
	rx_y = (float*) _mm_malloc(TRANS_X * TRANS_Y * sizeof(float),16);
	if (rx_y == NULL) fprintf(stderr, "Bad malloc on rx_y\n");
	rx_data = (float*) _mm_malloc(DATA_LEN * TRANS_X * TRANS_Y * sizeof(float),16);
	if (rx_data == NULL) fprintf(stderr, "Bad malloc on rx_data\n");

	point_x = (float*) _mm_malloc(PTS_R * sls_t * sls_p * sizeof(float),16);
	if (point_x == NULL) fprintf(stderr, "Bad malloc on point_x\n");
	point_y = (float *) _mm_malloc(PTS_R * sls_t * sls_p * sizeof(float),16);;
	if (point_y == NULL) fprintf(stderr, "Bad malloc on point_y\n");
	point_z = (float*) _mm_malloc(PTS_R * sls_t * sls_p * sizeof(float),16);
	if (point_z == NULL) fprintf(stderr, "Bad malloc on point_z\n");

	image = (float*) _mm_malloc(PTS_R * sls_t * sls_p * sizeof(float),16);
	if (image == NULL) fprintf(stderr, "Bad malloc on image\n");
	memset(image, 0, PTS_R * sls_t * sls_p * sizeof(float));

	char buff[128];
        #ifdef __MIC__
	  sprintf(buff, "/beamforming_input_%s.bin", argv[2]);
        #else // !__MIC__
	  sprintf(buff, "/cad2/ece1755s/assignment1_data/beamforming_input_%s.bin", argv[2]);
        #endif

        input = fopen(buff,"rb");
	if (!input) {
	  printf("Unable to open input file %s.\n", buff);
	  fflush(stdout);
	  exit(-1);
	}

	/* Load data from binary */
	fread(rx_x, sizeof(float), TRANS_X * TRANS_Y, input); 
	fread(rx_y, sizeof(float), TRANS_X * TRANS_Y, input); 

	fread(point_x, sizeof(float), PTS_R * sls_t * sls_p, input); 
	fread(point_y, sizeof(float), PTS_R * sls_t * sls_p, input); 
	fread(point_z, sizeof(float), PTS_R * sls_t * sls_p, input); 

	fread(rx_data, sizeof(float), DATA_LEN * TRANS_X * TRANS_Y, input); 
        fclose(input);

	printf("Beginning computation\n");
	fflush(stdout);

	/* get start timestamp */
 	struct timeval tv;
    	gettimeofday(&tv,NULL);
    	uint64_t start = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
 
    /* Thread Creation */
    thread_info *thread_info_arr = calloc(num_threads, sizeof(thread_info));
	int ret_val_thread;
	int num_iter_tx = sls_t / num_threads;
	for (size_t thread_idx=0; thread_idx < num_threads; thread_idx++) {
		/* Create independent threads each of which will execute function */
		thread_info_arr[thread_idx].thread_ID = thread_idx;
		thread_info_arr[thread_idx].num_iter_tx = num_iter_tx; // 4*15*1560*64
		thread_info_arr[thread_idx].curr_tx_point = num_iter_tx * thread_idx * PTS_R * sls_p; // Starting index into image space
		thread_info_arr[thread_idx].tx_point_x = point_x;
		thread_info_arr[thread_idx].tx_point_y = point_y;
		thread_info_arr[thread_idx].tx_point_z = point_z;
		thread_info_arr[thread_idx].rx_point_x = rx_x;
		thread_info_arr[thread_idx].rx_point_y = rx_y;
		thread_info_arr[thread_idx].rx_data = rx_data;
		thread_info_arr[thread_idx].image = image + (num_iter_tx * thread_idx * PTS_R * sls_p);
		thread_info_arr[thread_idx].sls_theta = sls_t;
		thread_info_arr[thread_idx].sls_phi = sls_p;

		ret_val_thread = pthread_create(&thread_info_arr[thread_idx].curr_thread, NULL,
							compute_txrx_dist, &thread_info_arr[thread_idx]);
		if (ret_val_thread != 0){
			handle_error_en(ret_val_thread, "pthread_create");
		}
	}
	for (size_t thread_idx=0; thread_idx < num_threads; thread_idx++){
		ret_val_thread = pthread_join(thread_info_arr[thread_idx].curr_thread, NULL);
		if (ret_val_thread != 0)
                   handle_error_en(ret_val_thread, "pthread_join");
	}


	free(thread_info_arr);

	
		/* --------------------------------------------------------------------- */

	/* get elapsed time */
    	gettimeofday(&tv,NULL);
    	uint64_t end = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
    	uint64_t elapsed = end - start;

	printf("@@@ Elapsed time (usec): %lu\n", elapsed);
	printf("Processing complete.  Preparing output.\n");
	fflush(stdout);

	/* Write result to file */
	char* out_filename;
        #ifdef __MIC__
	  out_filename = "/home/micuser/beamforming_output.bin";
        #else // !__MIC__
	  out_filename = "beamforming_output.bin";
        #endif
        output = fopen(out_filename,"wb");
	fwrite(image, sizeof(float), PTS_R * sls_t * sls_p, output); 
	fclose(output);

	printf("Output complete.\n");
	fflush(stdout);

	/* Cleanup */
	_mm_free(rx_x);
	_mm_free(rx_y);
	_mm_free(rx_data);
	_mm_free(point_x);
	_mm_free(point_y);
	_mm_free(point_z);
	_mm_free(image);
	return 0;
}

