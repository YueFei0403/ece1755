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
	float *tx_point_x;
	float *tx_point_y;
	float *tx_point_z;
	float *rx_point_x;
	float *rx_point_y;
	float *rx_point_iter;
	float *rx_data;
	float *image;  // Pointer to full image (accumulated so far)
	int sls_theta;
	int sls_phi;
} thread_info;


void *compute_txrx_dist( thread_info *curr_thread_info )
{
	/* Variables for image space points */
	/* --------------------------- COMPUTATION ------------------------------ */
	/* First compute transmit distance */
	int point;
	int it_rx; // Iterator for recieve transducer
	int it_r; // Iterator for r
	int it_t; // Iterator for theta
	int it_p; // Iterator for phi
	float dist; // Full distance

	/* Variables for distance calculation and index conversion */
	float x_comp; // Intermediate value for dist calc
	float y_comp; // Intermediate value for dist calc
	float z_comp; // Intermediate value for dist calc
	int index;    // Index into transducer data
	int offset = 0; // Offset into rx_data
	float *image_pos;
	point = curr_thread_info->curr_tx_point;
	// Iterate over entire image space
	/* Now compute reflected distance, find index values, add to image */
	image_pos = curr_thread_info->image;
	float curr_dist;
	for (it_t = 0; it_t < curr_thread_info->num_iter_tx; it_t++) {
		for (it_p = 0; it_p < curr_thread_info->sls_phi; it_p++) {
			for (it_r = 0; it_r < PTS_R; it_r++) {

				x_comp = TX_X - curr_thread_info->tx_point_x[point];
				x_comp = x_comp * x_comp;
				y_comp = TX_Y - curr_thread_info->tx_point_y[point];
				y_comp = y_comp * y_comp;
				z_comp = TX_Z - curr_thread_info->tx_point_z[point];
				z_comp = z_comp * z_comp;

				curr_dist = (float)sqrt(x_comp + y_comp + z_comp);

				offset = 0;
				for (it_rx = 0; it_rx < TRANS_X * TRANS_Y; it_rx++) {
					// printf("-- THREAD_IDX %d  it_rx= %d\n",curr_thread_info->thread_ID,it_rx);
					x_comp = curr_thread_info->rx_point_x[it_rx] - curr_thread_info->tx_point_x[point];
					x_comp = x_comp * x_comp;
					y_comp = curr_thread_info->rx_point_y[it_rx] - curr_thread_info->tx_point_y[point];
					y_comp = y_comp * y_comp;
					z_comp = RX_Z - curr_thread_info->tx_point_z[point];
					z_comp = z_comp * z_comp;

					dist = curr_dist + (float)sqrt(x_comp + y_comp + z_comp);
					index = (int)(dist/IDX_CONST + FILTER_DELAY + 0.5);
					float tmp = curr_thread_info->rx_data[index+offset];
					*image_pos += tmp;
					offset += DATA_LEN;
				}
				point++;
				image_pos++;
			}
		}
	}
}


int main (int argc, char **argv) {

    int num_threads = atoi(argv[1]);
    int size = atoi(argv[2]);

	float *point_x; // Point x position
	float *point_y; // Point y position
	float *point_z; // Point z position

	float *rx_x; // Receive transducer x position
	float *rx_y; // Receive transducer y position

	int sls_t = size; // Number of scanlines in theta
	int sls_p = size; // Number of scanlines in phi

	float *image;  // Pointer to full image (accumulated so far)
	float *rx_data; // Pointer to pre-processed receive channel data

	float *dist_tx; // Transmit distance (ie first leg only)
	FILE* input;
	FILE* output;

	/* Allocate space for data */
	rx_x = (float*) malloc(TRANS_X * TRANS_Y * sizeof(float));
	if (rx_x == NULL) fprintf(stderr, "Bad malloc on rx_x\n");
	rx_y = (float*) malloc(TRANS_X * TRANS_Y * sizeof(float));
	if (rx_y == NULL) fprintf(stderr, "Bad malloc on rx_y\n");
	rx_data = (float*) malloc(DATA_LEN * TRANS_X * TRANS_Y * sizeof(float));
	if (rx_data == NULL) fprintf(stderr, "Bad malloc on rx_data\n");

	point_x = (float *) malloc(PTS_R * sls_t * sls_p * sizeof(float));
	if (point_x == NULL) fprintf(stderr, "Bad malloc on point_x\n");
	point_y = (float *) malloc(PTS_R * sls_t * sls_p * sizeof(float));
	if (point_y == NULL) fprintf(stderr, "Bad malloc on point_y\n");
	point_z = (float *) malloc(PTS_R * sls_t * sls_p * sizeof(float));
	if (point_z == NULL) fprintf(stderr, "Bad malloc on point_z\n");

	dist_tx = (float*) malloc(PTS_R * sls_t * sls_p * sizeof(float));
	if (dist_tx == NULL) fprintf(stderr, "Bad malloc on dist_tx\n");

	image = (float *) malloc(PTS_R * sls_t * sls_p * sizeof(float));
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
		thread_info_arr[thread_idx].num_iter_tx = num_iter_tx;
		thread_info_arr[thread_idx].curr_tx_point = num_iter_tx * thread_idx * PTS_R * sls_p; // Starting index into image space
		thread_info_arr[thread_idx].tx_point_x = point_x;
		thread_info_arr[thread_idx].tx_point_y = point_y;
		thread_info_arr[thread_idx].tx_point_z = point_z;
		thread_info_arr[thread_idx].rx_point_x = rx_x;
		thread_info_arr[thread_idx].rx_point_y = rx_y;
		thread_info_arr[thread_idx].rx_data = rx_data;
		thread_info_arr[thread_idx].image = image + num_iter_tx * thread_idx * PTS_R * sls_p;
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
	free(rx_x);
	free(rx_y);
	free(rx_data);
	free(point_x);
	free(point_y);
	free(point_z);
	free(dist_tx);
	free(image);
	return 0;
}

