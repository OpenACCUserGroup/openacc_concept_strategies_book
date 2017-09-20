#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define WIDTH      1000
#define HEIGHT     1000

#define TEMP_TOLERANCE 0.01

double Temperature[HEIGHT+2][WIDTH+2];
double Temperature_previous[HEIGHT+2][WIDTH+2];

void initialize();
void track_progress(int iter);


int main(int argc, char *argv[]) {

    int i, j;
    int iteration=1;
    double worst_dt=100;
    struct timeval start_time, stop_time, elapsed_time;

    gettimeofday(&start_time,NULL);

    initialize();

    while ( worst_dt > TEMP_TOLERANCE ) {

        #pragma omp parallel for private(i,j)
        for(i = 1; i <= HEIGHT; i++) {
            for(j = 1; j <= WIDTH; j++) {
                Temperature[i][j] = 0.25 * (Temperature_previous[i+1][j] + Temperature_previous[i-1][j] +
                                            Temperature_previous[i][j+1] + Temperature_previous[i][j-1]);
            }
        }
        
        worst_dt = 0.0;

        #pragma omp parallel for reduction(max:worst_dt) private(i,j)
        for(i = 1; i <= HEIGHT; i++){
            for(j = 1; j <= WIDTH; j++){
	      worst_dt = fmax( fabs(Temperature[i][j]-Temperature_previous[i][j]), worst_dt);
	      Temperature_previous[i][j] = Temperature[i][j];
            }
        }

        if((iteration % 100) == 0) {
 	    track_progress(iteration);
        }

	iteration++;
    }

    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time);

    printf("\nMax error at iteration %d was %f\n", iteration-1, worst_dt);
    printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);

}


void initialize(){

    int i,j;

    for(i = 0; i <= HEIGHT+1; i++){
        for (j = 0; j <= WIDTH+1; j++){
            Temperature_previous[i][j] = 0.0;
        }
    }

    for(i = 0; i <= HEIGHT+1; i++) {
        Temperature_previous[i][0] = 0.0;
        Temperature_previous[i][WIDTH+1] = (100.0/HEIGHT)*i;
    }
    
    for(j = 0; j <= WIDTH+1; j++) {
        Temperature_previous[0][j] = 0.0;
        Temperature_previous[HEIGHT+1][j] = (100.0/WIDTH)*j;
    }
}


void track_progress(int iteration) {

    int i;

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = HEIGHT-5; i <= HEIGHT; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, Temperature[i][i]);
    }
    printf("\n");
}
