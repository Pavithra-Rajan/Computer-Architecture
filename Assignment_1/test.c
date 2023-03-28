#include <stdio.h>
#include <time.h>
#define CACHE_MIN (1024) /* smallest cache */
#define CACHE_MAX (4*1024*1024)
#define SAMPLE 10
// array to be used for cache analysis
int x[CACHE_MAX];

//  converts the number of clocks taken to seconds
double get_seconds() 
{ 
    /* routine to read time */
    clock_t t;
    t = clock();
    return (double) t/ (double) CLOCKS_PER_SEC;
}

/*
x - the array to be accessed
Stride - the number of elements to skip between each access
Limit - the number of elements to access in the array
Steps - the number of times the loop is executed
Csize - the size of the cache
tsteps - the number of times the empty loop is executed

*/


int main(void)
{
    FILE *outptr; 

    int register i, index, stride, limit, temp;
    int steps, tsteps, csize;
    double sec0, sec; /* timing variables. */

    outptr = fopen("output.txt","w");

    for (csize = CACHE_MIN; csize <= CACHE_MAX; csize = csize * 2)
    {
        for (stride = 1; stride <= csize/2; stride = stride * 2)
        {
            sec = 0; /* initialize timer */
            
            limit = csize - stride + 1; /* cache size this loop */
            steps = 0;
            // this do while loop is used to collect 1 second of data
            do 
            { /* repeat until collect 1 second */
                sec0 = get_seconds(); /* start timer */
                for (i = SAMPLE * stride; i != 0; i = i - 1) /* larger sample */
                {
                    for (index = 0; index < limit; index = index + stride)
                    {
                        x[index] = x[index] + 1; /* cache access */
                    }
                }
                steps = steps + 1; /* count while loop iterations */
                sec += (get_seconds() - sec0); /* end timer */
            } while (sec < 1.0); /* until collect 1 second */

            /* repeat empty loop to subtract loop overhead */
            tsteps = 0; /* used to match no. while iterations */
            do { /* repeat until same no. of iterations as above */
                sec0 = get_seconds(); /* start timer */
                for (i = SAMPLE * stride; i != 0; i = i - 1) /* larger sample */    
                {
                    for (index = 0; index < limit; index = index + stride)
                    {
                    temp = temp + index; /* dummy code */
                    }
                }
                tsteps = tsteps + 1; /* count while loop iterations */
                sec -= (get_seconds() - sec0); /* - overhead */
            } while (tsteps < steps); /* until = no. iterations */

            // at this point, sec contains the time taken to  perform data access in steps iterations of the while loop

            /*
                read+write = (steps*SAMPLE*stride*((limit-1)/stride+1))
            */
            // csize, stride
            double readWrite =  (double) sec*1e9/(steps*SAMPLE*stride*((limit-1)/stride+1));
            printf("Size:%7d Stride:%7d read+write:%f ns\n", csize , stride, readWrite);
            fprintf(outptr," %7d %7d %f \n", csize , stride , readWrite);
        } /* end of both outer for loops */
    }
    fclose(outptr); 
}
