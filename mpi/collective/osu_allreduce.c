#define BENCHMARK "OSU MPI%s Allreduce Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include "osu_coll.h"

int main(int argc, char *argv[])
{
    int i, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    float *sendbuf, *recvbuf;
    int po_ret;
    size_t bufsize;
    MPI_Comm sub_comm;
    int sub_rank, sub_numprocs, size_loop = 0, k = 0;
    double *avg_time_stats, *min_time_stats, *max_time_stats;

    set_header(HEADER);
    set_benchmark_name("osu_allreduce");
    enable_accel_support();
    po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    switch (po_ret) {
        case po_bad_usage:
            print_bad_usage_message(rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        case po_help_message:
            print_help_message(rank);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_version_message:
            print_version_message(rank);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (options.max_message_size > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit;
    }

    options.min_message_size /= sizeof(float);
    if (options.min_message_size < DEFAULT_MIN_MESSAGE_SIZE) {
        options.min_message_size = DEFAULT_MIN_MESSAGE_SIZE;
    }

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(sendbuf, options.accel, 1, bufsize);

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&recvbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);

    /* create sub communicator */
    if (options.num_comms == 1) {
        sub_comm = MPI_COMM_WORLD;
    }
    else {
        MPI_Comm_split(MPI_COMM_WORLD, rank % options.num_comms, rank, &sub_comm);
    }

    MPI_Comm_rank(sub_comm, &sub_rank);
    MPI_Comm_size(sub_comm, &sub_numprocs);

    if (sub_rank == 0) {
        /* count how many size loop */
        for(size=options.min_message_size; size*sizeof(float) <= options.max_message_size; size *= 2) {
            size_loop++;
        }

        avg_time_stats = (double *)malloc(sizeof(double) * size_loop);
        min_time_stats = (double *)malloc(sizeof(double) * size_loop);
        max_time_stats = (double *)malloc(sizeof(double) * size_loop);
    }

    for(size=options.min_message_size; size*sizeof(float) <= options.max_message_size; size *= 2) {

        if(size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        timer=0.0;
        for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();
            MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, sub_comm );
            t_stop=MPI_Wtime();
            if(i>=options.skip){

            timer+=t_stop-t_start;
            }
            MPI_Barrier(sub_comm);
        }
        latency = (double)(timer * 1e6) / options.iterations;

        if (sub_rank == 0) {
            MPI_Reduce(&latency, &min_time_stats[k], 1, MPI_DOUBLE, MPI_MIN, 0,
                       sub_comm);
            MPI_Reduce(&latency, &max_time_stats[k], 1, MPI_DOUBLE, MPI_MAX, 0,
                       sub_comm);
            MPI_Reduce(&latency, &avg_time_stats[k], 1, MPI_DOUBLE, MPI_SUM, 0,
                       sub_comm);
            avg_time_stats[k] = avg_time_stats[k]/sub_numprocs;

            k++;
        }
        else {
            MPI_Reduce(&latency, NULL, 1, MPI_DOUBLE, MPI_MIN, 0,
                       sub_comm);
            MPI_Reduce(&latency, NULL, 1, MPI_DOUBLE, MPI_MAX, 0,
                       sub_comm);
            MPI_Reduce(&latency, NULL, 1, MPI_DOUBLE, MPI_SUM, 0,
                       sub_comm);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);

    for (i = 0; i < numprocs; i++) {
        if (i == rank && sub_rank == 0) {
            print_preamble(sub_rank);
            printf("### Group ID: %d ###\n", rank % options.num_comms);

            k = 0;
            for (size=options.min_message_size; size*sizeof(float) <= options.max_message_size; size *= 2) {
                print_stats(sub_rank, size * sizeof(float),
                            avg_time_stats[k], min_time_stats[k], max_time_stats[k]);
                k++;
            }
            printf("\n");
        }
    }

    if (sub_rank == 0) {
        free(avg_time_stats);
        free(min_time_stats);
        free(max_time_stats);
    }

    /* free sub communicator */
    if (options.num_comms != 1) {
        MPI_Comm_free(&sub_comm);
    }

    MPI_Finalize();

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}
