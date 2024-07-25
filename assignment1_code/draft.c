for (size_t thread_idx=0; thread_idx < num_threads; thread_idx++){
		ret_val_thread = pthread_join(tinfo_arr[thread_idx].curr_thread, NULL);
		if (ret_val_thread != 0)
                   handle_error_en(ret_val_thread, "pthread_join");

               printf("Joined with thread %d", tinfo_arr[thread_idx].thread_ID);
	}

	/* validate command line parameter */
	if (argc < 1 || !(strcmp(argv[1],"16") || strcmp(argv[1],"32") || strcmp(argv[1],"64"))) {
	  printf("Usage: %s {16|32|64}\n",argv[0]);
	  fflush(stdout);
	  exit(-1);
	}