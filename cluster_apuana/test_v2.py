import run_cluster_final_v2
run_cluster_final_v2.DATASETS = [{'tid': 1464, 'name': 'blood-transfusion-service-center', 'regime': 'small'}]
run_cluster_final_v2.N_TRIALS = 1
run_cluster_final_v2.AG_EXTREME_TIME_LIMIT = 30
run_cluster_final_v2.AG_DEFAULT_TIME_LIMIT = 30
run_cluster_final_v2.RESULTS_FILE = "test_results.csv"
run_cluster_final_v2.main()
