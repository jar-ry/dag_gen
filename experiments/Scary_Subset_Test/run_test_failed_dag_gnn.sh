# Mix Mech Select Bias 2
python ./../run_test_vm.py dense_med_linear_no_issue_2 linear_mechanism DAG_GNN
python ./../run_test_vm.py dense_med_mixed_selection_bias_2 mix_mechanism DAG_GNN
python ./../run_test_vm.py dense_small_mixed_confounder_2 mix_mechanism DAG_GNN
python ./../run_test_vm.py small_mixed_unfaithful_3 mix_mechanism DAG_GNN
python ./../run_test_vm.py x_large_mixed_no_issue_1 mix_mechanism DAG_GNN

