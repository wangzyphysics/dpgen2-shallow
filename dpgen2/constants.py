train_index_pattern = "%04d"
train_task_pattern = "task." + train_index_pattern
train_script_name = "input.json"
train_log_name = "train.log"
model_name_pattern = "model.%03d.pb"
model_name_match_pattern = r"model\.[0-9]{3,}\.pb"
lmp_index_pattern = "%06d"
lmp_task_pattern = "task." + lmp_index_pattern
lmp_conf_name = "conf.lmp"
lmp_input_name = "in.lammps"
plm_input_name = "input.plumed"
plm_output_name = "output.plumed"
lmp_traj_name = "traj.dump"
lmp_log_name = "log.lammps"
lmp_model_devi_name = "model_devi.out"
fp_index_pattern = "%06d"
fp_task_pattern = "task." + fp_index_pattern
fp_default_log_name = "fp.log"
fp_default_out_data_name = "data"
calypso_log_name = "caly.log"
calypso_input_file = "input.dat"
calypso_index_pattern = "%06d"
calypso_task_pattern = "caly_task." + calypso_index_pattern
calypso_run_opt_file = "calypso_run_opt.py"
calypso_check_opt_file = "calypso_check_opt.py"
calypso_opt_log_name = "opt.log"
calypso_traj_log_name = "traj.traj"

default_image = "dptechnology/dpgen2:latest"
default_host = "127.0.0.1:2746"
