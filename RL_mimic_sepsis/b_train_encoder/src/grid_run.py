import subprocess

param_file = "F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/2_train_encoder/logs/AIS_grid.txt"
python_path ="D:/Software/anaconda3/envs/rl4h_rep_new/python.exe"
script_path = "F:/time_step/OfflineRL_FactoredActions/RL_mimic_sepsis/2_train_encoder/opt.py"

with open(param_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue 
        print("Running with:", line)

        cmd = [python_path, script_path] + line.split()
        subprocess.run(cmd)
