@echo off
cd /d "%~dp0"

pip install numpy scipy matplotlib pyvista numba pytest filelock tqdm
pip install -e .

git config merge.records-npy.name "Keep lowest-energy .npy cluster"
git config merge.records-npy.driver "python git_merge_records.py npy %%O %%A %%B %%P"
git config merge.records-log.name "Keep lowest-energy lines in _by_N.log"
git config merge.records-log.driver "python git_merge_records.py log %%O %%A %%B %%P"

pause