export  OMP_NUM_THREADS=8
python -u -m torch.distributed.launch --nproc_per_node 4 guorongxiao/EEG_Video_Fusion/trainmm.py