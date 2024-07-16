# 08-02:
# test:
python testV2.py
# evaluate:
cd ./evaluation
python evaluate_iou.py

# 单机四卡训练
python -m torch.distributed.launch --nproc_per_node=4 trainV2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 trainV2.py
TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 trainV3_step2.py
torchrun --nproc_per_node=4 trainV2.py
torchrun --nproc_per_node=4 trainOneStepV2.py
torchrun --nproc_per_node=4 trainOL.py
