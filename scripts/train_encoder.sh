# CUDA_VISIBLE_DEVICES=4,5 \
# torchrun --standalone --nproc_per_node=2 \
# -m ecg_bench.train_encoder \
# --ecg_signal \
# --encoder=mtae \
# --data=ecg-qa-mimic-iv-ecg-250-1250 \
# --batch_size=16 \
# --optimizer=adamw \
# --lr=2e-4 \
# --weight_decay=1e-5 \
# --distributed \
# --dev


for encoder in mtae mlae merl st_mem; do
    python -m ecg_bench.train_encoder \
    --ecg_signal \
    --encoder=$encoder \
    --data=ecg-qa-mimic-iv-ecg-250-1250 \
    --batch_size=256 \
    --optimizer=adamw \
    --lr=2e-4 \
    --weight_decay=1e-5 \
    --device=cuda:0 \
    --wandb \
    --epochs=50
done