mkdir pile_uncopyrighted && cd pile_uncopyrighted
for i in $(seq -w 0 29); do
(
    wget https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/${i}.jsonl.zst
    unzstd ${i}.jsonl.zst
    python ../process.py ${i}
) &
done
wait
