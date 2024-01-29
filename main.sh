for i in $(seq 0 1)
do
    echo "Running $i sample..."
    poetry run python main.py --sample-number $i --model MBZUAI/LaMini-GPT-124M
done