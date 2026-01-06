# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    eval_corpus_path = '../birth_dev.tsv'
    # 读取文件以确定样本数量
    with open(eval_corpus_path, encoding='utf-8') as f:
        lines = f.readlines()

    # 构造全为 "London" 的预测列表
    predictions = ["London"] * len(lines)

    # 调用 utils 中的函数计算准确率
    total, correct = utils.evaluate_places(eval_corpus_path, predictions)

    if total > 0:
        accuracy = (correct / total) * 100
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
