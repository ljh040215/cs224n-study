import random
import argparse

import dataset
import models
import trainer
import utils

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

random.seed(0)

argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose pretrain, finetune, or evaluate")
argp.add_argument('variant', help="Choose vanilla or rope")
argp.add_argument('pretrain_corpus_path', default=None)
argp.add_argument('--reading_params_path',default=None)
argp.add_argument('--writing_params_path',default=None)
argp.add_argument('--finetune_corpus_path', default=None)
argp.add_argument('--eval_corpus_path', default=None)
argp.add_argument('--outputs_path', default=None)
argp.add_argument('--pretrain_lr', default=6e-3, type=float)
argp.add_argument('--finetune_lr', default=6e-4, type=float)
argp.add_argument('--tb_expt_name', help='debug string for tb log.',
                  default='run')
args = argp.parse_args()

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
elif torch.backends.mps.is_available() and args.variant == 'vanilla':
    device = 'mps'

# TensorBoard training log
writer = SummaryWriter(log_dir='expt/%s/%s_%s_pt_lr_%f_ft_lr_%f' % (
    args.function,
    args.tb_expt_name,
    args.variant,
    args.pretrain_lr,
    args.finetune_lr))

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path, encoding='utf-8').read()
pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the RoPE models
mconf = models.GPTConfig(
    pretrain_dataset.vocab_size,
    pretrain_dataset.block_size,
    n_layer=4,
    n_head=8,
    n_embd=256)

# define models.
# note: models should moved to device defined on lines 30-34.

model = None
if args.variant == 'vanilla':
    # TODO: [part c] Make some model here
    ### YOUR CODE HERE ###
    model = models.GPT(mconf).to(device)
    ### END YOUR CODE ###
elif args.variant == 'rope':
    # TODO: [part g] Make some other model here
    # set mconf.rope parameter
    ### YOUR CODE HERE ###
    mconf.rope = True
    model = models.GPT(mconf)
    model = model.to(device)
    ### END YOUR CODE ###
else:
    raise ValueError("Unknown model variant")

print('Model on device: ', next(model.parameters()).device)

# Perform pretraining, finetuning, or evaluation
if args.function == 'pretrain':
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path

    # - Make sure to use the following hyperparameters for pretraining:
    # Hyperparameters for pretraining:
    # max_epochs=650
    # batch_size=128
    # learning_rate=args.pretrain_lr
    # lr_decay=True
    # warmup_tokens=512*20
    # final_tokens=650*len(pretrain_dataset)*block_size
    # num_workers=4
    # writer=writer

    ### YOUR CODE HERE ###
    # 1. 配置预训练的超参数
    tconf = trainer.TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=args.pretrain_lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=650 * len(pretrain_dataset) * block_size,
        num_workers=4,
        ckpt_path=args.writing_params_path,  # 训练结束或变好时保存模型到此路径
        writer=writer
    )

    # 2. 初始化 Trainer 并开始训练
    # pretrain 阶段不需要测试集 (test_dataset=None)
    trainer_obj = trainer.Trainer(model, pretrain_dataset, None, tconf)
    trainer_obj.train()
    ### END YOUR CODE ###
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     [part d] Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     [part f] Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     You can use the args.reading_params_path flag to switch between the
    #     number of epochs for each case.

    ### YOUR CODE HERE ###
    # 1. 构造微调数据集
    finetune_text = open(args.finetune_corpus_path, encoding='utf-8').read()
    train_dataset = dataset.NameDataset(pretrain_dataset, finetune_text)

    # 2. 根据是否提供预训练参数路径来决定超参数
    if args.reading_params_path is None:
        # Case 1: [part c] 没有预训练模型 (从头训练)
        max_epochs = 75
        batch_size = 256
        learning_rate = args.finetune_lr
        lr_decay = True
        warmup_tokens = 512 * 20
        final_tokens = 200 * len(pretrain_dataset) * block_size
        num_workers = 4
    else:
        # Case 2: [part f] 有预训练模型 (加载权重并微调)
        # 加载预训练的 state_dict
        model.load_state_dict(torch.load(args.reading_params_path))

        max_epochs = 10  # 预训练后只需要很少的微调
        batch_size = 256
        learning_rate = args.finetune_lr
        lr_decay = True
        warmup_tokens = 512 * 20
        final_tokens = 200 * len(pretrain_dataset) * block_size
        num_workers = 4

    # 3. 初始化 TrainerConfig
    tconf = trainer.TrainerConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        num_workers=num_workers,
        ckpt_path=args.writing_params_path,  # 保存微调后的模型
        writer=writer
    )

    # 4. 初始化 Trainer 并开始训练
    trainer_obj = trainer.Trainer(model, train_dataset, None, tconf)
    trainer_obj.train()
    '''
    # 1. 构造数据集 (使用 NameDataset)
    # 读取微调语料库的文本内容
    finetune_text = open(args.finetune_corpus_path, encoding='utf-8').read()
    # 实例化 NameDataset，传入预训练的数据集对象(为了共享词表)和微调文本
    train_dataset = dataset.NameDataset(pretrain_dataset, finetune_text)

    # 2. 设置超参数
    # 根据是否传入了预训练参数路径 (args.reading_params_path) 来判断是 Part c 还是 Part f
    if args.reading_params_path is None:
        # Case: Without pretraining (Part c - 从头训练)
        max_epochs = 75
        batch_size = 256
        learning_rate = args.finetune_lr
        lr_decay = True
        warmup_tokens = 512 * 20
        final_tokens = 200 * len(pretrain_dataset) * block_size
        num_workers = 4
    else:
        # Case: With pretraining (Part f - 加载预训练权重)
        model.load_state_dict(torch.load(args.reading_params_path))
        max_epochs = 10
        batch_size = 256
        learning_rate = args.finetune_lr
        lr_decay = True
        warmup_tokens = 512 * 20
        final_tokens = 200 * len(pretrain_dataset) * block_size
        num_workers = 4

    # 3. 初始化 TrainerConfig
    tconf = trainer.TrainerConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        num_workers=num_workers,
        ckpt_path=args.writing_params_path,
        writer=writer
    )

    # 4. 初始化 Trainer 并开始训练
    trainer_obj = trainer.Trainer(model, train_dataset, None, tconf)
    trainer_obj.train()
'''
    ### END YOUR CODE ###
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w', encoding='utf-8') as fout:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x],
                             dtype=torch.long)[None,...].to(device)
            pred = utils.sample(model, x, 32, sample=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            fout.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to {args.outputs_path}; no targets provided')

