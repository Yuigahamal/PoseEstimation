import torch.nn as nn
import torch
from torch import optim
from torch.utils import tensorboard
import time
from tqdm import tqdm
from dataset import get_dataloader
from tokenizer import ChineseTokenizer,EnglishTokenizer
import config
from model import TranslationModel

def train():
    # 1.定义设备
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备名称为{torch.cuda.get_device_name(device)}')

    # 2.获取数据
    train_data_loader = get_dataloader(is_train=True)

    # 3.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

    # 4.模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_idx=zh_tokenizer.pad_token_index,
                             en_padding_idx=en_tokenizer.pad_token_index).to(device)

    # 5.损失函数
    loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)

    # 6.优化器
    optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    # 7.TensorBoard Writer
    writer = tensorboard.SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))


    # 8.训练模型
    best_loss = float('inf')
    model.train()
    for epoch in range(1,config.NUM_EPOCHS+1):
        print("=" * 10, f"开始训练第{epoch}轮", "=" * 10)
        # 训练一个epoch
        loss = train_one_epoch(model,train_data_loader,loss_fn,optimizer,device)
        print(f"loss:{loss}")
        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(),config.MODEL_DIR / "best.pth")

    # 9.关闭Writer
    writer.close()

def train_one_epoch(model:TranslationModel,train_data_loader,loss_fn,optimizer,device):
    total_loss = 0
    for inputs,targets in tqdm(train_data_loader,desc='训练中'):
        inputs = inputs.to(device)
        targets = targets.to(device)

        src = inputs # 编码器输入 src.shape (batch_size,src_seq_len)

        tgt = targets[:,:-1] # 解码器输入  tgt.shape (batch_size,tgt_seq_len)
        decoder_targets = targets[:,1:] # 解码器预期输出 decoder_targets.shape (batch_size,tgt_seq_len)

        src_key_padding_mask = (src == model.zh_embedding.padding_idx) # src_key_padding_mask.shape (batch_size,src_seq_len)
        memory_key_padding_mask = src_key_padding_mask  # memory_key_padding_mask 其实跟 src_key_padding_mask 一摸一样的

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[-1]).to(device) # tgt_mask.shape(tgt_seq_len,tgt_seq_len)
        # nn.Transformer.generate_square_subsequent_mask()可以直接生成一个因果掩码，专门用于masked自注意力层

        decoder_outputs = model(src=src,tgt=tgt,
                        src_key_padding_mask=src_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        tgt_mask=tgt_mask,) # outputs.shape (batch_size,tgt_seq_len,en_vocab_size)

        decoder_targets = decoder_targets.reshape(-1)  # decoder_targets.shape(batch_size * tgt_seq_len)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])  # decoder_outputs.shape(batch_size*tgt_seq_len,en_vocab_size)

        loss = loss_fn(decoder_outputs,decoder_targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(train_data_loader)


if __name__ == "__main__":
    train()





