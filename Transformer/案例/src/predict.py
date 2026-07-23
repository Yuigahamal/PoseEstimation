import torch
import torch.nn as nn
from torch import device

from tokenizer import ChineseTokenizer,EnglishTokenizer
from model import TranslationModel
import config

def run_predict():
    print("模型加载中")
    # 1.设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'使用设备名称为{torch.cuda.get_device_name(device)}')

    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

    # 3.模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_idx=zh_tokenizer.pad_token_index,
                             en_padding_idx=en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / 'best.pth'))
    print("模型加载成功")

    print("欢迎使用中英翻译小模型(输入q或quit退出)")

    while True:
        user_input = input("> ")
        if user_input == "q" or user_input == "quit":
            break
        if user_input.strip() == "":
            print("请输入中文内容")
            continue


        result = predict(user_input,model,zh_tokenizer,en_tokenizer,device)

        print(f"翻译结果：{result}")


def predict(user_input,model,zh_tokenizer,en_tokenizer,device):
    # 1.处理输入
    indexes = zh_tokenizer.encode(user_input)
    input_tensor = torch.tensor([indexes],dtype= torch.long).to(device) # input_tensor.shape(batch_size,seq_len)

    # 2.预测一批
    batch_result = predict_batch(model,input_tensor,en_tokenizer,device)

    return en_tokenizer.decode(batch_result[0])

def predict_batch(model,input_tensor,en_tokenizer,device):
    model.eval()
    with torch.no_grad():
        batch_size = input_tensor.size(0)
        src = input_tensor  # src.shape (batch_size,src_seq_len)
        src_key_padding_mask =  (src == model.zh_embedding.padding_idx) # src_key_padding_mask.shape (batch_size,src_seq_len)

        memory = model.encode(src=input_tensor,src_key_padding_mask=src_key_padding_mask) # memory.shape (batch_size,src_seq_len)

        decoder_input = torch.full([batch_size,1],en_tokenizer.sos_token_index,dtype=torch.long,device=device)
        # decoder_input.shape(batch_size,tgt_seq_len)

        # 预测结果缓存
        generate = []

        # 记录这一批样本是否已经生成结束符
        if_finish =  torch.full([batch_size],False,dtype=torch.bool,device=device)




        # 自回归生成
        for i in range(config.MAX_SEQ_LENGTH):
            # 生成因果遮罩
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[-1]).to(device)

            decoder_output = model.decode(tgt=decoder_input,memory=memory,
                         memory_key_padding_mask=src_key_padding_mask,
                         tgt_mask=tgt_mask)
            # decoder_output.shape(batch_size,tgt_seq_len,en_vocab_size)

            next_token_indexes = torch.argmax(decoder_output[:,-1,:],dim=-1,keepdim=True) #next_token_indexes.shape(batch_size,1)

            generate.append(next_token_indexes)

            # 更新输入
            decoder_input = torch.cat((decoder_input,next_token_indexes),dim=-1)

            if_finish |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if if_finish.all():
                break

        generate_tensor = torch.cat(generate,dim=1)
        generate_list = generate_tensor.tolist()

        for index,sentence in enumerate(generate_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generate_list[index] = generate_list[index][:eos_pos]

        return generate_list



if __name__ == '__main__':
    run_predict()
