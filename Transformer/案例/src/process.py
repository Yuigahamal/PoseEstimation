from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd
import config
from tokenizer import ChineseTokenizer,EnglishTokenizer

def process():
    print("开始处理数据")

    # 1.读取原数据
    path = config.RAW_DATA_DIR/ "cmn.txt"
    df = pd.read_table(path,header=None,names=["en","zh","License"],encoding="utf-8").drop(columns=["License"]).dropna()

    # 2.划分训练集和测试集
    train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)

    # 3.根据训练集构建词表
    ChineseTokenizer.build_vocab(train_df["zh"].to_list(),config.PROCESSED_DATA_DIR/"zh_vocab.txt") # 中文词表
    EnglishTokenizer.build_vocab(train_df["en"].to_list(),config.PROCESSED_DATA_DIR/"en_vocab.txt") # 英文词表

    # 构造tokenizer实例
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR/"zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR/"en_vocab.txt")

    # 4.构造训练集并保存
    train_df['zh']=train_df['zh'].apply(lambda x: zh_tokenizer.encode(x,add_sos_eos=False))
    train_df['en']=train_df['en'].apply(lambda x: en_tokenizer.encode(x,add_sos_eos=True))
    train_df.to_json(config.PROCESSED_DATA_DIR/"train.jsonl",orient="records",lines=True)

    # 5.构造测试集并保存
    test_df['zh']=test_df['zh'].apply(lambda x: zh_tokenizer.encode(x,add_sos_eos=False))
    test_df['en']=test_df['en'].apply(lambda x: en_tokenizer.encode(x,add_sos_eos=True))
    test_df.to_json(config.PROCESSED_DATA_DIR/"test.jsonl",orient="records",lines=True)


    print("数据处理完成")

if __name__=="__main__":
    process()