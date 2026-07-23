import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import config



class  TranslationDataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        df = pd.read_json(data_path,lines=True,orient='records')
        self.x = df['zh'].to_list()
        self.y = df['en'].to_list()
        self.len = len(self.x)


    def __getitem__(self, index) :
        
        return torch.tensor(self.x[index],dtype=torch.long),torch.tensor(self.y[index],dtype=torch.long)


    def __len__(self):
        return self.len  


def collate_fn(batch):
    # print(batch) # 一批原始样本,一个二元组列表
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]

    input_tensors=pad_sequence(input_tensors,batch_first=True,padding_value=0) # input_tensor.shape = [batch_size,sqn_len]
    target_tensors=pad_sequence(target_tensors,batch_first=True,padding_value=0) # output_tensor.shape = [batch_size,sqn_len]

    return input_tensors,target_tensors
    


def get_dataloader(is_train=True):
    data_path = config.PROCESSED_DATA_DIR / f'{"train.jsonl" if is_train else "test.jsonl"}'
    dataset = TranslationDataset(data_path)
    dataloader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True if is_train else False,num_workers=config.NUM_WORKERS,collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    train_loader = get_dataloader(is_train=True)
    test_loader = get_dataloader(is_train=False)

    for x,y in train_loader:
        print(x.shape,y.shape)
        break