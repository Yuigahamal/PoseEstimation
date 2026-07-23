import torch.nn as nn
import torch
import config


class PositionEncoding(nn.Module):
        def  __init__(self):
            super().__init__()
            pos = torch.arange(config.MAX_SEQ_LENGTH,dtype=torch.float).unsqueeze(1) # pos.shape(MAX_SEQ_LENGTH,1)
            _2i = torch.arange(0,config.DIM_MODEL,step=2,dtype=torch.float) # _2i.shape(DIM_MODEL/2)
            div_term  =  torch.pow(10000, _2i / config.DIM_MODEL)

            sins = torch.sin(pos / div_term) # sins.shape(MAX_SEQ_LENGTH,DIM_MODEL/2)
            coss = torch.cos(pos / div_term) # coss.shape(MAX_SEQ_LENGTH,DIM_MODEL/2)

            pe = torch.zeros([config.MAX_SEQ_LENGTH,config.DIM_MODEL],dtype=torch.float)
            pe[:,0::2]=sins
            pe[:,1::2]=coss
            
            self.register_buffer('pe',pe)

        def forward(self,x):
             # x.shape ( batch_size, seq_length, dim_model)
             batch_size,seq_length,dim_model = x.shape
             return x+self.pe[:seq_length] # type: ignore

# class PositionEncoding(nn.Module): 
#     def  __init__(self):
#         super().__init__()
#         self.pe = torch.zeros([configs.MAX_SEQ_LENGTH,configs.DIM_MODEL],dtype=torch.float)
#         for pos in range(configs.MAX_SEQ_LENGTH):
#             for i in range(configs.DIM_MODEL):
#                 if i % 2 == 0:
#                     self.pe[pos][i] = math.sin(pos / 10000**(i/configs.DIM_MODEL))
#                 else:
#                     self.pe[pos][i] = math.cos(pos / 10000**(i/configs.DIM_MODEL))

#     def forward(self,x):
#         # x:(batch_size, seq_length, dim_model)
#         batch_size,seq_length,dim_model = x.shape
#         part_pe = self.pe[:seq_length,:].unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
#         return x+part_pe  


class TranslationModel(nn.Module):
    def __init__(self,zh_vocab_size,en_vocab_size,zh_padding_idx,en_padding_idx):
        super().__init__()

        # 词嵌入器
        self.zh_embedding = nn.Embedding(num_embeddings=zh_vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=zh_padding_idx)

        self.en_embedding = nn.Embedding(num_embeddings=en_vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=en_padding_idx)


        # 位置编码
        self.position_encoding = PositionEncoding()

        # transformer模型
        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                                                nhead=config.N_HEAD,
                                                                num_encoder_layers=config.NUM_ENCODER_LAYER,
                                                                num_decoder_layers=config.NUM_DECODER_LAYER,
                                                                dim_feedforward=config.DIM_FEEDFORWARD,
                                                                dropout=0.1,
                                                                batch_first=True)
        
        self.fc = nn.Linear(in_features=config.DIM_MODEL,out_features=en_vocab_size)

        # def forward(self,src,tgt,src_key_padding_mask,tgt_mask,mem)

    def forward(self,src,tgt,src_key_padding_mask=None,tgt_mask=None,memory_key_padding_mask=None):
        memory = self.encode(src=src,src_key_padding_mask=src_key_padding_mask)
        outputs = self.decode(tgt=tgt,memory=memory,tgt_mask=tgt_mask,memory_key_padding_mask=memory_key_padding_mask)
        return outputs

    def encode(self,src,src_key_padding_mask=None):
        # src.shape (batch_size,seq_len)
        # src_key_padding_mask.shape (batch_size,seq_len)
        vocab_vector = self.zh_embedding(src) # vocab_vector.shape (batch_size,seq_len,dim_model)
        x = self.position_encoding(vocab_vector)  # x.shape(batch_size,seq_len,dim_model)
        memory = self.transformer.encoder(src=x,src_key_padding_mask=src_key_padding_mask)  #memory.shape(batch_size,seq_len,dim_model)
        return memory

    def decode(self,tgt,memory,tgt_mask=None,memory_key_padding_mask=None):
        # tgt.shape (batch_size,tgt_seq_len)
        # memory.shape (batch_size,seq_len,dim_model)
        # tgt_mask.shape (batch_size,tgt_)

        vocab_vector = self.en_embedding(tgt) # vocab_vector.shape (batch_size,tgt_seq_len,dim_model)
        x = self.position_encoding(vocab_vector)  # x.shape(batch_size,tgt_seq_len,dim_model)
        out = self.transformer.decoder(tgt=x,memory=memory,tgt_mask=tgt_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)
        # out.shape(batch_size,tgt_seq_len,dim_model)

        outputs = self.fc(out) # outputs.shape(batch_size,tgt_seq_len,en_vocab_size)
        return outputs






