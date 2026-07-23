
import tqdm
import nltk


class BaseTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token=  '<sos>'
    eos_token = '<eos>'
    def __init__(self,vocab_list:list[str]):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def tokenize(cls,text) -> list[str]:
        pass
        return []
    
    def encode(self,text,add_sos_eos=False):
        tokens = self.tokenize(text)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        encoded = [self.word2index.get(token,self.unk_token_index) for token in tokens]
        return encoded
    

    #构建词表并保存词表到txt文件
    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        # 构建词表
        vocab_set = set()
        for sentence in tqdm.tqdm(sentences,desc='构建词表'):
            tokens  = cls.tokenize(sentence)
            vocab_set.update(tokens) # update方法可以直接将列表中的所有元素到集合中

        vocab_list = [token  for token in vocab_set if token.strip()!='']
        vocab_list.insert(0,cls.eos_token)
        vocab_list.insert(0,cls.sos_token)
        vocab_list.insert(0,cls.unk_token)
        vocab_list.insert(0,cls.pad_token)
        

        #保存词表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word+'\n')
        print(f'词表大小为{len(vocab_list)}')



    # 返回一个JiebaTokenizer实例用于调用
    @classmethod
    def from_vocab(cls,vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [word.strip() for word in f.readlines()]
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    def __init__(self,vocab_list:list[str]):
        super().__init__(vocab_list)

    @classmethod
    def tokenize(cls,text) -> list[str]:
        return list(text)
    

class EnglishTokenizer(BaseTokenizer):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    detokenizer = nltk.tokenize.TreebankWordDetokenizer()

    def __init__(self,vocab_list:list[str]):
        super().__init__(vocab_list)


    @classmethod
    def tokenize(cls,text) -> list[str]:
        return cls.tokenizer.tokenize(text)
    

    def decode(self,token_index_list):
        tokens = [self.index2word[index] for index in token_index_list]
        return self.detokenizer.detokenize(tokens)

        

        
if __name__ == '__main__':
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    detokenizer = nltk.tokenize.TreebankWordDetokenizer()
    word_list = tokenizer.tokenize("This is a test sentence.")
    print(word_list)
    print(detokenizer.detokenize(word_list))






