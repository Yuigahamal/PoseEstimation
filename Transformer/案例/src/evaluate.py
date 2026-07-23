import torch
from model import TranslationModel
import config
from dataset import get_dataloader
from tokenizer import ChineseTokenizer, EnglishTokenizer
from predict import predict_batch
from nltk.translate.bleu_score import corpus_bleu


def evaluate(model, test_data_loader, device, en_tokenizer):
    predictions = []
    # predictions = [[***],[******],[****]]
    references = []
    # references = [[***],[******],[****]]
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_data_loader:
            inputs = inputs.to(device)
            targets = targets.tolist()
            # targets =  [[*****],[***<pad><pad>],[**<pad><pad><pad>]]

            batch_result = predict_batch(model, inputs, en_tokenizer, device)
            # batch_result = [[***],[******],[****]]

            predictions.extend(batch_result)
            references.extend([[target[1:target.index(en_tokenizer.eos_token_index)]] for target in targets])

    return corpus_bleu(references, predictions)


def run_evluate():
    # 1.设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.构建tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

    # 3.加载模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_idx=zh_tokenizer.pad_token_index,
                             en_padding_idx=en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODEL_DIR / 'best.pth'))
    model.to(device)
    print("加载模型成功")

    # 4.测试集数据
    test_data_loader = get_dataloader(is_train=False)

    # 5.评估
    bleu = evaluate(model, test_data_loader, device, en_tokenizer)

    print("评估结果")
    print(f"BLEU:{bleu}")


if __name__ == "__main__":
    run_evluate()