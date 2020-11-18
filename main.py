from simpletransformers.classification import ClassificationModel
import pandas as pd


# https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3

def main():
    train_df = pd.read_csv('data/train.csv', header=None).iloc[1:]
    eval_df = pd.read_csv('data/test.csv', header=None).iloc[1:]

    # 0 = human
    # 1 = bot
    train_df[2] = (train_df[2] == 'bot').astype(int)
    eval_df[2] = (eval_df[2] == 'bot').astype(int)

    train_df = pd.DataFrame({
        'text': train_df[1].replace(r'\n', ' ', regex=True),
        'label': train_df[2]
    })
    eval_df = pd.DataFrame({
        'text': eval_df[1].replace(r'\n', ' ', regex=True),
        'label': eval_df[2]
    })

    print(train_df.head())

    model = ClassificationModel('roberta', 'roberta-base', use_cuda=False)
    model.train_model(train_df)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df)


if __name__ == '__main__':
    main()
