import argparse
import pandas as pd
import string
import re

# 1. converted to lowercase
def lowercase_text(input_file, output_file):
    # 加载数据
    df = pd.read_csv(input_file)
    # 转换 'text' 列为小写
    df['text'] = df['text'].str.lower()
    # 保存修改后的 DataFrame
    df.to_csv(output_file, index=False)
    print(f"Data has been converted to lowercase and saved to {output_file}")

def remove_punctuation(text):
    # 创建一个翻译表，其中所有标点符号都映射为 None
    translator = str.maketrans('', '', string.punctuation)
    # 使用翻译表删除文本中的所有标点符号
    return text.translate(translator)

# 2. 删除多余的标点符号
def remove_punctuation_main(input_file, output_file):
    df = pd.read_csv(input_file)
    df['text'] = df['text'].apply(remove_punctuation)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} after removing punctuation.")


# 缩写词字典
contractions = {
    "can't": "cannot",
    "can’t": "cannot",
    "can‘t": "cannot",
    "won't": "will not",
    "won’t": "will not",
    "won‘t": "will not",
    "i'm": "i am",
    "i’m": "i am",
    "i‘m": "i am",
    "you're": "you are",
    "you’re": "you are",
    "you‘re": "you are",
    "he's": "he is",
    "he‘s": "he is",
    "he’s": "he is",
    "she's": "she is",
    "she’s": "she is",
    "she‘s": "she is",
    "it's": "it is",
    "it’s": "it is",
    "it‘s": "it is",
    "we're": "we are",
    "we’re": "we are",
    "we‘re": "we are",
    "they're": "they are",
    "they‘re": "they are",
    "they’re": "they are",
    "i've": "i have",
    "i’ve": "i have",
    "i‘ve": "i have",
    "you've": "you have",
    "you’ve": "you have",
    "you‘ve": "you have",
    "we've": "we have",
    "we’ve": "we have",
    "we‘ve": "we have",
    "they've": "they have",
    "they’ve": "they have",
    "they‘ve": "they have",
    "isn't": "is not",
    "isn’t": "is not",
    "isn‘t": "is not",
    "aren't": "are not",
    "aren’t": "are not",
    "aren‘t": "are not",
    "wasn't": "was not",
    "wasn’t": "was not",
    "wasn‘t": "was not",
    "weren't": "were not",
    "weren’t": "were not",
    "weren‘t": "were not",
    "haven't": "have not",
    "haven’t": "have not",
    "haven‘t": "have not",
    "hasn't": "has not",
    "hasn’t": "has not",
    "hasn‘t": "has not",
    "hadn't": "had not",
    "hadn’t": "had not",
    "hadn‘t": "had not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "doesn‘t": "does not",
    "don't": "do not",
    "don’t": "do not",
    "don‘t": "do not",
    "didn't": "did not",
    "didn’t": "did not",
    "didn‘t": "did not",
    # 更多缩写词
}

def expand_contractions(text):
    for word, expanded_word in contractions.items():
        text = text.replace(word, expanded_word)
    return text

# 加载数据、处理缩写词、保存数据
def expand_contractions_main(input_file, output_file):
    df = pd.read_csv(input_file)
    df['text'] = df['text'].apply(expand_contractions)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} after expanding contractions.")


def remove_specific_chars(text, chars_to_remove):
    # 创建一个正则表达式，匹配所有指定的字符
    regex_pattern = '[' + re.escape(chars_to_remove) + ']'
    # 删除这些字符
    cleaned_text = re.sub(regex_pattern, '', text)
    return cleaned_text

def remove_symbol(input_file, output_file):
    df = pd.read_csv(input_file)
    df['text'] = df['text'].apply(lambda x: remove_specific_chars(x, "‘’“”"))
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file} after removing specific characters.")

def main():
    parser = argparse.ArgumentParser(description="Data Cleaning Tool")
    parser.add_argument('--input_file', type=str, default='/root/autodl-tmp/cmu-mosi/label.csv', help='The path to the input CSV file')
    parser.add_argument('--output_file', type=str, default='/root/autodl-tmp/cmu-mosi/label.csv', help='The path to the output CSV file')
    parser.add_argument('--lowercase', action='store_true', help='Convert text to lowercase')
    parser.add_argument('--remove_punct', action='store_true', help='Remove all punctuation from text')
    parser.add_argument('--expand_contractions', action='store_true', help='Expand contractions in text')
    parser.add_argument('--remove_symbol', action='store_true', help='Expand contractions in text')

    args = parser.parse_args()

    if args.lowercase:
        lowercase_text(args.input_file, args.output_file)

    if args.remove_punct:
        remove_punctuation_main(args.input_file, args.output_file)

    if args.expand_contractions:
        expand_contractions_main(args.input_file, args.output_file)

    if args.remove_symbol:
        remove_symbol(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

# 2. 处理缩词

