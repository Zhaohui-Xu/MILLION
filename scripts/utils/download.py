# import torchtext; torchtext.disable_torchtext_deprecation_warning()

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
os.environ['https_proxy'] = '127.0.0.1:7897'


from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
# from torchtext.datasets import WikiText103
from .Timer import tprint

import argparse


def download_models(path, root):
    tprint('Downloading model {path} to {root}')

    tprint('Downloading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(path)
    tprint('Downloading model')
    model = AutoModelForCausalLM.from_pretrained(path)
    os.makedirs(root, exist_ok=True)

    model_name = path.split('/')[-1]

    tokenizer.save_pretrained(root + model_name)
    model.save_pretrained(root + model_name)

def download_wikitext103(root):
    load_dataset(path='wikitext', name='wikitext-103-v1').save_to_disk(root + 'wikitext-103-v1')

def download_wikitext103_raw(root):
    load_dataset(path='wikitext', name='wikitext-103-raw-v1').save_to_disk(root + 'wikitext-103-raw-v1')


def download_PTB(root):
    load_dataset("ptb_text_only").save_to_disk(root + "ptb_text_only")

def download_wikitext2(root):
    load_dataset("wikitext", "wikitext-2-raw-v1").save_to_disk(root + "wikitext-2-raw-v1")
    
def download_gsm8k(root):
    load_dataset("gsm8k", "main").save_to_disk(root + "gsm8k")
    
def download_longbench(root):
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # datasets = ["narrativeqa"]

    for dataset in datasets:
        load_dataset('THUDM/LongBench', dataset).save_to_disk(root + dataset)
        
def download_longbench_e(root):
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    
    for dataset in datasets:
        dataset = dataset + "_e"
        data = load_dataset('THUDM/LongBench', dataset).save_to_disk(root + dataset)

if __name__ == "__main__":
    model_hints = [
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl",
        "bigscience/bloom-7b1",
        "google-bert/bert-base-cased",
        "facebook/opt-6.7b",
        "mosaicml/mpt-7b"
    ]

    parser = argparse.ArgumentParser(description='Download models and datasets')
    parser.add_argument('--model', type=str, help='Model name to download, choose from ' + ', '.join(model_hints) + ' or any other model name from Huggingface model hub')
    parser.add_argument('--dataset', type=str, help='Dataset name to download')
    # parser.add_argument('--root', type=str, help='Root directory to save models and datasets')
    args = parser.parse_args()

    file_path = os.path.abspath(__file__)
    os.chdir(os.path.dirname(file_path))
    
    if args.model:
        download_models(args.model, root='../../models/')
    if args.dataset:
        if args.dataset == 'wikitext103':
            download_wikitext103(root='../../datasets/')
        elif args.dataset == 'PTB':
            download_PTB(root='../../datasets/')
        elif args.dataset == 'wikitext2':
            download_wikitext2(root='../../datasets/')
        elif args.dataset == 'wikitext103-raw':
            download_wikitext103_raw(root='../../datasets/')
        elif args.dataset == 'gsm8k':
            download_gsm8k(root='../../datasets/')
        elif args.dataset == 'longbench':
            download_longbench(root='../../datasets/')
        elif args.dataset == 'longbench_e':
            download_longbench_e(root='../../datasets/')
        else:
            raise ValueError('Dataset not supported')