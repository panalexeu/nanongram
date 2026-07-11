import argparse 
from pathlib import Path

from preproc import TgPreporcStrategy
from model import Model
from tokenizer import BaseTokenizer

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="tg ingestion pipeline: cnv preprocessing -> ngram model training -> counts/probs export")
    # preproc args 
    parser.add_argument("user_id", type=str)
    parser.add_argument("cnv_path", type=str) 
    parser.add_argument("--preproc_export_path", type=str, default='./out.txt')
    # model training arg s
    parser.add_argument("--ngram", type=int, default=2)  # TODO change to 1 defautl here, once unigram sampling is supported 
    parser.add_argument("--export_count_path", type=str, default=None)
    parser.add_argument("--export_prob_path", type=str, default='./outprob.pkl')
    parser.add_argument
    args = parser.parse_args()

    preproc_ = TgPreporcStrategy(
        path=Path(args.cnv_path), 
        id_=args.user_id
    )
    preproc_.export(path=args.preproc_export_path)
    
    tokenizer = BaseTokenizer()
    model = Model(
        ngram=args.ngram,
        tokenizer=tokenizer, 
        export_count_path=args.export_count_path, 
        export_prob_path=args.export_prob_path 
    )
    model.load_raw(args.preproc_export_path)
    model.tokenize()
    model.count()
    model.count_prob()
    if args.export_count_path: model.export_count_dict() 
    model.export_prob_dict()    
