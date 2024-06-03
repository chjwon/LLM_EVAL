"""
All function has to be
input1 : sentence | type : str | sentence_generated
input2 : sentence | type : str | sentence_gold

output : score | type float

"""



import torch
from bert_score import score
import evaluate
from evaluate import load

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
openaiKey = open("openaiKey.txt",'r').readline()
# print(openaiKey)


def bert_score(sentence_generated,sentence_gold):
    cands = [sentence_generated]
    refs = [sentence_gold]
    (P, R, F), hashname = score(cands, refs, lang="en", return_hash=True)
    return P.mean().item()


def selfcheck_nli_score(sentence_generated,sentence_gold):
    selfcheck_nli = SelfCheckNLI(device=device)

    sent_scores_nli = selfcheck_nli.predict(
    sentences = [sentence_gold],                          
    sampled_passages = [sentence_generated],
    )
    return normalize_selfcheck_score(sent_scores_nli[0])

def normalize_selfcheck_score(score):
    return 1 - score

def bleu_score(sentence_generated,sentence_gold):
    references = [sentence_gold.split()]
    candidate = sentence_generated.split()
    score = sentence_bleu(references, candidate)
    return score

def rougue_score(sentence_generated,sentence_gold):
    references = [[sentence_gold]]
    candidate = [sentence_generated]
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=candidate, references=references)
    
    score = results['rougeL']
    return score