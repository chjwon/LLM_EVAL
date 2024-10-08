import pandas as pd
from LLM_EVAL.utils import bleu_score, rougue_score, bert_score, selfcheck_nli_score
import ast
import numpy as np
fileName = "/home/jaewon/HMD_files/src/task1/results/Oct/task1_genBK_llava_zero_result_3.csv"
file = pd.read_csv(fileName)
output_fileName = f"{fileName[:-4]}_with_scores.csv"

"""
precision recall calculation

precision(Bj): max_i metric(Bj, Ai)
recall(Ai): max_j metric(Bj, Ai)
F1(avg precision, avg recall) = 2PR/(P+R) *****

A1=B1, A2=B2, A3 new
=> avg recall: 2/3
=> avg precision: 1

"""

bleu_scores = []
rouge_scores = []
bert_scores = []
selfscore_scores = []

def safe_convert_to_string(value):
    if isinstance(value, float) and pd.isna(value):
        return "no_response"
    return str(value)

def f1_convert(precision, recall):
    return (2*precision*recall)/(precision+recall)

def pre_recall(generated_text, reference_text, metric_function):
    bleu_matrix = []
    for curr_gen in generated_text:
        temp = []
        for curr_ref in reference_text:
            temp.append(metric_function(curr_gen, curr_ref))
        bleu_matrix.append(temp)
    np_matrix = np.array(bleu_matrix)
    row_max = np_matrix.max(axis=1).mean()
    col_max = np_matrix.max(axis=0).mean()
    return row_max, col_max

    
    
for idx, row in file.iterrows():
    print("Current idx",idx)
    generated_text = safe_convert_to_string(row['task1_genBK_llava_zero_parsed'])
    if generated_text == "no_response":
        generated_text = '["no_response"]'
    generated_text = ast.literal_eval(generated_text)
    reference_text = safe_convert_to_string(row['background_knowledge'])
    if reference_text == "no_response":
        reference_text = '["no_response"]'
    reference_text = ast.literal_eval(reference_text)
    
    bleu_pre , bleu_recall = pre_recall(generated_text, reference_text, bleu_score)
    bleu = f1_convert(bleu_pre,bleu_recall)
    
    rougue_pre , rougue_recall = pre_recall(generated_text, reference_text, rougue_score)
    rougue = f1_convert(rougue_pre,rougue_recall)
    
    bert_pre , bert_recall = pre_recall(generated_text, reference_text, bert_score)
    bert = f1_convert(bert_pre,bert_recall)
    
    selfscore_pre , selfscore_recall = pre_recall(generated_text, reference_text, selfcheck_nli_score)
    selfscore = f1_convert(selfscore_pre,selfscore_recall)
    


    bleu_scores.append(bleu)
    rouge_scores.append(rougue)
    bert_scores.append(bert)
    selfscore_scores.append(selfscore)

file['BLEU_Score'] = bleu_scores
file['ROUGE_Score'] = rouge_scores
file['BERT_Score'] = bert_scores
file['SelfCheckNLI_Score'] = selfscore_scores

file.to_csv(output_fileName, index=False)

file.head()
