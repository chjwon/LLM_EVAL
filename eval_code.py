import pandas as pd
from LLM_EVAL.utils import bleu_score, rougue_score, bert_score, selfcheck_nli_score
import ast
fileName = "/home/jaewon/HMD_files/src/task1/results/Oct/task1_genBK_llava_zero_result.csv"
file = pd.read_csv(fileName)



bleu_scores = []
rouge_scores = []
bert_scores = []
selfscore_scores = []

def safe_convert_to_string(value):
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)

for idx, row in file.iterrows():
    print("Current idx",idx)
    generated_text = safe_convert_to_string(row['task1_genBK_llava_zero'])
    generated_text = ast.literal_eval(generated_text)[0]
    reference_text = safe_convert_to_string(row['background_knowledge'])
    reference_text = ast.literal_eval(reference_text)[0]
    
    if generated_text and reference_text:
        bleu = bleu_score(generated_text, reference_text)
        rouge = rougue_score(generated_text, reference_text)
        bert = bert_score(generated_text, reference_text)
        selfscore = selfcheck_nli_score(generated_text, reference_text)
    else:
        bleu = rouge = bert = deberta_cos = None

    bleu_scores.append(bleu)
    rouge_scores.append(rouge)
    bert_scores.append(bert)
    selfscore_scores.append(selfscore)

file['BLEU_Score'] = bleu_scores
file['ROUGE_Score'] = rouge_scores
file['BERT_Score'] = bert_scores
file['SelfCheckNLI_Score'] = selfscore_scores

output_fileName = f"{fileName[:-4]}_with_scores.csv"
file.to_csv(output_fileName, index=False)

file.head()
