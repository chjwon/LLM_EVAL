import json
from LLM_EVAL.utils import bleu_score, rougue_score, bert_score, selfcheck_nli_score
import pandas as pd
import numpy as np
import ast
import re
import argparse

# def txt_to_list(text):
#     if text is None:
#         return ['No_response']
#     split_text = re.split(r'(?<=\d\.)\s', text)

#     if len(split_text) == 1:  # No numbering found
#         sentences = [text.strip()]  # Treat the entire text as a single list item
#     else:
#         sentences = [re.sub(r'^\d+\.\s*', '', sentence).strip() for sentence in split_text]
#     return sentences

def safe_convert_to_string(value):
    if isinstance(value, float) and pd.isna(value):
        return "no_response"
    return str(value)

def f1_convert(precision, recall):
    if precision == recall:
        return recall
    else:
        return (2*precision*recall)/(precision+recall)

def pre_recall(generated_text, reference_text, metric_function):
    # print(type(generated_text), len(generated_text))
    reference_text = reference_text[0]
    # print(type(reference_text), len(reference_text))
    if generated_text is None:
        generated_text = "no_response"
    if reference_text is None:
        reference_text = "no_response"
    row_max = metric_function(generated_text, reference_text)
    col_max = metric_function(reference_text, generated_text)
    return row_max, col_max

def main():

    parser = argparse.ArgumentParser(description="Set the target value.")

    # Add an argument for the target
    parser.add_argument(
        '--target', 
        type=str, 
        default='none_ft',  # Default value if no input is provided
        help='Specify the target value (default: none_ft)'
    )
    args = parser.parse_args()
    target = args.target
    print(f"Target value: {target}")


    # Load the updated JSON file
    json_file_path = "/home/jaewon/auto_eval/1124_all_outputs_updated.json"
    # json_file_path = "/home/jaewon/auto_eval/1124_all_outputs_final.json"

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    
    

    # Task 1: Calculate scores for each entry
    for key in data.keys():
        print(f"Current : {target} - {key}")
        sentence_generated = data[key][f'mc/{target}']  # Extract "mc/none_ft"
        sentence_gold = data[key]['implicit_message']  # Extract "implicit_message"
        # sentence_generated = txt_to_list(sentence_generated)
        sentence_gold = ast.literal_eval(sentence_gold)
        
        # Calculate scores
        bleu_pre , bleu_recall = pre_recall(sentence_generated, sentence_gold,bleu_score)
        bleu = f1_convert(bleu_pre , bleu_recall)
        rougue_pre , rougue_recall = pre_recall(sentence_generated, sentence_gold,rougue_score)
        rouge = f1_convert(rougue_pre , rougue_recall)
        bert_pre , bert_recall = pre_recall(sentence_generated, sentence_gold,bert_score)
        bert = f1_convert(bert_pre , bert_recall)
        nli_pre, nli_recall = pre_recall(sentence_generated, sentence_gold,selfcheck_nli_score)
        nli = f1_convert(nli_pre, nli_recall)
        
        # Save scores back to JSON
        data[key][f'{target}_bleu_score'] = bleu
        data[key][f'{target}_rougue_score'] = rouge
        data[key][f'{target}_bert_score'] = bert
        data[key][f'{target}_selfcheck_nli_score'] = nli

    # Task 2: Calculate mean scores based on 'hateful'
    scores_hateful = {'bleu_score': [], 'rougue_score': [], 'bert_score': [], 'selfcheck_nli_score': []}
    scores_non_hateful = {'bleu_score': [], 'rougue_score': [], 'bert_score': [], 'selfcheck_nli_score': []}

    for key in data.keys():
        hateful = data[key]['hateful']
        
        # Append scores to respective lists
        target_scores = scores_hateful if hateful == 1 else scores_non_hateful
        target_scores['bleu_score'].append(data[key][f'{target}_bleu_score'])
        target_scores['rougue_score'].append(data[key][f'{target}_rougue_score'])
        target_scores['bert_score'].append(data[key][f'{target}_bert_score'])
        target_scores['selfcheck_nli_score'].append(data[key][f'{target}_selfcheck_nli_score'])

    # Calculate mean scores
    def calculate_mean(scores_dict):
        return {score_name: sum(values) / len(values) if values else 0.0 for score_name, values in scores_dict.items()}

    mean_hateful_scores = calculate_mean(scores_hateful)
    mean_non_hateful_scores = calculate_mean(scores_non_hateful)

    # Print mean scores
    print("Mean Scores for Hateful Messages:")
    print(mean_hateful_scores)

    print("\nMean Scores for Non-Hateful Messages:")
    print(mean_non_hateful_scores)

    # Save updated JSON file
    updated_json_file_path = f"/home/jaewon/auto_eval/1125_{target}_score.json"
    with open(updated_json_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    # print(f"Updated JSON file with scores saved to {updated_json_file_path}")
if __name__ == "__main__":
    main()