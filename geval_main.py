from openai import OpenAI
import argparse
import time
import pandas as pd
import os

def send_gpt(cur_prompt, apikey):
    client = OpenAI(api_key=apikey)
    score = 0
    try:
        _response = client.chat.completions.create(model="gpt-4-0613",
        messages=[{"role": "system", "content": cur_prompt}],
        temperature=2,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=20)
        time.sleep(0.5)

        all_responses = [_response.choices[i].message.content for i in
                            range(len(_response.choices))]
        scores = [float(response) for response in all_responses if response.replace('.', '', 1).isdigit()]
        if scores:
            score = sum(scores) / len(scores)
        else:
            print(f"No valid scores returned")
    except Exception as e:
        print(e)
        if ("limit" in str(e)):
            time.sleep(2)
        else:
            print('ignored')
    return score

def save_result(data_pd):
    file_name = './geval_result.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df = pd.concat([df,data_pd],ignore_index=True)
    else:
        df = data_pd
    df.to_csv(file_name, index=False)


def main(sentence_generated, sentence_gold):
    apikey = open("./openaiKey.txt",'r').readline()
    coh_prompt = open("./prompt/coh_detailed.txt").read() # /5
    con_prompt = open("./prompt/con_detailed.txt").read() # /5
    flu_prompt = open("./prompt/flu_detailed.txt").read() # /3
    rel_prompt = open("./prompt/rel_detailed.txt").read() # /5
    
    coh_prompt = coh_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    con_prompt = con_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    flu_prompt = flu_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    rel_prompt = rel_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    
    coh_score = send_gpt(cur_prompt=coh_prompt, apikey=apikey)
    con_score = send_gpt(cur_prompt=con_prompt, apikey=apikey)
    flu_score = send_gpt(cur_prompt=flu_prompt, apikey=apikey)
    rel_score = send_gpt(cur_prompt=rel_prompt, apikey=apikey)
    print("coh_score, con_score, flu_score, rel_score:", coh_score, con_score, flu_score, rel_score)
    myCsvRow = [sentence_gold, sentence_generated, coh_score, con_score, flu_score, rel_score]
    data = {
    'sentence_gold': [sentence_gold],
    'sentence_generated': [sentence_generated],
    'coh_score': [coh_score],
    'con_score': [con_score],
    'flu_score': [flu_score],
    'rel_score': [rel_score]
    }
    new_row = pd.DataFrame(data)
    save_result(new_row)
    return coh_score, con_score, flu_score, rel_score
    

if __name__ == '__main__':




    mock_gold = 'Trump is good and Biden is bad.'
    mock_generated_1 = 'Biden is good and Trump is bad.' # opposite+order
    mock_generated_2 = 'Biden is bad and Trump is good.' # order
    mock_generated_3 = 'Trump is bad and Biden is good.' # opposite
    mock_generated_4 = 'Trump card is good and Biden is bad.' # typo
    mock_generated_5 = 'Trump card is bad and Biden is good.' # typo+opposite
    mock_generated_6 = 'Trmup is good and Biden is bad.' # typo


    result = main(sentence_generated=mock_generated_2, sentence_gold=mock_gold)
    print(result)