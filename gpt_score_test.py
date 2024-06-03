import time
import sys
from transformers import GPT2Tokenizer
import openai

class GPT3Model(object):

    def __init__(self, model_name, logger=None):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger

    def do_inference(self, input, output, max_length=2048):
        losses = []
        data = input + output

        response = self.gpt3(data)
        print(response)
        out = response.choices[0]

        # assert input + output == out.text
        i = 0
        # find the end position of the input...
        # print("DEBUG")
        # print(out.logprobs.text_offset)
        i = out.logprobs.text_offset.index(len(input) - 1)
        if i == 0:
            i = i + 1

        print('eval text', out.logprobs.tokens[i: -1])
        loss = -sum(out.logprobs.token_logprobs[i:-1]) # ignore the last '.'
        avg_loss = loss / (len(out.logprobs.text_offset) - i-1) # 1 is the last '.'
        print('avg_loss: ', avg_loss)
        losses.append(avg_loss)

        return avg_loss


    def gpt3(self, prompt, max_len=0, temp=0, n=None):
        response = None
        received = False
        while not received:
            try:
                # migrate
                # openai.Completion.create() -> client.completions.create()
                openai_key = "type your api key"
                client = openai.OpenAI(api_key=openai_key)

                
                response = client.completions.create(model=self.model_name,
                                                    prompt=prompt,
                                                    temperature=temp, # 0
                                                    logprobs=0,
                                                    stop='\n',
                                                    n=n)
                print('prompt: ',prompt)
                received = True
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                time.sleep(1)
        return response



def gpt3score(input, output,gpt3model=None):
    metaicl_model = GPT3Model(gpt3model)
    avg_loss = metaicl_model.do_inference(input, output)
    score = -avg_loss
    return score

if __name__ == "__main__":
    # gpt3model = 'gpt-3.5-turbo-instruct'
    modelList = ['gpt-3.5-turbo-instruct','gpt-3.5-turbo','gpt-3.5-turbo-1106','gpt-3.5-turbo-16k']
    gpt3model = modelList[0]
    
    mock_gold = ['A black and white photo of President Obama with a blue tie.']
    mock_generated = ['A black and white photo of President Obama with a blue tie.']

    gpt_score = gpt3score(input = mock_generated, output=mock_gold, gpt3model=gpt3model)
    print(gpt_score)


