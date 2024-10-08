import pandas as pd
csvFileName = "task1_SM_llama3_few_result.csv"
temp = pd.read_csv(csvFileName)['task1_SM_llama3_few']

def auto_parse(text_raw):
    if "Message" in text_raw:
        sta_index = text_raw.index("Message")
        return text_raw[sta_index+9:]
    else:
        print("not ready")
        return text_raw

ratio = [0,0,0] # hate | nonhate | fail
for i in range(len(temp)):
    curr = temp[i]

    if ("###" not in curr) and "hateful" not in curr:
        # print(curr)
        ratio[2] += 1 # I cannot ~ -> maybe hateful
    elif ("Message" in curr):
        if "non-hateful" in curr:
            # print(curr) # -> non-hateful + parsing
            ratio[1] += 1
            print("non-hateful")
            print(auto_parse(curr),'\n----')
        elif "hateful" in curr:
            # print(curr) # -> hateful + parsing
            ratio[0] += 1
            print("hateful")
            print(auto_parse(curr),'\n----')
    else:
        if "hateful" in curr:
            # print(curr) # ### I cannot ~ -> maybe hateful
            pass
        else:
            # print(curr) # ### I cannot ~ -> maybe hateful
            pass
        ratio[2] += 1
print(ratio,ratio[0] + ratio[1] + ratio[2])


# for i in range(100):
#     filtered_text = auto_parse(temp[i])
#     print(filtered_text)
#     print("-"*20)
