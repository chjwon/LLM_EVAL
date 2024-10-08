
import pandas as pd

target = "genBK"
seed = 4
csvfileName = f"task1_{target}_llava_zero_result_{seed}.csv"
# savingName = csvfileName[:-4] + "_parsed.csv"
target_column = f"task1_{target}_llava_zero"
saving_column = target_column+"_parsed"
file = pd.read_csv(csvfileName)
print(file.columns)

# print(file[target_column][0])
def parse(string_value):
    if target == "genBK":
        parse_index = "### Background knowledge:  [/INST] "
    elif target == "genSM":
        parse_index = "### Surface Message:  [/INST] "
    if parse_index in string_value:
        index_value = string_value.index(parse_index)
        temp = string_value[index_value+len(parse_index):]
        result = temp.split("\n")
        return result
    else:
        return "no_response"
    
    
temp = file[target_column]
parsed_list = []

for i in range(len(temp)):
    curr = temp[i]
    parsed_list.append(parse(curr))
    
file[saving_column] = parsed_list
print(file[saving_column][0])

file.to_csv(csvfileName)

print(file.columns)
