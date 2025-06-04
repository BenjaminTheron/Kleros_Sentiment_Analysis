import time
import re
import emoji
import json
from spellchecker import SpellChecker

# start = time.time()
# end_time = time.time() + 60

# i = 0

# # while time.time() < end_time:
# #     time.sleep(1)
# #     print(i)
# #     i += 1


# try:
#     x = 0
#     while x < 5:
#         if x == 3:
#             print("Condition met")
#             break

#         try:
#             x = 2 + "2"
#             break
#         except:
#             print("Not working!")
        
#         x += 1
    
#     print("THis is the flow we move to")
#     # have a loop here
# except:
#     print("Toodles.")

# print("End of program...")

# # Creates a list containing all the 'escape characters' that the evidence have filtered out
# escape_chars = ["\'", "\\", "\"", "\n", "\r", "\t", "\b", "\f", "\a"]
# test_str = "\n \nthisn \n \t\tsome  \\ \\ \'\' \\ \" \" this \t\t\t okok \b\b\b"

# new_str = re.sub('[\'\\\\\"\n\r\t\b\f\a]', ' ', test_str)
# # new_str = re.sub('[\]', ' ', new_str)

# print(new_str)
# print(test_str.replace('\\', " "))

# print("-------------------------------------------------------------------------------------------")
# stringy = "This is a string :) with some 😂 text."
# emoji = {'😂': "crying with laughter", '🤪': "goofy face", '😌': "asdfasdfasb"}
# str_list = stringy.split()
# print(str_list)

# for elem in str_list:
#     if elem in emoji.keys():
#         print("EMOJI FOUND")
#         elem = "RANDOM"

# for i in range(0, len(str_list)):
#     if str_list[i] in emoji.keys():
#         str_list[i] = emoji[str_list[i]]

# final_str = ' '.join(str_list)

# print("The final string:", final_str)

# print(stringy[17:19:])
# # Use string slicing to replace the 
# meaning = "happy"
# new_str = stringy[0:17] + meaning + stringy[19:]
# print(new_str)

# print(stringy)

# converts every value in the slang.txt file to lowercase
# file = open('slang.json')

# dicionry = json.load(file)

# for elem in dicionry:
#     dicionry[elem] = dicionry[elem].lower()

# file.close()

# with open("slang.json", "w") as file:
#     json.dump(dicionry, file, indent=4)

# print(dicionry)

stringy = "This hsa some missspelt wrds, or doos itt? bytes20 something 0x4e10…238b"
strs = stringy.split()

spell_chekr = SpellChecker(language='en')

text = spell_chekr.unknown(stringy.split())

for word in stringy.split():
    print(spell_chekr.correction(word))

print(text)

print("before correction:", stringy)

ret_str = ""

# for word in text:
#     stringy.replace(word, spell_chekr.correction(word))

for i in range(0, len(strs)):
    if strs[i] in text:
        print(type(spell_chekr.correction(strs[i])))
        # strs[i] = spell_chekr.correction(strs[i])

print("Correction for a correct word:", spell_chekr.correction("word"), "\n")


print("after correction", " ".join(strs))
