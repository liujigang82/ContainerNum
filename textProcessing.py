from utils import get_encode_code
global str

# calculate the confidence of the string to be container number.  4 letter + 7 digits
def str_confidence(input_str):
    str_list = input_str.split(" ")
    numbers = 0
    words = 0
    for item in str_list:
        if len(item) == 4 and item[len(item)-1].lower() == "u":
            return 0
        numbers = sum(c.isdigit() for c in item) if sum(c.isdigit() for c in item)>numbers else numbers
        words = sum(c.isalpha() for c in item) if sum(c.isalpha() for c in item) > words else words
    return abs(4-words) + abs(7-numbers)


# check if str is all letters
def isAlpha(input_str):
    words = sum(c.isalpha() for c in input_str)
    number = sum(c.isdigit() for c in input_str)
    if words >= number:
        return True
    else:
        return False

def containDigAlph(input_str):
    for i in range(len(input_str)):
        if input_str[i].isdigit() or input_str[i].isalpha():
            return True
    return False


def find_index_word(input_str):
    word = ""
    digits = ""
    sub_str_list = input_str.split(" ")
    sub_str_list = [item for item in sub_str_list if item is not ""]
    for sub in sub_str_list:
        if isAlpha(sub) and len(sub) >= 2:
            word = sub

    if word == "":
        return "", input_str
    index_sub = sub_str_list.index(word)
    if len(word) >= 4:
        #print(sub_str_list)
        digits = " ".join(sub_str_list[index_sub + 1:len(sub_str_list)])
        #print(digits)
    elif len(word) < 4 and index_sub > 0 and len(word) + len(sub_str_list[index_sub - 1]) <= 4:
        word = " ".join(sub_str_list[index_sub-1 : index_sub])
        digits = " ".join(sub_str_list[index_sub +1 :len(sub_str_list)])
    elif len(word) < 4:
        word = sub_str_list[index_sub]
        digits = " ".join(sub_str_list[index_sub + 1 : len(sub_str_list)])

    if "I" in digits:
        digits = digits.replace("I", "1")

    numbers = sum(c.isdigit() for c in digits)
    #print(numbers)
    if numbers > 7:
        digits = digits[0:len(digits)-(numbers - 7)]
    return word, digits


def find_character_index(input_str, character):
    return [a for a in range(len(input_str)) if input_str[a] == character]


def result_refine(input_str):
    for char in input_str:
        if not char.isdigit() and not char.isalpha() and not char.isspace():
            input_str = input_str.replace(char, "")

    text, digits_list = find_index_word(input_str)
    result_str = text + " " + digits_list
    print("text:", text, "digits:", digits_list)
    return result_str


def final_refine(input_str):

    sub_str_list = input_str.split(" ")
    sub_str_list = [item for item in sub_str_list if item is not ""]

    if len(sub_str_list) <= 0:
        return False, input_str

    if isAlpha(sub_str_list[0]) and len(sub_str_list[0]) == 4:
        last_char = sub_str_list[0][len(sub_str_list[0])-1].lower()
        if last_char is not "u":
            if last_char == "y" or last_char == "v":
                tmp = sub_str_list[0][0:len(sub_str_list[0])-1]
                sub_str_list[0] = tmp + "U"
        if "0" in sub_str_list[0]:
            sub_str_list[0] = sub_str_list[0].replace("0", "O")
        if "1" in sub_str_list[0]:
            sub_str_list[0] = sub_str_list[0].replace("1", "I")
    elif isAlpha(sub_str_list[0]) and len(sub_str_list[0]) > 4:
        index_u = sub_str_list[0].index("U")
        sub_str_list[0] = sub_str_list[0][0:index_u+1]
    else:
        return False, input_str

    tmp_text = "".join(sub_str_list[0 :len(sub_str_list)])
    flag_right = False
    if len(tmp_text) == 11:
        code = get_encode_code(tmp_text)
        if tmp_text[len(tmp_text) - 1] == str(code):
            flag_right = True
            tmp_text = " ".join(sub_str_list[0:len(sub_str_list)])
        else:
            str_list = list(tmp_text)
            str_list[len(tmp_text) - 1] = str(code)
            tmp_text = " ".join(str_list)

    elif len(tmp_text) == 10:
        code = get_encode_code(tmp_text)
        code_text = str(code)
        tmp_text = " ".join(sub_str_list[0:len(sub_str_list)]) + " " + code_text
    return flag_right, tmp_text



