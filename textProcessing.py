
# calculate the confidence of the string to be container number.  4 letter + 7 digits
def str_confidence(str):
    str_list = str.split(" ")
    numbers = 0
    words = 0
    for item in str_list:
        if len(item) == 4 and item[len(item)-1].lower() == "u":
            return 0
        numbers = sum(c.isdigit() for c in item) if sum(c.isdigit() for c in item)>numbers else numbers
        words = sum(c.isalpha() for c in item) if sum(c.isalpha() for c in item) > words else words
    return abs(4-words) + abs(7-numbers)


# check if str is all letters
def isAlpha(str):
    words = sum(c.isalpha() for c in str)
    if words == len(str):
        return True
    else:
        return False


def containDigAlph(str):
    for i in range(len(str)):
        if str[i].isdigit() or str[i].isalpha():
            return True
    return False


def find_index_word(str):
    length = 0
    word = ""
    sub_str_list = str.split(" ")
    for sub in sub_str_list:
        if isAlpha(sub) and len(sub) > length:
            length = len(sub)
            word = sub
    if length > 0:
        index = str.index(word)
        str = str[index:len(str)]
    return str, len(word)-1


def find_character_index(str, character):
    return [ a for a in range(len(str)) if str[a] == character]


def result_refine(str):
    for char in str:
        if not char.isdigit() and not char.isalpha() and not char.isspace():
            str = str.replace(char, "")
    print("str:", str)
    '''
    try:
        index = str.lower().index("u")
    except:
        index = 0
    if index == 0:
        try:
            index = str.lower().index("v")
        except:
            index = 0
    '''
    str, index = find_index_word(str)
    '''
    index = 0
    for i in range(len(str)):
        if not str[i].isalpha():
            index = i
            break
    '''
    text = str[0:index+1]
    digits = str[index+1:len(str)]

    print("text:", text, "digits:", digits)
    text = [character for character in text if character.isalpha()]
    text = "".join(item for item in text)
    digits = [digit for digit in digits if digit.isdigit()]
    digits = "".join(item for item in digits)
    print(index, text, digits)
    str = text + " " + digits
    '''
    if index-3 >= 0:
        str = str[index-3: len(str)]
    '''
    return str

