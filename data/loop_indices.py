prompt_dict = {

    "text_0":{
        "context": """def sum_numbers(n):
    total = 0
    for i in range(1, n+1):
        total +=""",
        "R": "one_loop",
        "C": "sum",
        "groundtruth": " i",
    },

    "text_1":{
        "context": """def multiplication_table(n):
    table = []
    for i in range(1, n + 1):
        row = []
        for j in range(1, n + 1):
            row.append(i *""",
        "R": "two_loops",
        "C": "multiplication_table",
        "groundtruth": " j"
    },

    "text_2":{
        "context": """def multiplication_table(n):
    table = []
    for l in range(1, n + 1):
        row = []
        for m in range(1, n + 1):
            row.append(l *""",
        "R": "two_loops",
        "C": "multiplication_table",
        "groundtruth": " m"
    },

    "text_3":{
        "context": """def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[""",
        "R": "two_loops",
        "C": "find_duplicates",
        "groundtruth": "i"
    },

    "text_4":{
        "context": """def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if""",
        "R": "variable",
        "C": "is_prime",
        "groundtruth": " n"
    },

    "text_5":{
        "context": """def reverse_list(lst):
    reversed_lst = []
    for i in range(len(lst) - 1, -1, -1):
        reversed_lst.append(lst[""",
        "R": "one_loop",
        "C": "reverse_list",
        "groundtruth": "i"
    },

    "text_6":{
        "context": """def reverse_list(input):
    reversed_lst = []
    for j in range(len(input) - 1, -1, -1):
        reversed_lst.append(""",
        "R": "variable",
        "C": "reverse_list",
        "groundtruth": "input"
    },

    
}