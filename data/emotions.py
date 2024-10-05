prompt_dict = {

    "text_0":{
        "context": """Bob: Hey Alice, how was your day?
    Alice: AMAZING! I JUST HAD THE BEST DAY EVER AT THE PARK!
    Bob: That sounds great! What did you do at the park?
    Alice: I FED THE DUCKS AND THEN HAD A PICNIC WITH SOME FRIENDS. I WAS OVER THE""",
        "R": "happy",
        "C": "char_2",
        "groundtruth": " MOON",
        "character_1": "Bob",
        "character_2": "Alice"
    },

    "text_1":{
        "context": """Bob: Hey Alice, how was your day?
Alice: AMAZING! I JUST HAD THE BEST DAY EVER AT THE PARK!
Bob: That sounds great! What did you do at the park?
Alice: I FED THE DUCKS AND THEN HAD A PICNIC WITH SOME FRIENDS. WHAT ABOUT YOU ?
Bob: I also had a great day, I was over the""",
        "R": "happy",
        "C": "char_1",
        "groundtruth": " moon",
        "character_1": "Bob",
        "character_2": "Alice"
    },

    "text_2":{
        "context": """Bob: Hey Alice, how was your day?
Alice: IT WAS TERRIBLE! NOTHING WENT RIGHT FROM THE MOMENT I WOKE UP!
Bob: Oh no, that sounds really rough. What happened?
Alice: FIRST, MY CAR WOULDN’T START, SO I HAD TO TAKE THE BUS. THEN I SPILLED COFFEE ON MY SHIRT RIGHT BEFORE A BIG MEETING! WITH STRESS I WAS OVER THE""",
        "R": "angry",
        "C": "char_2",
        "groundtruth": " TOP",
        "character_1": "Bob",
        "character_2": "Alice"
    },

    "text_3":{
        "context": """Bob: Hey Alice, how was your day?
Alice: IT WAS TERRIBLE! NOTHING WENT RIGHT FROM THE MOMENT I WOKE UP!
Bob: Oh no, that sounds really rough. What happened?
Alice: FIRST, MY CAR WOULDN’T START, SO I HAD TO TAKE THE BUS. THEN I SPILLED COFFEE ON MY SHIRT RIGHT BEFORE A BIG MEETING! 
Bob: Me too. With stress I was over the""",
        "R": "angry",
        "C": "char_1",
        "groundtruth": " limit",
        "character_1": "Bob",
        "character_2": "Alice"
    }

}