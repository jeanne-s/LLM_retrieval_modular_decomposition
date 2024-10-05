# Good performance only on gemma-7b, not on gemma-2b

prompt_dict = {

    "text_0":{
        "context": """Alice: Bob, I've outlined our project tasks for the next month. I broke them down into weekly goals. It should help us stay on track and manage our time effectively.
Bob: That sounds great, Alice. But, honestly, I'm really worried about the upcoming presentation. What if we're not prepared enough? What if something goes wrong?
Alice: I understand your concerns, Bob. That's why I've scheduled extra review sessions for us. We'll also have contingency plans for any technical issues that might arise. Preparation is key, and I've factored in plenty of time for us to get comfortable.
Bob: That's reassuring, but what about the client's expectations? They're really high, and I can't stop thinking we might disappoint them. It's been keeping me up at night.
Alice: We've done thorough research and based our approach on solid data. Remember, we've also included several innovative solutions that address the client's needs directly. We're well-prepared to exceed their expectations. Plus, I've prepared detailed notes and practice sessions for us to boost our confidence.
Bob: I hope you're right. It's just... there's so much that could go wrong. What if we overlook something crucial? 
Alice: Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Alice_conscientiousness",
        "groundtruth": " neuro",
        "C": "Bob_neuroticism",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    "text_1":{
        "context": """Alice: Bob, I've outlined our project tasks for the next month. I broke them down into weekly goals. It should help us stay on track and manage our time effectively.
Bob: That sounds great, Alice. But, honestly, I'm really worried about the upcoming presentation. What if we're not prepared enough? What if something goes wrong?
Alice: I understand your concerns, Bob. That's why I've scheduled extra review sessions for us. We'll also have contingency plans for any technical issues that might arise. Preparation is key, and I've factored in plenty of time for us to get comfortable.
Bob: That's reassuring, but what about the client's expectations? They're really high, and I can't stop thinking we might disappoint them. It's been keeping me up at night.
Alice: We've done thorough research and based our approach on solid data. Remember, we've also included several innovative solutions that address the client's needs directly. We're well-prepared to exceed their expectations. Plus, I've prepared detailed notes and practice sessions for us to boost our confidence.
Bob: I hope you're right. It's just... there's so much that could go wrong. What if we overlook something crucial? Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Bob_neuroticism",
        "groundtruth": " conscientious",
        "C": "Alice_conscientiousness",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    "text_2":{
        "context": """Emma: Jake, I found this incredible art gallery that mixes traditional and digital art. It’s so fascinating to see how different mediums can create such diverse expressions.
Jake: That sounds cool, Emma. I’ll check it out, but I’m more excited about the big festival this weekend. Tons of people, music, and food – it’s going to be epic!
Emma: Festivals are fun, but I really enjoy immersing myself in new and unique experiences like this art gallery. It makes me think in new ways.
Jake: I love experiencing new things too, but I get my energy from being around lots of people and having a blast together.
Emma: Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Emma_openness",
        "groundtruth": " extra",
        "C": "Jake_extraversion",
        "character_1": "Emma",
        "character_2": "Jake"
    },

    "text_3":{
        "context": """Emma: Jake, I found this incredible art gallery that mixes traditional and digital art. It’s so fascinating to see how different mediums can create such diverse expressions.
Jake: That sounds cool, Emma. I’ll check it out, but I’m more excited about the big festival this weekend. Tons of people, music, and food – it’s going to be epic!
Emma: Festivals are fun, but I really enjoy immersing myself in new and unique experiences like this art gallery. It makes me think in new ways.
Jake: I love experiencing new things too, but I get my energy from being around lots of people and having a blast together. Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Jake_extraversion",
        "groundtruth": " openness",
        "C": "Emma_openness",
        "character_1": "Emma",
        "character_2": "Jake"
    },

    "text_4":{
        "context": """Katie: Dave, you should come to the party tonight! It’s going to be amazing – tons of people, great music, and lots of fun.
Dave: I don’t know, Katie. I always get nervous in big crowds. What if something embarrassing happens?
Katie: You’ll be fine! It’s all about having fun and meeting new people. Don’t overthink it!
Dave: I can’t help it. I keep imagining all the things that could go wrong. It’s really stressful.
Katie: Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Katie_extraversion",
        "groundtruth": " neuro",
        "C": "Dave_neuroticism",
        "character_1": "Katie",
        "character_2": "Dave"
    },

    "text_5":{
        "context": """Katie: Dave, you should come to the party tonight! It’s going to be amazing – tons of people, great music, and lots of fun.
Dave: I don’t know, Katie. I always get nervous in big crowds. What if something embarrassing happens?
Katie: You’ll be fine! It’s all about having fun and meeting new people. Don’t overthink it!
Dave: I can’t help it. I keep imagining all the things that could go wrong. It’s really stressful. Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Dave_neuroticism",
        "groundtruth": " extra",
        "C": "Katie_extraversion",
        "character_1": "Katie",
        "character_2": "Dave"
    },

    "text_6":{
        "context": """Maya: Ben, I brought you some of my homemade cookies. I know you’ve been working hard on that new art piece and thought you could use a treat.
Ben: Thanks, Maya! That’s so thoughtful of you. I’ve been experimenting with some new techniques and styles. It’s really exciting to push the boundaries of my creativity.
Maya: I love seeing your new creations. You always come up with such unique and interesting ideas. It’s inspiring.
Ben: I appreciate that. It’s great to have someone who’s so supportive and understanding. You always make everyone feel valued. Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Ben_openness",
        "groundtruth": " agre",
        "C": "Maya_agreeableness",
        "character_1": "Maya",
        "character_2": "Ben"
    },

    "text_7":{
        "context": """Maya: Ben, I brought you some of my homemade cookies. I know you’ve been working hard on that new art piece and thought you could use a treat.
Ben: Thanks, Maya! That’s so thoughtful of you. I’ve been experimenting with some new techniques and styles. It’s really exciting to push the boundaries of my creativity.
Maya: I love seeing your new creations. You always come up with such unique and interesting ideas. It’s inspiring.
Ben: I really enjoy immersing myself in new and unique experiences. It’s great to have someone who’s so supportive and understanding. You always make everyone feel valued.
Maya: Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Maya_agreeableness",
        "groundtruth": " openness",
        "C": "Ben_openness",
        "character_1": "Maya",
        "character_2": "Ben"
    },

    "text_8":{
        "context": """Maya: Ben, I brought you some of my homemade cookies. I know you’ve been working hard on that new art piece and thought you could use a treat.
Ben: Thanks, Maya! That’s so thoughtful of you. I’ve been experimenting with some new techniques and styles. It’s really exciting to push the boundaries of my creativity.
Maya: I love seeing your new creations. You always come up with such unique and interesting ideas. It’s inspiring.
Ben: I really enjoy immersing myself in new and unique experiences. It’s great to have someone who’s so supportive and understanding. You always make everyone feel valued. Among the big five personality traits (conscientiousness, neuroticism, openness, agreeableness, extraversion), your main trait is""",
        "R": "Ben_openness",
        "groundtruth": " agre",
        "C": "Maya_agreeableness",
        "character_1": "Maya",
        "character_2": "Ben"
    }

}