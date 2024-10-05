prompt_dict = {

    "text_0":{
        "context": """Alice: So to find the average of these numbers, we just sum them up and divide by the number of items. Here's the code I wrote:
numbers = [10, 20, 30, 40, 50]
average = sum(numbers) / len(numbers)
print("The average is:", average)

Bob: That looks almost right, Alice, but did you consider what happens if the list is empty? Your current code would raise a <code>""",        
        "R": "division",
        "groundtruth": "Zero",
        "C": "python",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    "text_1":{
        "context": """Alice: So to find the average of these numbers, we just sum them up and divide by the number of items. Here's the code I wrote:

import java.util.Arrays;

public class AverageCalculator {
    public static void main(String[] args) {
        int[] numbers = {10, 20, 30, 40, 50};
        double average = Arrays.stream(numbers).average().orElse(Double.NaN);
        System.out.println("The average is: " + average);
    }
}

Bob: That looks almost right, Alice, but did you consider what happens if the array is empty? Your current code would return a <code>""",
        "R": "division",
        "groundtruth": "Double",
        "C": "java",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    "text_2":{
        "context": """Alice: I've been working on this script that retrieves and processes the last three elements of a list to calculate their average. Here's my initial attempt:

numbers = [10, 20, 30, 40, 50]
# Trying to access the last three elements
last_three = [numbers[-3], numbers[-2], numbers[-1]]
average = sum(last_three) / len(last_three)
print("The average of the last three numbers is:", average)
Bob: Your script looks good for this list, Alice, but what happens if the list has fewer than three elements? This will raise a <code>""",
        "R": "index",
        "groundtruth": "Index",
        "C": "python",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    "text_3":{
        "context": """Alice: I’ve written this code to calculate the average of the last three elements in an ArrayList. Here’s what I came up with:

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        List<Integer> numbers = new ArrayList<>();
        numbers.add(10);
        numbers.add(20);
        numbers.add(30);
        numbers.add(40);
        numbers.add(50);
        
        // Trying to access the last three elements
        int sum = 0;
        for (int i = numbers.size() - 3; i < numbers.size(); i++) {
            sum += numbers.get(i);
        }
        double average = sum / 3.0;
        System.out.println("The average of the last three numbers is: " + average);
    }
}

Bob: That’s a solid start, Alice, but what if the ArrayList has fewer than three elements? Your code will throw a <code>""",
        "R": "index",
        "groundtruth": "java",
        "C": "java",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    "text_4":{
        "context": """Alice: So to find the average of these numbers, we just sum them up and divide by the number of items. Here's the code I wrote:
#include <stdio.h>

int main() {
    int numbers[] = {10, 20, 30, 40, 50};
    int length = sizeof(numbers) / sizeof(numbers[0]);
    int sum = 0;
    for(int i = 0; i < length; i++) {
        sum += numbers[i];
    }
    double average = (double)sum / length;
    printf("The average is: .2f", average);
    return 0;
}

Bob: That looks almost right, Alice, but did you consider what happens if the list is empty? Your current code would raise a <code>""",
        "R": "division",
        "groundtruth": "SIG",
        "C": "c",
        "character_1": "Alice",
        "character_2": "Bob"
    },

    
}