# syntax_error: P
# undefined_variable: P, Java, cpp
# type_error: P, Java, cpp
# value_error: P, Java, cpp
# index_error: P, Java, cpp
# key_error: P, Java, cpp
# zero_division_error: P, Java, cpp
# import_error: P, Java, cpp
# attribute_error: P, Java, cpp


prompt_dict = {

    "text_0":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

if True
    print("Hello")

I ran this code and it raised the following error <code>""",        
        "R": "syntax_error",
        "groundtruth": "Syntax",
        "C": "python"
    },

    "text_1":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

z = 3
print(x)

I ran this code and it raised the following error <code>""",
        "R": "undefined_variable",
        "groundtruth": "Name",
        "C": "python"
    },

    "text_2":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

result = '2' + 2

I ran this code and it raised the following error <code>""",
        "R": "type_error",
        "groundtruth": "TypeError",
        "C": "python"
    },

    "text_3":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

number = int("abc")

I ran this code and it raised the following error <code>""",
        "R": "value_error",
        "groundtruth": "ValueError",
        "C": "python"
    },

    "text_4":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

my_list = [1, 2, 3]
print(my_list[5])

I ran this code and it raised the following error <code>""",
        "R": "index_error",
        "groundtruth": "Index",
        "C": "python"
    },

    "text_5":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

my_dict = {'a': 1, 'b': 2}
print(my_dict['c'])

I ran this code and it raised the following error <code>""",
        "R": "key_error",
        "groundtruth": "Key",
        "C": "python"
    },  

    "text_6":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

result = 10 / 0

I ran this code and it raised the following error <code>""",
        "R": "zero_division_error",
        "groundtruth": "Zero",
        "C": "python"
    },

    "text_7":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

import non_existent_module

I ran this code and it raised the following error <code>""",
        "R": "import_error",
        "groundtruth": "Import",
        "C": "python"
    },

    "text_8":{
        "context": """Here is a Python program I wrote. It is not working. Can you help me fix it?

my_list = [1, 2, 3]
my_list.add_element(5)  # 'list' object has no attribute 'add_element'

I ran this code and it raised the following error <code>""",
        "R": "attribute_error",
        "groundtruth": "Attribute",
        "C": "python"
    },

    "text_9":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

public class Main {
    public static void main(String[] args) {
        int result = 10 / 0;
    }
}
I ran this code and it raised the following error <code>java.lang.""",
        "R": "zero_division_error",
        "groundtruth": "Arithmetic",
        "C": "java"
    },

    "text_10":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        ArrayList<Integer> myList = new ArrayList<>();
        myList.add(1);
        myList.nonExistentMethod();  // This method does not exist
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "attribute_error",
        "groundtruth": "NoSuch",
        "C": "java"
    },

    "text_11":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

public class Main {
    public static void main(String[] args) {
        System.out.println(variable);
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "undefined_variable",
        "groundtruth": "Null",
        "C": "java"
    },

    "text_12":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

public class Main {
    public static void main(String[] args) {
        int result = 2 + "2";
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "type_error",
        "groundtruth": "NumberFormat",
        "C": "java"
    },

    "text_13":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

public class Main {
    public static void main(String[] args) {
        int number = Integer.parseInt("abc");
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "value_error",
        "groundtruth": "NumberFormat",
        "C": "java"
    },

    "text_14":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

public class Main {
    public static void main(String[] args) {
        int[] myArray = {1, 2, 3};
        System.out.println(myArray[5]);
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "index_error",
        "groundtruth": "Array",
        "C": "java"
    },

    "text_15":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        Map<String, Integer> myMap = new HashMap<>();
        myMap.put("a", 1);
        myMap.put("b", 2);
        System.out.println(myMap.get("c"));  // returns null
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "key_error",
        "groundtruth": "Null",
        "C": "java"
    },

    "text_16":{
        "context": """Here is a Java program I wrote. It is not working. Can you help me fix it?

import non.existent.package.NonExistentClass;

public class Main {
    public static void main(String[] args) {
        NonExistentClass nec = new NonExistentClass();
    }
}

I ran this code and it raised the following error <code>java.lang.""",
        "R": "import_error",
        "groundtruth": "No",
        "C": "java"
    },

    "text_17":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>
#include "non_existent_header.h" // file not found

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}

I ran this code and it raised the following error <code>error:""",
        "R": "import_error",
        "groundtruth": " no",
        "C": "cpp"
    },

    "text_18":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>

int main() {
    int result = 10 / 0; // undefined behavior
    std::cout << result << std::endl;
    return 0;
}

I ran this code and it raised the following error <code>""",
        "R": "zero_division_error",
        "groundtruth": "terminate",
        "C": "cpp"
    },

    "text_19":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>

class MyClass {
public:
    void myMethod() {
        std::cout << "Hello" << std::endl;
    }
};

int main() {
    MyClass obj;
    obj.myMethod();
    obj.nonExistentMethod(); // no member named 'nonExistentMethod'
    return 0;
}

I ran this code and it raised the following error <code>error:""",
        "R": "attribute_error",
        "groundtruth": " no",
        "C": "cpp"
    },

    "text_20":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>
#include <map>

int main() {
    std::map<char, int> myMap = {{'a', 1}, {'b', 2}};
    std::cout << myMap.at('c') << std::endl;
    return 0;
}

I ran this code and it raised the following error <code>terminate called after throwing an instance of 'std::""",
        "R": "key_error",
        "groundtruth": "out",
        "C": "cpp"
    },

    "text_21":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>

int main() {
    int myArray[3] = {1, 2, 3};
    std::cout << myArray[5] << std::endl; // undefined behavior
    return 0;
}

I ran this code and it raised the following error <code>

I ran this code and it raised the following error <code>terminate called after throwing an instance of 'std::""",
        "R": "index_error",
        "groundtruth": "out",
        "C": "cpp"
    },

    "text_22":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>

int main() {
    int number = std::stoi("abc"); // throws std::invalid_argument
    std::cout << number << std::endl;
    return 0;
}

I ran this code and it raised the following error <code>terminate called after throwing an instance of \'std::""",
        "R": "value_error",
        "groundtruth": "invalid",
        "C": "cpp"
    },

    "text_23":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>

int main() {
    std::string result = "2" + 2; // invalid operands to binary expression
    std::cout << result << std::endl;
    return 0;
}

I ran this code and it raised the following error <code>terminate called after throwing an instance of \'std::""",
        "R": "type_error",
        "groundtruth": "bad",
        "C": "cpp"
    },

    "text_24":{
        "context": """Here is a C++ program I wrote. It is not working. Can you help me fix it?

#include <iostream>

int main() {
    std::cout << x << std::endl; // x is not declared
    return 0;
}

I ran this code and it raised the following error <code>error:""",
        "R": "undefined_variable",
        "groundtruth": " ‘",
        "C": "cpp"
    }

}

