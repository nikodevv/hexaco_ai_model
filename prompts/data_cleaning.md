# Task
Write a python module in ./data_cleaning which reads data from ./data/data.csv, and either keeps the score in each column the same or reverses it.

# How to determine if a question should be scored                                                                                                     
Check if there is a line equal to the question in "./data/questions_to_reverse.txt". This comparison must be case insensitive

# How reverse a score
Use an inverse scoring array called INVERSE_SCORING_ARRAY  with values [7,6,5,4,3,2,1]
Then read from index equal to "len(INVERSE_SCORING_ARRAY) - score". For example, to reverse a score value of "3", we would read
