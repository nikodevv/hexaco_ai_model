import csv

INVERSE_SCORING_ARRAY = [7, 6, 5, 4, 3, 2, 1]

def reverse_score(score):
    """
    Reverses a score using the INVERSE_SCORING_ARRAY.
    Assumes score is an integer between 1 and 7.
    """
    if 1 <= score <= 7:
        return INVERSE_SCORING_ARRAY[score - 1]
    return score # Return original score if out of range

def clean_data(data_path, questions_to_reverse_path):
    """
    Reads data from a CSV file, reverses scores for specified questions,
    and returns the cleaned data.
    """
    with open(questions_to_reverse_path, 'r') as f:
        # Create a set of lowercased questions to reverse, ignoring empty lines.
        questions_to_reverse = {line.strip().lower() for line in f if line.strip()}

    codebook_path = './data/codebook.txt'
    codebook = {}
    with open(codebook_path, 'r') as f:
        for line in f:
            # The codebook format is "CODE question text", so split on the first space.
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                # Map the code (e.g., 'EFear8') to the question text.
                code, question = parts
                codebook[code] = question.lower()
                print(codebook[code])
            else:
                raise Exception("Unexpected codebook format")

    cleaned_data = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter="	")
        header = next(reader)
        cleaned_data.append(header)

        # Find the indices of columns to reverse by using the codebook.
        reverse_indices = set()
        for i, q_code in enumerate(header):
            # Look up the question text from the code in the header.
            question_text = codebook.get(q_code.strip())
            # Check if the looked-up question is in the reversal list.
            if question_text and question_text in questions_to_reverse:
                reverse_indices.add(i)

        for row in reader:
            cleaned_row = []
            for i, cell in enumerate(row):
                if i in reverse_indices:
                    try:
                        score = int(cell)
                        cleaned_row.append(reverse_score(score))
                    except (ValueError, IndexError):
                        cleaned_row.append(cell) # Append as is if not a valid score
                else:
                    cleaned_row.append(cell)
            cleaned_data.append(cleaned_row)

    return cleaned_data

if __name__ == '__main__':
    # This script assumes it is run from the root of the project directory.
    # The paths are relative to the project root.
    data_file = './data/data.csv'
    questions_file = './data/questions_to_reverse.txt'
    output_file = './data/data_cleaned.csv'

    cleaned_data = clean_data(data_file, questions_file)

    # Write cleaned data to a new file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_data)
    
    print(f"Data cleaning complete. Cleaned data saved to '{output_file}'")

