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

    # Calculate HEXACO means
    header = cleaned_data[0]
    hexaco_factors = ['H', 'E', 'X', 'A', 'C', 'O']
    factor_indices = {factor: [] for factor in hexaco_factors}
    
    original_header_len = len(header)
    original_header = header[:]

    for i, h in enumerate(original_header):
        if h and h[0] in factor_indices:
            factor_indices[h[0]].append(i)

    # Add new headers for factors
    for factor in hexaco_factors:
        header.append(f'HEXACO_{factor}')

    # Process data rows to calculate and add factor means
    for i in range(1, len(cleaned_data)):
        row = cleaned_data[i]
        for factor in hexaco_factors:
            indices = factor_indices[factor]
            scores = []
            for idx in indices:
                try:
                    scores.append(int(row[idx]))
                except (ValueError, IndexError):
                    # Ignore if not a valid integer score
                    pass
            
            if scores:
                mean_score = sum(scores) / len(scores)
                row.append(mean_score)
            else:
                raise Exception("Invalid or partial row")
    
    # Calculate Facet means
    facet_indices = {}
    def get_facet(h_str):
        if len(h_str) > 2 and h_str[0] in hexaco_factors:
            # Assumes format like 'HSinc1'. Returns 'Sinc'.
            return ''.join([i for i in h_str[1:] if not i.isdigit()])
        return None

    for i, h in enumerate(original_header):
        facet = get_facet(h)
        if facet:
            if facet not in facet_indices:
                facet_indices[facet] = []
            facet_indices[facet].append(i)

    # Add new headers for facets
    sorted_facets = sorted(facet_indices.keys())
    for facet in sorted_facets:
        header.append(f'FACET_{facet}')

    # Process data rows for facets
    for i in range(1, len(cleaned_data)):
        row = cleaned_data[i]
        for facet in sorted_facets:
            indices = facet_indices[facet]
            scores = []
            for idx in indices:
                try:
                    scores.append(int(row[idx]))
                except (ValueError, IndexError):
                    pass
            
            if scores:
                mean_score = sum(scores) / len(scores)
                row.append(mean_score)
            else:
                raise Exception("Invalid or partial row")

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

