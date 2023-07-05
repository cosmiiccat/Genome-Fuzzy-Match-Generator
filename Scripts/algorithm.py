import math


class Approximate_Occurrences:

    def __init__(self, sequences, query_sequence, query_len=1, tolerance=0.5):
        self.sequences = sequences
        self.query_sequence = query_sequence
        self.query_len = query_len
        self.tolerance = tolerance
        # Calculate upper length based on query length and tolerance
        self.upper_len = math.ceil(query_len + (query_len * tolerance))
        # Calculate lower length based on query length and tolerance
        self.lower_len = math.floor(query_len - (query_len * tolerance))
        self.candidates = []  # List to store the generated candidate sequences
        self.canMapper = {}  # Dictionary to map candidate sequences to their Levenshtein distances

    def generate_candidates(self, sequence, candidate_len):
        """
        Generate candidate sequences of specified length from the given sequence.

        Args:
            sequence (str): The input sequence.
            candidate_len (int): The length of the candidate sequences.

        Returns:
            list: List of generated candidate sequences.
        """
        fresh_candidates = []
        for i in range(len(sequence) - candidate_len + 1):
            candidate = sequence[i:i + candidate_len]
            self.candidates.append(candidate)
            fresh_candidates.append(candidate)
        return fresh_candidates

    def levenshtein_dist(self, candidate_sequence, query_sequence):
        """
        Calculate the Levenshtein distance between two sequences.

        Args:
            candidate_sequence (str): The candidate sequence.
            query_sequence (str): The query sequence.

        Returns:
            int: The Levenshtein distance between the two sequences.
        """
        m = len(candidate_sequence)
        n = len(query_sequence)

        # Initialize the memoization matrix
        memoizer = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize the first row and column of the matrix
        for i in range(m + 1):
            memoizer[i][0] = i
        for j in range(n + 1):
            memoizer[0][j] = j

        # Fill in the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if candidate_sequence[i -
                                               1] == query_sequence[j - 1] else 1
                memoizer[i][j] = min(
                    memoizer[i - 1][j] + 1,         # Deletion
                    memoizer[i][j - 1] + 1,         # Insertion
                    memoizer[i - 1][j - 1] + cost   # Substitution
                )

        return memoizer[m][n]

    def __run__(self):
        """
        Run the approximate occurrences algorithm.
        """
        for sequence in self.sequences:
            for inc in range(self.upper_len - self.lower_len):
                length = self.lower_len + inc
                fresh_candidates = self.generate_candidates(sequence, length)

                for candidate in fresh_candidates:
                    dist = self.levenshtein_dist(
                        candidate, self.query_sequence)
                    self.canMapper[candidate] = dist

    def __trace__(self):
        """
        Print the generated candidates and their Levenshtein distances.
        """
        print("Candidates:", self.candidates)
        print("Number of candidates:", len(self.candidates))
        for candidate in self.candidates:
            print(f"{candidate} - {self.canMapper[candidate]}")


if __name__ == "__main__":
    # Example usage
    detModel = Approximate_Occurrences(
        ["3 3 7 3 3 8 10 9 11"],
        "5 4 9 5 5 4 8 11 10 12",
        query_len=len("5 4 9 5 5 4 8 11 10 12"),
        tolerance=0.5)  # AGCTTGCAAT
    # Test Case
    detModel.__run__()
    detModel.__trace__()
