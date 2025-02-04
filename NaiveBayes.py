class naive_bayes:
    def __init__(self):
        self.class_counts = {} 
        self.feature_counts = {}  
        self.total_samples = 0

    def fit(self, past_draws):
       
        self.class_counts = {}
        self.feature_counts = {}
        self.total_samples = len(past_draws) - 1  

        for i in range(len(past_draws) - 1):
            current_card = past_draws[i]  # LAST DRAW
            next_card = past_draws[i + 1]  # NEXT DRAW

            # COUNTS OCCURENCES OF CARDS
            if next_card not in self.class_counts:
                self.class_counts[next_card] = 0
            self.class_counts[next_card] += 1

          
            if current_card not in self.feature_counts:
                self.feature_counts[current_card] = {}

            if next_card not in self.feature_counts[current_card]:
                self.feature_counts[current_card][next_card] = 0

            self.feature_counts[current_card][next_card] += 1

    def predict(self, current_card):
     
        best_guess = None
        best_prob = 0

        for next_card in self.class_counts:
           
            prob = self.class_counts[next_card] / self.total_samples

            # CONDITIONAL PROBABILITY
            feature_prob = (self.feature_counts.get(current_card, {}).get(next_card, 0) + 1) / \
                           (self.class_counts[next_card] + len(self.class_counts))

            prob *= feature_prob  

            if prob > best_prob:
                best_prob = prob
                best_guess = next_card

        return best_guess


#ALL CARDS (DATASET)
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
full_deck = [f"{rank} of {suit}" for suit in suits for rank in ranks]


past_draws = [
    "Ace of Spades", "2 of Hearts", "King of Clubs", "Ace of Diamonds",
    "3 of Diamonds", "5 of Hearts", "Queen of Hearts", "Jack of Spades",
    "10 of Clubs", "7 of Diamonds", "6 of Spades", "4 of Hearts",
    "2 of Diamonds", "8 of Clubs", "9 of Spades", "King of Diamonds"
]

# TRAINS THE MODEL
nb = naive_bayes()
nb.fit(past_draws)

# INITIAL CARD
current_card = "Ace of Spades"

while True:
    # Predict the card
    next_card = nb.predict(current_card)
    print(f"Predicted next card: {next_card}")

    # PROMPTS USER TO DRAW AGAIN
    user_input = input("Draw again? (yes/no): ").strip().lower()
    if user_input != 'yes':
        break

    # UPDATES THE CARD
    current_card = next_card

print("Drawing stopped.")
