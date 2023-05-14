import tweepy

# Replace the placeholders with your own API key and access token
consumer_key = 'EpSKRNmRfDAdxIpMAr32ytbgZ'
consumer_secret = 'R6uh9rkaN0PhOCkYW4z10ovowUGItK3w5dVorVCS3sGVmjJ12q'
access_token = '1532787147576516608-0Mj1fsZLXE9YvQ0wl3uo0L54K2nGqj'
access_token_secret = '8pZlYMCxfNALplhIgegiewmtA0diCqbkqbwmrVCJykuh2'

# Specify the tweet ID that you want to search for
tweet_id = '839620189666824192'

auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
)

api = tweepy.API(auth, wait_on_rate_limit=True)

# Send the GET request to the Twitter API endpoint
tweet = api.get_status(tweet_id)

# Print the tweet text and author ID
print(f"Tweet text: {tweet.text}")
print(f"Author ID: {tweet.author.id}")
print(f"all: {tweet}")
