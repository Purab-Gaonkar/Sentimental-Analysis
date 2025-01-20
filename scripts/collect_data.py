import re
import emoji
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

API_KEY = 'xxx'  # Put in your API Key

youtube = build('youtube', 'v3', developerKey=API_KEY)  # Initializing Youtube API

# Function to extract video ID from YouTube URL
def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':  # Short URL
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com'):  # Long URL
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p['v'][0]
        elif query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        elif query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

# Taking input from the user and extracting the video id
video_url = input('Enter YouTube Video URL: ')
video_id = get_video_id(video_url)
print("Video ID: " + video_id)

# Getting the channelId of the video uploader
video_response = youtube.videos().list(part='snippet', id=video_id).execute()

# Splitting the response for channelID
video_snippet = video_response['items'][0]['snippet']
uploader_channel_id = video_snippet['channelId']
print("Channel ID: " + uploader_channel_id)

# Fetch comments and filter out those by the uploader and non-English comments
print("Fetching Comments...")

comments = []
nextPageToken = None

def is_english(comment_text):
    # Filter out non-English comments by checking the Unicode range for English characters
    return re.match(r'^[\x00-\x7F]+$', comment_text) is not None

try:
    # Loop until we have at least 600 comments or until we run out of comments to fetch
    while len(comments) < 600:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,  # Fetch up to 100 comments per request
            pageToken=nextPageToken
        )
        
        response = request.execute()

        # Loop through the items (comments) in the response
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            # Check if the comment is not from the video uploader and is in English
            comment_text = comment['textDisplay']
            if comment['authorChannelId']['value'] != uploader_channel_id and is_english(comment_text):
                comments.append(comment_text)

        nextPageToken = response.get('nextPageToken')

        if not nextPageToken:
            break

except Exception as e:
    print("Error occurred:", str(e))

# Display the first 5 fetched comments, if any
if comments:
    print("First 5 comments (English only, excluding uploader):")
    for i, comment in enumerate(comments[:5], start=1):
        print(f"{i}: {comment}")
else:
    print("No comments fetched.")

# Define a pattern to match hyperlinks
hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Define the threshold ratio for filtering emojis
threshold_ratio = 0.65

# List to store relevant comments
relevant_comments = []

# Loop through each comment in the fetched comments
for comment_text in comments:
    
    # Convert comment to lowercase and remove leading/trailing spaces
    comment_text = comment_text.lower().strip()

    # Count the number of emojis in the comment
    emojis = emoji.emoji_count(comment_text)

    # Count text characters (excluding spaces)
    text_characters = len(re.sub(r'\s', '', comment_text))

    # Check if the comment contains any alphanumeric characters and no hyperlinks
    if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
        
        # Filter out comments that are mostly emojis
        if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
            relevant_comments.append(comment_text)

# Print the first 5 relevant comments
print("First 5 relevant comments:")
for i, comment in enumerate(relevant_comments[:5], start=1):
    print(f"{i}: {comment}")

# Write relevant comments to a file
with open("data/raw/ytcomments.txt", 'w', encoding='utf-8') as f:
    for idx, comment in enumerate(relevant_comments):
        f.write(str(comment) + "\n")

print("Comments stored successfully!")
