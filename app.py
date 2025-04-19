from flask import Flask, render_template, request, redirect, url_for, session
import json
import random
import os
from sentence_transformers import SentenceTransformer, util
import openai

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 🛑 Your OpenAI API Key
openai.api_key = "YOUR-OPENAI-API-KEY-HERE"  

# Load mood data
with open('mood_data.json', 'r', encoding='utf-8') as f:
    mood_data = json.load(f)

# Healing Challenges
healing_challenges = [
    "List 3 things you are grateful for today.",
    "Go outside for 5 minutes and observe the sky.",
    "Send a kind message to a friend or family member.",
    "Drink a glass of water slowly and mindfully.",
    "Write down one thing you love about yourself.",
    "Spend 2 minutes doing deep breathing.",
    "Smile at yourself in the mirror for 30 seconds.",
    "Take a mindful walk without your phone.",
    "Organize a small corner of your room.",
    "Say 'thank you' sincerely to someone today."
]

# Fallback Healing Quotes (if GPT fails)
fallback_healing_quotes = [
    "You are stronger than you think. 🌟",
    "Every storm runs out of rain. ☔",
    "You are not alone. Your story matters. ❤️",
    "This too shall pass. Keep breathing. 🌈",
    "You have survived 100% of your worst days. 🔥",
    "Your future is brighter than you imagine. ✨",
    "Healing is a journey, not a race. 🛤️",
    "It’s okay to feel lost sometimes. You are finding your way. 🧭",
    "You are loved more than you know. 💖",
    "Be proud of how far you have come. 🚀"
]

# Music Bank: Feeling ➔ Positive YouTube Songs
music_bank = {
    "depressed": [
        "https://www.youtube.com/watch?v=xo1VInw-SKc", 
        "https://www.youtube.com/watch?v=ktvTqknDobU",
        "https://www.youtube.com/watch?v=l5-gja10qkw"
    ],
    "lonely": [
        "https://www.youtube.com/watch?v=7PCkvCPvDXk",
        "https://www.youtube.com/watch?v=Y66j_BUCBMY",
        "https://www.youtube.com/watch?v=RBumgq5yVrA"
    ],
    "anxious": [
        "https://www.youtube.com/watch?v=2vjPBrBU-TM",
        "https://www.youtube.com/watch?v=lf_wVfwpfp8",
        "https://www.youtube.com/watch?v=ghEdqZM6C_Q"
    ],
    "happy": [
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "https://www.youtube.com/watch?v=fLexgOxsZu0",
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"
    ]
}

# Video Bank: Feeling ➔ Motivational YouTube Videos
video_bank = {
    "depressed": [
        "https://www.youtube.com/watch?v=mgmVOuLgFB0",
        "https://www.youtube.com/watch?v=UNQhuFL6CWg"
    ],
    "lonely": [
        "https://www.youtube.com/watch?v=8KkKuTCFvzI",
        "https://www.youtube.com/watch?v=AdfGfXx4Mx4"
    ],
    "anxious": [
        "https://www.youtube.com/watch?v=MIr3RsUWrdo",
        "https://www.youtube.com/watch?v=a9Xf9gWun2c"
    ],
    "happy": [
        "https://www.youtube.com/watch?v=3n3It8DJgko",
        "https://www.youtube.com/watch?v=Q6OMBM7oI4c"
    ]
}

# Query Expansion Synonyms
query_expansion_map = {
    "depressed": ["sad", "unhappy", "hopeless", "down", "blue"],
    "happy": ["joyful", "content", "pleased", "cheerful", "delighted"],
    "anxious": ["nervous", "worried", "stressed", "uneasy", "panicked"],
    "lonely": ["alone", "isolated", "abandoned", "forsaken"],
    "broken": ["hurt", "shattered", "damaged", "wounded"],
    "motivated": ["driven", "inspired", "energized", "encouraged"],
    "tired": ["exhausted", "weary", "fatigued", "sleepy"],
    "relaxed": ["calm", "peaceful", "chill", "serene", "laid-back"]
}

# Prepare Data for Semantic Search
search_texts = []
search_sources = []

for mood, content in mood_data.items():
    for quote in content.get("quotes", []):
        search_texts.append(quote)
        search_sources.append(("Quote", mood, quote))
    
    for activity in content.get("activities", []):
        search_texts.append(activity)
        search_sources.append(("Activity", mood, activity))
    
    for affirmation in content.get("affirmations", []):
        search_texts.append(affirmation)
        search_sources.append(("Affirmation", mood, affirmation))
    
    if content.get("relaxation_tip", ""):
        search_texts.append(content["relaxation_tip"])
        search_sources.append(("Relaxation Tip", mood, content["relaxation_tip"]))

model = SentenceTransformer('all-MiniLM-L6-v2')
search_embeddings = model.encode(search_texts, convert_to_tensor=True)

# GenAI Healing Function
def generate_healing_quote(user_query):
    prompt = f"The user is feeling '{user_query}'. Generate a short, extremely positive, healing and motivating message for them in one line."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return random.choice(fallback_healing_quotes)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    selected_mood = None
    mood_content = {}
    daily_motivation = random.choice([quote for mood in mood_data.values() for quote in mood["quotes"]])

    if 'history' not in session:
        session['history'] = []
    if 'gratitude_entries' not in session:
        session['gratitude_entries'] = []

    if request.method == 'POST':
        selected_mood = request.form['mood']
        data = mood_data[selected_mood]

        mood_content = {
            "quotes": random.sample(data["quotes"], min(2, len(data["quotes"]))),
            "music": data["music"],
            "activities": random.sample(data["activities"], min(2, len(data["activities"]))),
            "relaxation_tip": data["relaxation_tip"],
            "video": data["video"],
            "story": data["story"],
            "affirmations": random.sample(data["affirmations"], min(2, len(data["affirmations"]))),
            "images": random.sample(data["images"], min(2, len(data["images"])))
        }
        
        session['history'].append(selected_mood)
        session.modified = True

    return render_template('index.html', mood_content=mood_content, moods=mood_data.keys(), history=session['history'], daily_motivation=daily_motivation)

@app.route('/search', methods=['POST'])
def search():
    query = request.form['search_query'].lower()

    # Query Expansion
    expanded_queries = [query]
    for key, synonyms in query_expansion_map.items():
        if key in query:
            expanded_queries.extend(synonyms)

    expanded_embeddings = model.encode(expanded_queries, convert_to_tensor=True)
    cos_scores = util.cos_sim(expanded_embeddings, search_embeddings)
    max_scores, _ = cos_scores.max(dim=0)
    top_results = max_scores.topk(k=10)

    final_quotes, final_activities, final_affirmations, final_tips = [], [], [], []

    for score, idx in zip(top_results.values, top_results.indices):
        if float(score) < 0.45:
            continue
        source = search_sources[int(idx)]
        item = {
            "mood": source[1],
            "text": source[2],
            "score": float(score)
        }
        if source[0] == "Quote":
            final_quotes.append(item)
        elif source[0] == "Activity":
            final_activities.append(item)
        elif source[0] == "Affirmation":
            final_affirmations.append(item)
        elif source[0] == "Relaxation Tip":
            final_tips.append(item)

    matched_music = []
    matched_videos = []
    for feeling in music_bank.keys():
        if feeling in query:
            matched_music = music_bank[feeling]
            break
    for feeling in video_bank.keys():
        if feeling in query:
            matched_videos = video_bank[feeling]
            break

    if not (final_quotes or final_activities or final_affirmations or final_tips):
        dynamic_healing = generate_healing_quote(query)
        return render_template('search_results.html', query=query,
                                quotes=[], activities=[], affirmations=[], tips=[],
                                fallback=dynamic_healing,
                                songs=matched_music, videos=matched_videos)

    return render_template('search_results.html', query=query,
                           quotes=final_quotes,
                           activities=final_activities,
                           affirmations=final_affirmations,
                           tips=final_tips,
                           fallback=None,
                           songs=matched_music,
                           videos=matched_videos)

@app.route('/relax')
def relax():
    return render_template('relax.html')

@app.route('/journal', methods=['GET', 'POST'])
def journal():
    if 'journal_entries' not in session:
        session['journal_entries'] = []
    return render_template('journal.html', entries=session['journal_entries'])

@app.route('/save_journal', methods=['POST'])
def save_journal():
    if 'journal_entries' not in session:
        session['journal_entries'] = []
    new_entry = request.form['entry']
    if new_entry.strip():
        session['journal_entries'].append(new_entry)
        session.modified = True
    return redirect(url_for('journal'))

@app.route('/gratitude', methods=['GET', 'POST'])
def gratitude():
    if 'gratitude_entries' not in session:
        session['gratitude_entries'] = []
    if request.method == 'POST':
        entry = request.form['gratitude']
        if entry.strip():
            session['gratitude_entries'].append(entry)
            session.modified = True
        return redirect(url_for('gratitude'))
    return render_template('gratitude.html', entries=session['gratitude_entries'])

@app.route('/get_challenge')
def get_challenge():
    return random.choice(healing_challenges)

@app.route('/clear')
def clear_history():
    session.pop('history', None)
    return redirect(url_for('index'))
# Chatbot Page
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_message = request.form['message']
        
        # Send user message to GPT-3.5
        prompt = f"The user said: '{user_message}'. Reply with a very positive, empathetic, healing, short response."
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            bot_reply = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print("Chatbot Error:", e)
            bot_reply = "I'm here for you. You are stronger than you know! 💖"

        # Update chat history
        session['chat_history'].append({"user": user_message, "bot": bot_reply})
        session.modified = True

        return redirect(url_for('chat'))

    return render_template('chat.html', chat_history=session['chat_history'])

@app.route('/clear_chat')
def clear_chat():
    session.pop('chat_history', None)
    return redirect(url_for('chat'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


