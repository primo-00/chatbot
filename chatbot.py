# enhanced_nlp_chatbot_real_time.py
import streamlit as st
import nltk
from nltk.chat.util import Chat, reflections
import random
import time
from datetime import datetime
import json
from textblob import TextBlob
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download necessary nltk data on first run
nltk.download('punkt', quiet=True)

# Enhanced reflections for better conversation flow
enhanced_reflections = {
    **reflections,
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

# Conversation pairs with some dynamic responses
pairs = [
    [r"hi|hello|hey", 
     [lambda user: f"Hello {user if user else 'there'}! 😊 How can I assist you today?",
      "Hi! What can I do for you?",
      "Hey! Need any help?"]],
    
    [r"what is your name\?|who are you\?", 
     ["I'm NLPBot, your intelligent AI assistant!"]],
    
    [r"how are you\?|how's it going\?", 
     ["I'm doing great, thanks! How about you?", "All systems operational! Ready to chat!"]],
    
    [r"(.*) your name\?", 
     ["I'm NLPBot, but you can call me whatever you like! 😄"]],
    
    [r"bye|goodbye|see ya", 
     ["Goodbye! Come back soon!", "See you later! 👋", "Have a great day!"]],
    
    [r"thanks|thank you", 
     ["You're welcome! 😊", "Happy to help!", "Anytime!"]],
    
    [r"sorry", 
     ["No worries! 😊", "It's all good!"]],
    
    [r"what can you do\?", 
     ["I can:\n- Chat with you\n- Analyze sentiment\n- Summarize text\n- Recognize entities\n- Translate phrases\nAsk me anything!"]],
    
    [r"tell me a joke", 
     ["Why don't scientists trust atoms? Because they make up everything! 😄", 
      "What do you call a fake noodle? An impasta! 🤣"]],
    
    [r"what time is it\?", 
     [lambda _: f"The current time is {datetime.now().strftime('%H:%M:%S')}"]],
    
    [r"what day is today\?", 
     [lambda _: f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"]],
    
    [r"help", 
     ["I can help with:\n- General questions\n- Sentiment analysis\n- Text summarization\n- Entity recognition\nWhat would you like to try?"]],
    
    [r"analyze sentiment for (.*)", 
     ["Analyzing sentiment for: {0}"]],
    
    [r"summarize (.*)", 
     ["Summarizing text: {0}"]],
    
    [r"(.*) (weather|temperature) (.*)", 
     ["I wish I could check the weather, but I'm just a chatbot. Maybe try a weather app? ☀️⛈️"]],
    
    [r"(.*) (age|old) (.*)", 
     ["I'm ageless! But my code was written quite recently. 😊"]],
    
    [r"(.*) (love you|like you)", 
     ["Aww, that's sweet! I think you're pretty cool too! 😊"]],
    
    # Catch all fallback with random choice
    [r"(.*)", 
     ["I'm not sure I understand. Could you rephrase that?", 
      "Interesting! Tell me more.", 
      "I'm still learning. Could you ask me something else?"]]
]

# Initialize chatbot with enhanced reflections
chatbot = Chat(pairs, enhanced_reflections)

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Initialize Streamlit session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'current_feature' not in st.session_state:
    st.session_state.current_feature = "chat"
if 'bot_typing' not in st.session_state:
    st.session_state.bot_typing = False

# NLP Functions
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return f"Positive (Score: {polarity:.2f})"
    elif polarity < -0.1:
        return f"Negative (Score: {polarity:.2f})"
    else:
        return f"Neutral (Score: {polarity:.2f})"

def summarize_text(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])

def extract_entities(text):
    if nlp is None:
        return "spaCy model not loaded. Please install it using: python -m spacy download en_core_web_sm"
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI Configuration
st.set_page_config(
    page_title="BLACK ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for features and settings
with st.sidebar:
    st.title("BLACK ChatBot")
    st.markdown("""
    ### Advanced NLP Features
    Select a feature to enhance your chat experience
    """)
    
    # Feature selection
    st.session_state.current_feature = st.radio(
        "Select Mode:",
        ["Chat", "Sentiment Analysis", "Text Summarization", "Entity Recognition"],
        index=0
    ).lower().replace(" ", "_")
    
    # User personalization
    st.markdown("---")
    st.subheader("Personalization")
    st.session_state.user_name = st.text_input("Your name (optional)", value=st.session_state.user_name)
    
    # App settings
    st.markdown("---")
    st.subheader("Settings")
    theme = st.selectbox("Color Theme", ["Blue", "Green", "Purple"])
    show_technical = st.checkbox("Show technical details", value=False)
    
    # Data management
    st.markdown("---")
    st.subheader("Data")
    if st.button("Save Conversation History"):
        if st.session_state.conversation:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(st.session_state.conversation, f, indent=2)
            st.success(f"Conversation saved to {filename}")
        else:
            st.info("No conversation to save yet.")
    
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.experimental_rerun()

# Main chat interface
st.title("🤖 BLACK ChatBot")
st.markdown("""
Experience natural language processing capabilities through conversation.
Select a mode from the sidebar to enable special features.
""")

# Display conversation history with proper styling
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details") and show_technical:
            st.json(message["details"])

# User input and processing
if prompt := st.chat_input("Type your message here..."):
    # Add user message immediately
    st.session_state.conversation.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show "bot is typing..." indicator
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("_NLPBot is typing..._")
        time.sleep(0.8)  # simulate "thinking" delay
    
    # Generate response based on feature
    user_name = st.session_state.user_name.strip()
    response = ""
    details = {}
    
    if st.session_state.current_feature == "chat":
        # Replace lambda responses dynamically with username
        resp_raw = chatbot.respond(prompt)
        # If response is callable, call it with username, else just use it
        if callable(resp_raw):
            response = resp_raw(user_name)
        else:
            response = resp_raw
    
    elif st.session_state.current_feature == "sentiment_analysis":
        analysis = analyze_sentiment(prompt)
        response = f"Sentiment Analysis Result: {analysis}"
        details = {
            "feature": "sentiment_analysis",
            "text": prompt,
            "result": analysis
        }
    
    elif st.session_state.current_feature == "text_summarization":
        try:
            summary = summarize_text(prompt)
            response = f"Text Summary: {summary}"
            details = {
                "feature": "text_summarization",
                "original_text": prompt,
                "summary": summary,
                "algorithm": "LexRank"
            }
        except Exception as e:
            response = "Oops, I couldn't summarize that. Please try a shorter or clearer text."
            details = {"error": str(e)}
    
    elif st.session_state.current_feature == "entity_recognition":
        if nlp:
            entities = extract_entities(prompt)
            if isinstance(entities, str):  # error message string
                response = entities
            elif not entities:
                response = "No named entities found in the text."
            else:
                response = "Named Entities Found:\n" + "\n".join([f"- {ent[0]} ({ent[1]})" for ent in entities])
            details = {
                "feature": "entity_recognition",
                "text": prompt,
                "entities": entities,
                "model": "en_core_web_sm"
            }
        else:
            response = "spaCy model not loaded. Please install by running: python -m spacy download en_core_web_sm"
    
    # Replace typing placeholder with actual response
    typing_placeholder.markdown(response)
    
    # Append assistant message
    st.session_state.conversation.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
        "details": details if details else None
    })

# Dark theme CSS with dynamic color theme
theme_colors = {
    "Blue": "#2196F3",
    "Green": "#4CAF50",
    "Purple": "#9C27B0"
}
primary_color = theme_colors.get(theme, "#2196F3")

st.markdown(f"""
<style>
    body, .stApp {{
        background-color: #121212;
        color: #ffffff;
    }}
    .stChatMessage {{
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 16px;
        line-height: 1.5;
    }}
    .stChatMessage.user {{
        background-color: {primary_color};
        color: white;
    }}
    .stChatMessage.assistant {{
        background-color: #2e2e2e;
        color: #e0e0e0;
    }}
    .stTextInput > div > div > input {{
        background-color: #2a2a2a;
        color: white;
    }}
    .stTextInput label {{
        color: white;
    }}
    .stButton > button {{
        background-color: {primary_color};
        color: white;
        border: none;
        border-radius: 6px;
    }}
    .stSelectbox label {{
        color: white;
    }}
    .stSidebar {{
        background-color: #1f1f1f;
        color: white;
    }}
    .stSidebar input, .stSidebar select {{
        background-color: #2e2e2e;
        color: white;
    }}
    hr {{
        border-top: 1px solid #444;
    }}
</style>
""", unsafe_allow_html=True)
