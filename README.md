# Natural Language Processing with Hugging Face Transformers

<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : [Your Name]

## My todo : 

### 1. Exercise 1 - Sentiment Analysis (Tweet-based Model)

```python
# TODO :
from transformers import pipeline

tweet = "Just watched the new One Piece episode. It was incredible! #anime"

# Using tweet-specific sentiment model
tweet_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
tweet_sentiment(tweet)

# Optionally compare with default sentiment model
default_sentiment = pipeline("sentiment-analysis")
default_sentiment(tweet)
```

Result : 

**Tweet-specific model:**
```
[{'label': 'LABEL_2', 'score': 0.9886412024497986}] 
```

**Default model:**
```
[{'label': 'POSITIVE', 'score': 0.9998113512992859}]
```

Analysis on exercise 1 : 

The tweet-specific sentiment model (cardiffnlp/twitter-roberta-base-sentiment) uses numerical labels where LABEL_2 corresponds to positive sentiment, showing high confidence (98.86%). The default model directly outputs "POSITIVE" with even higher confidence (99.98%). Both models correctly identify the positive sentiment, but the Twitter-specific model is better suited for social media content with its specialized training on 58M tweets.

### 2. Exercise 2 - Topic Classification (Zero-shot Classification)

```python
# TODO :
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "Luffy wants to become the Pirate King by finding the One Piece treasure."
labels = ["history", "science", "anime", "technology", "sports"]

classifier(text, candidate_labels=labels)
```

Result : 

```
{'sequence': 'Luffy wants to become the Pirate King by finding the One Piece treasure.',
 'labels': ['anime', 'technology', 'history', 'sports', 'science'],
 'scores': [0.9526888132095337,
  0.01857858896255493,
  0.012616857886314392,
  0.009363248012959957,
  0.006752511952072382]}
```

Analysis on exercise 2 : 

The zero-shot classifier correctly identifies "anime" as the most relevant category with overwhelming confidence (95.27%). This demonstrates the model's ability to understand context and associate "Luffy", "Pirate King", and "One Piece treasure" with anime content, even without specific training on anime-related classification tasks.

### 3. Exercise 3 - Text Generation

```python
# TODO :
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "In the world of anime, friendship and determination can"
generator(prompt, max_length=30, num_return_sequences=3)
```

Result : 

```
[{'generated_text': "In the world of anime, friendship and determination can be very powerful. The characters and their actions inspire and inspire each other, and there's no better way to illustrate this than by creating a love triangle that's as emotionally compelling as it is believable.\n\nFamitsu has also found inspiration from anime and manga. They create a unique and unique world where humanity is at war, and what matters is how to survive. The series also features a unique story that takes place in the middle of a massive war, as well as a unique characters-only setting.\n\nThe series is due out in Japan on September 6, 2017, and will air on Crunchyroll's Anime Central Network.\n\nSource: MangaHelpers via Otakomu"}]
```

Analysis on exercise 3 : 

The GPT-2 model generates coherent and contextually relevant text about anime themes. It successfully continues the prompt about friendship and determination, expanding into detailed discussion about anime storytelling elements, character development, and even includes realistic details about anime industry practices and platforms like Crunchyroll.

### 4. Exercise 4 - Named Entity Recognition (NER)

```python
# TODO :
from transformers import pipeline

ner_model = pipeline("ner", model="Jean-Baptiste/camembert-ner", grouped_entities=True)

text = "Emma and Ray escaped from Grace Field House and are searching for William Minerva in the forest."

ner_model(text)
```

Result : 

```
[{'entity_group': 'PER',
  'score': 0.48230842,
  'word': 'Emma',
  'start': 0,
  'end': 4},
 {'entity_group': 'PER',
  'score': 0.6580116,
  'word': 'Ray',
  'start': 8,
  'end': 12},
 {'entity_group': 'LOC',
  'score': 0.9436026,
  'word': 'Grace Field House',
  'start': 25,
  'end': 43},
 {'entity_group': 'PER',
  'score': 0.93255043,
  'word': 'William Minerva',
  'start': 65,
  'end': 81}] 
```

Analysis on exercise 4 : 

The NER model successfully identifies all entities in the text. It correctly classifies "Emma" and "Ray" as persons (PER), "Grace Field House" as a location (LOC), and "William Minerva" as a person (PER). The confidence scores vary, with location detection showing the highest confidence (94.36%), demonstrating the model's strength in identifying place names.

### 5. Exercise 5 - Question Answering

```python
# TODO :
from transformers import pipeline

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = "The Promised Neverland is an anime about children escaping from an orphanage that hides a dark secret."
question = "What is The Promised Neverland about?"

qa_model(question=question, context=context)
```

Result : 

```
{'score': 0.47103482484817505,
 'start': 41,
 'end': 76,
 'answer': 'children escaping from an orphanage'}
```

Analysis on exercise 5 : 

The question-answering model extracts the most relevant information from the context, identifying "children escaping from an orphanage" as the core answer to what The Promised Neverland is about. The moderate confidence score (47.10%) suggests the model found the answer span but indicates some uncertainty, possibly due to the complex nature of the question.

### 6. Exercise 6 - Text Summarization

```python
# TODO :
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = """
The Promised Neverland follows a group of orphans who live in a seemingly idyllic orphanage. However, they discover that the orphanage is a farm where children are raised to be fed to demons. Emma, Norman, and Ray devise a plan to escape and save the other children. Their journey tests their intelligence, courage, and trust in each other as they uncover deeper secrets about the world outside.
"""

summarizer(text)
```

Result : 

```
[{'summary_text': ' The Promised Neverland follows a group of orphans who live in a seemingly idyllic orphanage . They discover that the orphanage is a farm where children are raised to be fed to demons . Emma, Norman, and Ray devise a plan to escape and save the other children .'}]
```

Analysis on exercise 6 :

The summarization model effectively condenses the original text while preserving all key plot points: the orphanage setting, the dark secret about demons, the main characters (Emma, Norman, and Ray), and their escape plan. It maintains the narrative flow while reducing length by approximately 25%, demonstrating good content compression without losing essential information.

### 7. Exercise 7 - Translation (English to German)

```python
# TODO :
translator = pipeline("translation_en_to_de")

text = "The Promised Neverland is a thrilling and emotional anime series."

translator(text)
```

Result : 

```
[{'translation_text': 'The Promised Neverland ist eine spannende und emotionale Anime-Serie.'}]
```

Analysis on exercise 7 :

The translation model provides an accurate German translation, correctly translating "thrilling" to "spannende" and "emotional" to "emotionale". It properly maintains the sentence structure and keeps "The Promised Neverland" and "Anime-Serie" as appropriate terms that would be understood in German context, showing good cross-cultural linguistic understanding.

---

## Analysis on this project

This project demonstrates the versatility and power of Hugging Face Transformers for various NLP tasks. Each exercise showcases different aspects of natural language processing, from understanding sentiment and generating text to extracting information and translating between languages. The anime-themed examples make the learning process more engaging while demonstrating real-world applications of these technologies. The consistent high performance across different tasks highlights the effectiveness of pre-trained transformer models in solving diverse language problems without requiring extensive custom training.