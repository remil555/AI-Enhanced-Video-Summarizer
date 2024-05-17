import openai

# Set your OpenAI API key here
# openai.api_key = "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"

system = [{"role": "system", "content": "You are a chatbot who enjoys Python programming."}]
chat = []

# Replace the following with the paragraph you want to summarize
paragraph_to_summarize = "YOLO (You Only Look Once) is a real-time object detection algorithm widely used in computer vision. Unlike traditional two-stage detectors, YOLO processes an entire image in a single forward pass through a convolutional neural network. It divides the image into a grid and predicts bounding boxes, class probabilities, and confidence scores for multiple objects in each grid cell. YOLO excels in speed and efficiency, making it suitable for real-time applications such as autonomous vehicles and surveillance. Various versions exist, with YOLOv4 being one of the latest. The algorithm's key components include a grid system for bounding box predictions, class predictions, and object confidence scores. It employs non-maximum suppression to filter duplicate detections. YOLO has gained popularity for its balance between speed and accuracy, offering an effective solution for object detection tasks."

# Append user input to the chat
user = [{"role": "user", "content": paragraph_to_summarize}]
response = openai.ChatCompletion.create(
    messages=system + chat[-20:] + user,
    model="gpt-3.5-turbo", top_p=0.5, stream=True
)

reply = ""
for delta in response:
    if not delta['choices'][0]['finish_reason']:
        word = delta['choices'][0]['delta']['content']
        reply += word

chat.append(user[0])
chat.append({"role": "assistant", "content": reply})

# Print the generated summary
print("Generated Summary:")
print(reply)




# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk.tag import pos_tag
# from nltk.chunk import ne_chunk
# from textblob import TextBlob

# import openai

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# # Function for general summary using the first code
# def general_summary(paragraph):
#     sentences = sent_tokenize(paragraph)
#     key_sentences = sentences[:3]
#     summary = ' '.join(key_sentences)
#     return summary

# # Function for summary by high-ranking words using the first code
# def summary_by_high_ranking_words(paragraph):
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(paragraph)
#     filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
#     high_ranking_words = set(filtered_words[:5])
#     summary = ' '.join(sentence for sentence in sent_tokenize(paragraph) if any(word in sentence.lower() for word in high_ranking_words))
#     return summary

# # Function for summary by keywords using the first code
# def summary_by_keywords(paragraph, keywords):
#     summary = ' '.join(sentence for sentence in sent_tokenize(paragraph) if any(keyword in sentence.lower() for keyword in keywords))
#     return summary


# # Function for summary in simple words using TextBlob
# def summary_in_simple_words(paragraph):
#     blob = TextBlob(paragraph)
#     simplified_sentences = [sentence.words.lower().singularize() for sentence in blob.sentences]
#     simplified_summary = ' '.join(map(str, simplified_sentences))
#     return simplified_summary




# # Function for sentiment analysis summary using the first code
# def sentiment_analysis_summary(paragraph):
#     sid = SentimentIntensityAnalyzer()
#     sentiment_score = sid.polarity_scores(paragraph)['compound']
    
#     if sentiment_score >= 0.05:
#         sentiment = "positive"
#     elif sentiment_score <= -0.05:
#         sentiment = "negative"
#     else:
#         sentiment = "neutral"
    
#     summary = f"The overall sentiment is {sentiment}. {paragraph}"
#     return summary

# # Function for summary by key ideas using NER
# def summary_by_key_ideas(paragraph):
#     words = word_tokenize(paragraph)
#     tagged_words = pos_tag(words)
#     named_entities = ne_chunk(tagged_words)

#     key_ideas = []
#     for chunk in named_entities:
#         if hasattr(chunk, 'label') and chunk.label():
#             key_ideas.append(' '.join(c[0] for c in chunk.leaves()))

#     key_ideas_summary = ', '.join(key_ideas)
#     return key_ideas_summary


# # Set your OpenAI API key here
# openai.api_key = "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"

# system = [{"role": "system", "content": "You are a chatbot who enjoys Python programming."}]
# chat = []

# # Replace the following with the paragraph you want to summarize
# paragraph_to_summarize = """Generative Pre-trained Transformer, or GPT, stands as a groundbreaking development in the field of artificial intelligence, specifically within natural language processing. Created by OpenAI, GPT is the culmination of advancements in deep learning and transformer architectures, designed to comprehend and generate human-like text. The evolution of GPT has seen several iterations, with GPT-3 being the latest and most powerful version as of my last knowledge update in January 2022.

# At its core, GPT employs a transformer architecture, a model architecture introduced by Vaswani et al. in 2017. The transformer model revolutionized natural language processing by leveraging attention mechanisms to process input data in parallel, making it highly efficient and scalable. GPT takes this transformer architecture and enhances it with a pre-training approach, allowing the model to learn the intricacies of language from vast amounts of diverse textual data.

# Pre-training in the context of GPT involves exposing the model to a massive corpus of text data, such as books, articles, and websites. During this phase, GPT learns to predict the next word in a sequence, developing an understanding of grammar, context, and semantic relationships. The resulting pre-trained model becomes a versatile language understanding tool, capable of handling a myriad of tasks without task-specific training.

# One of the distinguishing features of GPT is its generative nature. The model can not only understand context and information but also generate coherent and contextually relevant text. This ability is harnessed through a decoding process where the model generates output word by word based on the given input. GPT's proficiency in generating human-like text has found applications in various domains, including content creation, chatbots, and language translation.

# The sheer scale of GPT-3 is noteworthy. With a staggering 175 billion parameters, GPT-3 dwarfs its predecessors and other language models. More parameters mean a broader understanding of context, nuanced language comprehension, and improved performance across an array of tasks. The model's capabilities range from language translation and summarization to code generation and creative writing.

# Despite its remarkable achievements, GPT-3 is not without challenges. Ethical concerns regarding bias in generated text, potential misuse for malicious purposes, and the environmental impact of training large models have sparked discussions within the AI community.

# Looking ahead, the future of GPT and similar models holds promise for even more sophisticated language understanding. Continued research and development in transformer architectures, ethical considerations, and efforts to mitigate biases will likely shape the trajectory of AI language models. GPT exemplifies the potential of pre-trained models in transforming the landscape of natural language processing and artificial intelligence, marking a significant milestone in the ongoing quest for machines to understand and generate human-like text."""

# # Append user input to the chat
# user = [{"role": "user", "content": paragraph_to_summarize}]
# response = openai.ChatCompletion.create(
#     messages=system + chat[-20:] + user,
#     model="gpt-3.5-turbo", top_p=0.5, stream=True
# )

# reply = ""
# for delta in response:
#     if not delta['choices'][0]['finish_reason']:
#         word = delta['choices'][0]['delta']['content']
#         reply += word

# chat.append(user[0])
# chat.append({"role": "assistant", "content": reply})

# # Combine all types of summaries
# combined_summary = f"General Summary: {general_summary(paragraph_to_summarize)}\n\nSummary by High-Ranking Words: {summary_by_high_ranking_words(paragraph_to_summarize)}\n\nSummary by Keywords: {summary_by_keywords(paragraph_to_summarize, ['yolo', 'computer vision', 'objects'])}\n\nSummary in Simple Words: {summary_in_simple_words(paragraph_to_summarize)}\n\nSentiment Analysis Summary: {sentiment_analysis_summary(paragraph_to_summarize)}\n\nSummary by Key Ideas: {summary_by_key_ideas(paragraph_to_summarize)}\n\nGenerated Summary: {reply}"

# # Print the combined summary
# print("Combined Summary:")
# print(combined_summary)
