# Analyzing Reddit User Sentiment and Trends with PySpark
# Overview
Social media listening, also known as social media monitoring or social media intelligence, refers to the process of tracking and analyzing conversations, mentions, and trends across social media platforms. Given that Reddit is a rich source of information and the data is relatively easy to obtain compared to other social media channels, I have attempted to build a social media listening using historical data. Historical data gives a retrospective view wherein we can study trends over time and obtain aggregated metrics over a larger volume of data. This can help with planning specific campaigns across certain times of the year.

# 1. Data Source
Kaggle - The primary source of historical data was the r/technology subreddit’s historical dataset which is updated on a weekly level and is available [here](https://www.kaggle.com/datasets/curiel/rtechnology-posts-and-comments/data).<br>
A brief description of the various columns is given below:
* register_index: A unique identifier for each row which we are excluding
* post_id: unique identifier for posts in r/technology
* comment_id: unique identifier for comments associated to a post
* author: Username of the post author or commentor
* datetime: Timestamp of post or comment creation
* title: Post title summarizing the topic, this is “NaN” for comments
* url: Web address associated with the post, this is “NaN” for comments
* score: Rating or score received by the post
* comments: Total number of comments for a post, this is “NaN” for comments
* text: Actual content of the post or comment
* author_post_karma: The total net score for the user, referred to as “karma” as
per Reddit terminology
* tag: tags associated with the post, this is a mandate for the subreddit

From this dataset, as of the end of April 2024, there are over 1.4M comments across 15k posts in the historical data (starting from June 2023). We used PySpark to process this data at scale.
For analysis, I first performed exhaustive exploratory data analysis on the posts and comments data separately, after which I've calculated multiple metrics like reach, impressions, sentiment, and share of voice and visualized information that was rendered in a Streamlit web application. These metrics help products, brands, and other concerned parties to understand the pulse of customers and their perception about certain topics, brands, and products from an unfiltered open discussion on Reddit.

# 2. Flow Diagram
The flow diagram describing the set of steps involved in going from raw data to web application is shown in Figure 1.
![Flow Diagram](https://raw.githubusercontent.com/lokaremonica/Social-Media-Listening-using-Reddit/main/Figure1.png)

This flow diagram outlines the process of social media listening using historical data from Reddit, specifically from the r/technology subreddit. The diagram describes a sequence of operations performed on the dataset to build a keyword-search-based dashboard for social media listening and analysis. A summary of the steps are described below:
1. As referenced earlier, the raw data is obtained from the source and this is written to a Google Cloud Storage bucket using a batch-processing approach.
2. The raw data is segregated into posts and comments and each data split goes through multiple processing steps using PySpark for processing large-scale data efficiently. The processing includes filtering, text mining, text cleaning, visualization, and Machine Learning. Spark’s in-memory data processing helps in performing these steps quickly and efficiently.
3. As a standalone one-time process, an unsupervised machine learning algorithm, Latent Dirichlet Allocation (LDA), is used to dissect the posts’ title text and identify topics within it. This approach facilitates the summarizing of the post titles into key themes. We will elaborate on the approach in upcoming sections of the report.
4. After processing, the cleaned data is separated into CSV files for post titles and comments respectively and written to a Google Cloud Storage (GCS) bucket. This storage acts as a repository for the processed data, making it accessible for further analysis or downstream applications/dashboards.
5. Finally, the processed data is used to build a dashboard using Streamlit. The dashboard provides a user-friendly interactive interface to analyze social media metrics and trends based on keyword searches on posts and comments views respectively.

# 3. Data Processing & transformation using Spark
PySpark is used as the backbone for data processing due to its ability to handle large datasets efficiently. It is utilized for data cleaning, transformation, and aggregation of Reddit data, which typically involves a high volume of posts and comments. I have performed data preprocessing and transformation using PySpark which is an Apache Spark Python API open-source system for processing and analyzing Reddit datasets. As Reddit comments data was quite large, Spark was the most convenient option for us to process the large comments data. Employed for its NLP capabilities, TextBlob analyzes sentiment and subjectivity in Reddit comments, which aids in gauging public opinion and emotional tone.
* Libraries such as Pandas: Used for handling smaller subsets of data for detailed analysis and quick transformations which are less intensive than those handled by PySpark. 
* NumPy: Provides support for operations on arrays and matrices, which are essential for numerical computations in data analysis.
* Seaborn and Matplotlib: These libraries are critical for visualizing data, helping to identify trends, patterns, and outliers in social media engagement on Reddit.
* re and nltk: These are crucial for text processing and NLP tasks. They help in cleaning and preparing the textual data extracted from Reddit for deeper analysis. 
* Also, some other of the Pyspark libraries were used are StopWordsRemover, NGram, Tokenizer, SentimentIntensityAnalyzer, pyspark.sql, pyspark.sql.functions, pyspark.ml, pyspark.ml.feature. 
> These libraries together provide the necessary tools and functionalities for processing, transforming, and analyzing Reddit posts and comments data using PySpark. These libraries collectively facilitate the effective transformation and analysis of social media data from Reddit, enabling insights into user behavior, trends, and sentiment. The notebook **PySpark_Reddit_DataProcessing.ipynb** contains the code implementation for PySpark.

# 4. EDA
The data contains over 15k unique posts and 1.4M comments. Given this considerable difference in volume, I decided to split the data into posts and comments separately. This helps us analyze post titles and comments separately. Post titles are shorter and less noisy, while comments are varied and may contain more noise. For this reason, I decided to use only post titles for unsupervised machine learning.
Once we separate the posts as a separate dataset, I can further classify posts as those with comments and the rest, without. <br>
Out of the total 15k posts between Jun-2023 and Apr-2024, I observed ~1.3k posts per month on average. Around 7% of the posts are “orphan posts” (posts without comments) and these were removed since there's no engagement in these posts.<br>

After excluding the posts without comments, I studied the distribution of posts based on comment volume. I can observe that the distribution is right-skewed with most posts having <200 comments.<br>

The r/technology subreddit had over 1.4M comments across posts (between Jun- 2023 and Apr-2024) with a monthly average of around 120k comments. To reduce noise, I only considered comments with at least 5 characters in length. By excluding the comments with less than 5 characters in length, I excluded around 1000 records which amounted to 0.06% reduction from the raw comments data and this is negligible.<br>

# 5. Text Composition
Next I studied the text composition of the post title. The unigrams and bigrams tend to be dominated by popular terms, trends, and personalities from the technology world. The distributions are dominated by Artificial Intelligence and Business terms since these two tags formed a combined 33% of all posts.<br>

# 6. Sentiment Analysis
Sentiment was extracted using the Python textblob library which provides a sentiment polarity value ranging from -1 to 1. Highly negative sentiment is indicated by - 1, whereas highly positive is indicated as 1, and neutral is represented by 0. <br>

# 7.Text Cleaning
Operations like changing the post title to lowercase, removal of English stopwords, removal of HTML tags and URLs, removal of emoji’s and non-ASCII characters were performed using PySpark. Contractions (i.e. expansions of “i’m” to “i am”) were also expanded. <br>

# 8. Unsupervised Learning on Post Titles
Since the dataset does not contain labeled data, I applied unsupervised learning techniques to uncover hidden patterns in the post titles. Two approaches were attempted:<br>

## 8.1 K-Means Clustering
K-Means clustering was applied to group similar Reddit post titles into clusters. The process involved:
1. TF-IDF Vectorization: Converting text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
2. Hopkins Test: Checking clustering tendency to determine if the data is suitable for clustering.
3. Elbow Method: Identifying the optimal number of clusters.
4. Clustering Execution: Applying K-Means clustering to assign posts into different clusters.<br>

Findings:<br>
* The Hopkins statistic was low, indicating that the data did not exhibit strong clustering tendencies.
* The Elbow method did not yield a distinct elbow point, suggesting that distinct clusters were not present in the dataset.
* Due to the lack of clear separation, K-Means was not effective in categorizing the Reddit post titles.<br>

## 8.2 Latent Dirichlet Allocation (LDA) for Topic Modeling
LDA, a probabilistic topic modeling technique, was used to extract latent topics from the dataset.<br>
Process:<br>
1. TF-IDF Vectorization: Extracting important words.
2. Training LDA Model: Generating topics with trigrams.
3. Visualizing Topics: Displaying keywords associated with each topic.<br>

Key Topics Identified:
* Topic 0: News around training data for AI models.
* Topic 1: Discussions about deepfakes and AI risks.
* Topic 2: AI-related tool launches (e.g., Google's Gemini).
* Topic 3: Mixed discussions on AI developments.
* Topic 4: New AI product launches and controversies. <br>
LDA successfully extracted meaningful themes, making it a more suitable method than K-Means for analyzing post titles.<br>

# 9. Streamlit Social Media Listening Dashboard
A Streamlit-based web application was developed to visualize and analyze social media trends on Reddit using historical data.<br>

## 9.1 Data Ingestion & Processing
* The data ingestion framework was designed to read processed CSV files directly from Google Cloud Storage (GCS).
* This ensures scalability and eliminates the need for redundant processing in the app layer.<br>

##  9.2 Dashboard Features
✔️ Keyword-based search – Users can input any keyword to track mentions in posts and comments. ✔️ Post & Comment Metrics – Aggregated reach, impressions, and sentiment analysis. ✔️ Heatmaps & Trend Charts – Visualizing post & comment trends over time. ✔️ Topic Modeling Insights – Extracting dominant themes from discussions. ✔️ Sentiment Distribution – Analyzing how sentiments fluctuate over time. ✔️ Top N-Grams – Understanding common words & phrases in discussions.<br>

##  9.3 Dashboard Layout
The dashboard is designed in five modular sections:<br>
1. Keyword Input & Data Loading<br>
    * Users input specific keywords to analyze.
    * Data is dynamically loaded from Google Cloud Storage.
2. Aggregate Metrics<br>
    * Total Posts & Comments mentioning the keyword.
    * Share of Voice & Reach percentages.
    * Overall Sentiment Breakdown.
3. Post & Comment Trends<br>
    * Heatmaps & Area Charts visualize volume trends.
4. Topic & Sentiment Analysis<br>
    * LDA Topic Modeling results.
    * Sentiment Analysis Distribution.
5. N-Gram Analysis<br>
    * Word Frequency trends for Unigrams, Bigrams, and Trigrams.
## 9.4 Live Dashboard <br>
🌐 Try it here: https://reddit-historical-listening.streamlit.app/<br>

# 10. Deployment and Setup <br>
To run this project locally, follow these steps:<br>
## 10.1 Prerequisites
✔️ Install Python 3.9+ ✔️ Install Streamlit, PySpark, Pandas, NumPy, and TextBlob <br>
## 10.2 Installation
Clone the repository and install dependencies:
git clone https://github.com/lokaremonica/Social-Media-Listening-using-Reddit.git
cd Social-Media-Listening-using-Reddit
pip install -r requirements.txt  <br>
##  10.3 Running the Streamlit App <br>
streamlit run streamlit_reddit_historical.py <br>
The app should now be accessible at http://localhost:8501/. <br>

# 11. Future Enhancements
🚀 Expand to Multiple Social Media Channels
* Integrate Twitter (X), Instagram, YouTube, and Google Trends data.
📊 Advanced NLP & ML Models
* Implement transformers (BERT, GPT-4, LangChain, LLaMA) for better sentiment and trend prediction.
🔍 Real-Time Data Streaming
* Enhance the system to track live discussions in real time.
🎯 Influencer & Brand Sentiment Tracking
* Identify top influencers discussing specific topics.
* Analyze brand sentiment over time.
🤖 Generative AI Insights
* Use LLMs (like OpenAI GPT) to summarize trends dynamically.
📈 Predictive Analytics for Social Media Trends
* Forecast consumer behavior and market trends using time-series analysis.

# 12. Conclusion
This project provides a scalable social media listening solution using Reddit data, PySpark, and NLP techniques. The Streamlit dashboard enables interactive exploration of historical trends, sentiment analysis, and topic modeling. Future improvements will focus on real-time monitoring, advanced AI models, and multi-platform integration.
