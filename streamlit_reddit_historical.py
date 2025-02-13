import streamlit as st
import pandas as pd
import numpy as np
from st_files_connection import FilesConnection
import time
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Defining SessionState class
class SessionState:
    def __init__(self):
        self.posts_df = None
        self.comments_df = None
        self.df = None

# Defining the layout of the view
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.sidebar.header('Choose the type of `data`')

# Header and Title
# Custom title with larger size and orange color
st.markdown("<h2 style='text-align: center; color: orange;'>Social Media Listening for r/technology subreddit</h2>", unsafe_allow_html=True)

# Official reddit icon URL
icon_url = "https://www.redditinc.com/assets/images/site/Reddit_Lockup_Logo.svg"
# Displaying Reddit icon at top right position
st.markdown(
    f'<img src="{icon_url}" style="position: absolute; top: 0; right: 0; width: 150px; height: 150px;">',
    unsafe_allow_html=True
)

# Dictionary to map views to datasets
view_datasets = {
    "Posts": "reddit_historical/posts_clean_final.csv",
    "Comments": "reddit_historical/comments_clean_final.csv"
    #"Topic modeling": "reddit_historical/lda_sample.csv"  # Placeholder for machine learning outputs
}

# Button for toggling between views
selected_view = st.sidebar.radio("View", list(view_datasets.keys()))

# Connecting to Google cloud storage bucket
conn = st.connection('gcs', type=FilesConnection)

# Function to load data for a specific view
@st.cache_data
def load_data(view):
    if view in view_datasets:
        dataset = view_datasets[view]
        try:
            df = conn.read(dataset, input_format="csv", ttl=600, encoding="utf-16")
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return pd.DataFrame()


# Creating session state for cache
if 'session_state' not in st.session_state:
    st.session_state.session_state = SessionState()

# Displaying a button to satart data load
upload_button = st.button("Load historical data")

# Loading data for selected view if upload button is clicked
if upload_button:
    with st.spinner(f"Loading {selected_view.lower()} data..."):
        # Load data for selected view
        if selected_view == "Posts":
            st.session_state.session_state.df = load_data(view_datasets[selected_view])
        elif selected_view == "Comments":
            st.session_state.session_state.df = load_data(view_datasets[selected_view])
        #elif selected_view == "Topic modeling":
        #    st.session_state.session_state.df = load_data(view_datasets[selected_view])

# Displaying content based on the selected view
if selected_view == "Posts":
    # Posts view logic
    if st.session_state.session_state.df is not None:
        # Displaying posts-related content
        pass
elif selected_view == "Comments":
    # Comments view logic
    if st.session_state.session_state.df is not None:
        # Displaying comments-related content
        pass
#elif selected_view == "Topic modeling":
    # ML 
#    if st.session_state.session_state.df is not None:
        # Display ML related content
#        pass



# Loading data for selected view if upload button is clicked
if upload_button:
    st.write(f"Loading data for {selected_view}...")
    df = load_data(selected_view)
    if not df.empty:
        st.session_state.df = df
        st.success("Dataset loaded successfully!")
        #st.write("Top 5 rows of the loaded dataset:")
        #st.write(df.head())  # Display top 5 rows of the loaded dataset
    else:
        st.error("Failed to load dataset.")

### Function definitions for data processing ###

# Creating custom metrics for comments
def calculate_reach_metrics_commment(filtered_df, keyword):
    # Converting the text column and keyword to lowercase
    filtered_text = filtered_df[text_column].str.lower()
    keyword_lower = keyword.lower()

    # Calculating the total number of times the keyword is mentioned
    keyword_mentions = filtered_text.str.count(keyword_lower).sum()

    # Calculating the total number of words in the comments
    total_words = filtered_text.str.split().apply(len).sum()

    # Calculating share of voice (filtered word/total words)
    share_of_voice = keyword_mentions / total_words

    # Calculating the total number of comments containing the input word
    total_comments_with_keyword = filtered_text.str.contains(keyword_lower).sum()
    
    # Calculating the total unique count of users referencing the keyword
    comments_with_keyword = filtered_df[filtered_text.str.contains(keyword_lower)]
    unique_users_with_keyword = comments_with_keyword['author'].nunique()

    return keyword_mentions, share_of_voice, total_comments_with_keyword, unique_users_with_keyword

# Creating custom metrics for posts
def calculate_reach_metrics_posts(filtered_df, keyword):
    # Converting the text column and keyword to lowercase
    filtered_text = filtered_df[text_column].str.lower()
    keyword_lower = keyword.lower()

    # Filtering posts that contain the keyword
    posts_with_keyword = filtered_df[filtered_text.str.contains(keyword_lower)]

    # Calculating the total number of times the keyword is mentioned
    keyword_mentions_posts = filtered_text.str.count(keyword_lower).sum()

    # Calculating the total number of words in the posts
    total_words_posts = filtered_text.str.split().apply(len).sum()

    # Calculating share of voice (filtered word/total words)
    share_of_voice_posts = keyword_mentions_posts / total_words_posts

    # Calculating the total score (sum of the score column)
    total_score_posts = filtered_df['score'].sum()

    # Calculating the share of posts containing the keyword
    total_posts = st.session_state.df.shape[0]
    posts_with_keyword_count = posts_with_keyword.shape[0]
    share_of_posts_with_keyword = posts_with_keyword_count / total_posts

    return keyword_mentions_posts, share_of_voice_posts, total_score_posts, share_of_posts_with_keyword


# Function to plot top 10 unigrams, bigrams and trigrams
# The function selects a random sample of 10k rows to reduce strain on the app layer
def visualize_associated_ngrams(filtered_df, text_column, keyword):
    # Filtering the dataframe to include only posts/comments containing the keyword
    keyword_lower = keyword.lower()
    filtered_lower = filtered_df[filtered_df[text_column].str.lower().str.contains(keyword_lower)]
    
    # Sampling 10,000 rows randomly if the dataframe has more than 10,000 rows
    if len(filtered_lower) > 10000:
        filtered_lower_sample = filtered_lower.sample(n=10000, random_state=123)
    else:
        filtered_lower_sample = filtered_lower.copy()  
        # Using the entire dataframe if it has 10,000 or fewer rows
    
    # Extracting unigrams, bigrams, and trigrams from the filtered comments
    unigrams = []
    bigrams = []
    trigrams = []
    for ip_text in filtered_lower_sample[text_column]:
        words = ip_text.split()
        unigrams.extend(words)
        bigrams.extend(zip(words, words[1:]))
        trigrams.extend(zip(words, words[1:], words[2:]))
    
    # Counting the frequency of each unigram, bigram, and trigram
    unigram_counts = Counter(unigrams)
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    # Getting top 10 unigrams, bigrams, and trigrams
    top_n_unigrams = dict(sorted(unigram_counts.items(), key=lambda item: item[1], reverse=True)[:10])
    top_n_bigrams = dict(sorted(bigram_counts.items(), key=lambda item: item[1], reverse=True)[:10])
    top_n_trigrams = dict(sorted(trigram_counts.items(), key=lambda item: item[1], reverse=True)[:10])
    
    
    # Plotting the unigrams, bigrams, and trigrams side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Chart title
    fig.suptitle('Top Unigrams, Bigrams, and Trigrams for "{}"'.format(keyword), fontsize=16)

    # Plotting top 10 unigrams
    ax = axes[0]
    ax.barh(list(top_n_unigrams.keys()), list(top_n_unigrams.values()), color='skyblue')
    ax.set_title('Top 10 unigrams')
    ax.set_xlabel('Frequency')

    # Plotting top 10 bigrams
    ax = axes[1]
    bigram_labels = [' '.join(bigram) for bigram in top_n_bigrams.keys()]
    ax.barh(bigram_labels, list(top_n_bigrams.values()), color='salmon')
    ax.set_title('Top 10 Bigrams')
    ax.set_xlabel('Frequency')

    # Plotting top 10 trigrams
    ax = axes[2]
    trigram_labels = [' '.join(trigram) for trigram in top_n_trigrams.keys()]
    ax.barh(trigram_labels, list(top_n_trigrams.values()), color='lightgreen')
    ax.set_title('Top 10 Trigrams')
    ax.set_xlabel('Frequency')

    # Adding subtext
    fig.text(0.5, -0.05, "Chart based on randomly sampled data (10,000 rows) after filtering for the keyword to reduce processing overhead in the app layer", ha='center', fontsize=8)

    plt.tight_layout()
    return fig


### Function definitions end ###


### Dictionaries definitions ###
# Defining color mapping for sentiment categories
sentiment_colors = {
    'positive': 'green',
    'neutral': 'orange',
    'negative': 'red'
}

# Defining colormaps for each sentiment - for use in wordclouds
colormaps = {
    'positive': 'Greens',
    'neutral': 'Oranges',
    'negative': 'Reds'
}

# Defining a dictionary mapping each view to the corresponding column name
view_column_mapping = {
    "Posts": "title_clean",
    "Comments": "comment_clean"
    #"Topic modeling": "title_clean"
}
### Dictionaries definitions end ###


## Dashboard building by view ##
# If data is loaded, display keyword input box and generate charts button
if 'df' in st.session_state and not st.session_state.df.empty:
    if 'keyword' not in st.session_state:
        st.session_state.keyword = ""
    keyword = st.text_input("Enter keyword:", value=st.session_state.keyword)
    generate_button = st.button("Generate charts")

    if generate_button and keyword:
        st.session_state.keyword = keyword
        text_column = view_column_mapping.get(selected_view)  
        # Getting the column name based on the selected view
        # title_clean for posts & topic modeling and comment_clean for comments
        
        if text_column:
            filtered_df = st.session_state.df[st.session_state.df[text_column].str.contains(keyword, case=False)]
            #st.write(f"##### Number of {selected_view.lower()} containing '{keyword}': {len(filtered_df)}")
            
            if not filtered_df.empty:
### Posts view ###
                if selected_view == "Posts":

                    ## Displaying metrics for posts ##
                    st.markdown('### :orange[Metrics]')
                    # Defining the metrics for 1st row
                    # Running posts specfic function
                    keyword_mentions_posts, share_of_voice_posts, total_score_posts, share_of_posts_with_keyword = calculate_reach_metrics_posts(filtered_df, keyword)
                    # Converting share of voice to %
                    share_of_voice_posts_percentage = f"{share_of_voice_posts:.2%}"
                    # Converting influence to %
                    influence_percentage = f"{share_of_posts_with_keyword:.2%}"

                    # Splitting 4 metrics in 4 columns
                    col1_p, col2_p, col3_p, col4_p = st.columns(4)
                    col1_p.metric("Count of posts", keyword_mentions_posts)
                    col2_p.metric(f"Influence (share of overall posts) of '{keyword}'", influence_percentage)
                    col3_p.metric("Share of Voice (share of all words)", share_of_voice_posts_percentage)
                    col4_p.metric("Total aggregate score (upvotes-downvotes)", total_score_posts)

                    ## Displaying charts for posts ##
                    st.markdown('### :orange[Charts]')
                    # Heatmap of post volume across time in 2nd row
                    # Extracting month and day to standarize time measurement
                    filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
                    filtered_df['month_hm'] = filtered_df['datetime'].dt.month # month number
                    filtered_df['month_name'] = filtered_df['datetime'].dt.strftime('%b') # month name
                    filtered_df['year'] = filtered_df['datetime'].dt.year
                    filtered_df['day_hm'] = filtered_df['datetime'].dt.day
                    filtered_df['day_name'] = filtered_df['datetime'].dt.day_name()

                    # Grouping by year, month and day for count of posts
                    posts_counts_hm = filtered_df.groupby(['year','month_name', 'day_name']).size().reset_index(name='post_count')

                    st.markdown('### Heatmap of post volume across time')
                    # Creating heatmap using Plotly
                    fig_hm_post = px.density_heatmap(posts_counts_hm, x='month_name', y='day_name', z='post_count', nbinsx=12, nbinsy=7)

                    # Customizing the heatmap
                    fig_hm_post.update_layout(
                        title='Heatmap of Post Volume',
                        xaxis_title='Month',
                        yaxis_title='Day',
                        coloraxis_colorbar=dict(title='Post Count'),
                        height=400
                    )

                    # Displaying the heatmap in Streamlit
                    st.plotly_chart(fig_hm_post, use_container_width=True)

                    # Defining the charts for 3rd row
                    # Defining 2 columns, equally distributed
                    c1_p, c2_p = st.columns((5,5))
                    with c1_p:
                        # Bar chart of count of posts by tag
                        tag_counts_post = filtered_df['tag'].value_counts()
                        tag_counts_post_df = pd.DataFrame({'Tag': tag_counts_post.index, 'Count': tag_counts_post.values})
                        tag_counts_post_df['Share'] = tag_counts_post_df['Count'] / tag_counts_post_df['Count'].sum()
                        fig_bar_post = px.bar(tag_counts_post_df, x='Tag', y='Count', hover_data={'Count': True, 'Share': ':.2%'}, 
                                        labels={'Tag': 'Tag', 'Count': 'Count of posts'}, 
                                        title='Count of posts by tag')
                        fig_bar_post.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1)
                        fig_bar_post.update_layout(hoverlabel=dict(bgcolor="black", font_size=12, font_family="Rockwell"))
                        # Displaying the bar chart in Streamlit
                        st.plotly_chart(fig_bar_post)
                    with c2_p:
                        # Area chart showing count of posts over time
                        fig_area_post = px.area(filtered_df.groupby('date').size().reset_index(name='count'), x='date', y='count', title='Count of posts over time')
                        fig_area_post.update_traces(line=dict(color='blue'), marker=dict(color='blue', size=12))
                        # Displaying the area chart in Streamlit
                        st.plotly_chart(fig_area_post)

                    # Defining the charts for 4th row
                    # Defining 2 columns, first one takes up 30%, the other 70%
                    p_c1, p_c2 = st.columns((3,7))
                    
                    with p_c1:
                        # Caculating the count of filtered rows by sentiment
                        sentiment_counts_post = filtered_df['sentiment'].value_counts()
                        # Donut chart showing count & share of posts by sentiment 
                        fig_donut_post = px.pie(sentiment_counts_post, values=sentiment_counts_post.values, names=sentiment_counts_post.index, hole=0.5, )
                        fig_donut_post.update_traces(marker=dict(colors=[sentiment_colors[sentiment] for sentiment in sentiment_counts_post.index]))
                        fig_donut_post.update_layout(width=300, height=400, title = 'Distribution of sentiment')
                        # Displaying the donut chart in Streamlit
                        st.plotly_chart(fig_donut_post)
                    
                    with p_c2:
                        # Wordclouds for each sentiment type
                        sentiment_wordclouds_post = {}
                        for sentiment in filtered_df['sentiment'].unique():
                            text_post = ' '.join(filtered_df[filtered_df['sentiment'] == sentiment][text_column])
                            wordcloud_post = WordCloud(mask=None, width=1200, height=800, background_color='black').generate(text_post)
                            sentiment_wordclouds_post[sentiment] = wordcloud_post

                        # Plotting the word clouds
                        fig_wordclouds_post, axes = plt.subplots(1, 3, figsize=(18, 6))
                        for ax, (sentiment, wordcloud_post) in zip(axes, sentiment_wordclouds_post.items()):
                            ax.imshow(wordcloud_post, interpolation='bilinear', cmap=colormaps[sentiment.lower()])
                            ax.set_title(f'{sentiment.capitalize()}', color=sentiment_colors.get(sentiment, 'black'), fontsize=24)
                            ax.axis('off')
                            ax.set_aspect('equal')
                        fig_wordclouds_post.suptitle('Wordcloud across sentiment categories', fontsize=28)
                        # Displaying the wordclouds chart in Streamlit
                        st.pyplot(fig_wordclouds_post)

                    # Defining the charts for 5th row
                    # Getting then-gram charts from the function output
                    ngram_posts_plot = visualize_associated_ngrams(filtered_df, text_column, keyword)
                    # Displaying the ngram chart in Streamlit
                    st.pyplot(ngram_posts_plot, use_container_width=True)

### Comments view ###
                elif selected_view =="Comments":

                    ## Displaying metrics for comments ##
                    st.markdown('### :orange[Metrics]')
                    # Defining the metrics for 1st row
                    # Running comment specfic function
                    keyword_mentions, share_of_voice, total_comments_with_keyword, unique_users_with_keyword = calculate_reach_metrics_commment(filtered_df, keyword)
                    # Converting share of voice to %
                    share_of_voice_percentage = f"{share_of_voice:.2%}"
                    
                    # Splitting 4 metrics in 4 columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Impressions (# of mentions)", keyword_mentions)
                    col2.metric(f"Total comments referencing '{keyword}'", total_comments_with_keyword)
                    col3.metric("Share of Voice (share of all words)", share_of_voice_percentage)
                    col4.metric(f"Reach i.e. unique users referencing '{keyword}'", unique_users_with_keyword)

                    ## Displaying charts for comments ##
                    st.markdown('### :orange[Charts]')
                    # Heatmap of comment volume across time in 2nd row
                    # Extracting month and day to standarize time measurement
                    filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
                    filtered_df['month_hm'] = filtered_df['datetime'].dt.month # month number
                    filtered_df['month_name'] = filtered_df['datetime'].dt.strftime('%b') # month name
                    filtered_df['year'] = filtered_df['datetime'].dt.year
                    filtered_df['day_hm'] = filtered_df['datetime'].dt.day
                    filtered_df['day_name'] = filtered_df['datetime'].dt.day_name()

                    # Grouping by year, month and day for count of comments
                    comment_counts_hm = filtered_df.groupby(['year','month_name', 'day_name']).size().reset_index(name='comment_count')

                    st.markdown('### Heatmap of comment volume across time')
                    # Creating heatmap using Plotly
                    fig_hm = px.density_heatmap(comment_counts_hm, x='month_name', y='day_name', z='comment_count', nbinsy=7)

                    # Customizing the heatmap
                    fig_hm.update_layout(
                        title='Heatmap of Comment Volume',
                        xaxis_title='Month',
                        yaxis_title='Day',
                        coloraxis_colorbar=dict(title='Comment Count'),
                        height=400
                    )

                    # Displaying the heatmap in Streamlit
                    st.plotly_chart(fig_hm, use_container_width=True)

                    # Defining the charts for 3rd row
                    # Defining 2 columns, equally distributed
                    c1, c2 = st.columns((5,5))
                    with c1:
                        # Bar chart of count of comments by tag
                        tag_counts_com = filtered_df['tag'].value_counts()
                        tag_counts_com_df = pd.DataFrame({'Tag': tag_counts_com.index, 'Count': tag_counts_com.values})
                        tag_counts_com_df['Share'] = tag_counts_com_df['Count'] / tag_counts_com_df['Count'].sum()
                        fig_bar_com = px.bar(tag_counts_com_df, x='Tag', y='Count', hover_data={'Count': True, 'Share': ':.2%'}, 
                                        labels={'Tag': 'Tag', 'Count': 'Count of comments'}, 
                                        title='Count of posts by tag')
                        fig_bar_com.update_traces(marker_color='skyblue', marker_line_color='black', marker_line_width=1)
                        fig_bar_com.update_layout(hoverlabel=dict(bgcolor="black", font_size=12, font_family="Rockwell"))
                        # Displaying the bar chart in Streamlit
                        st.plotly_chart(fig_bar_com)
                    with c2:
                        # Area chart showing count of comments over time
                        fig_area_com = px.area(filtered_df.groupby('date').size().reset_index(name='count'), x='date', y='count', title='Count of comments over time')
                        fig_area_com.update_traces(line=dict(color='blue'), marker=dict(color='blue', size=12))
                        # Displaying the area chart in Streamlit
                        st.plotly_chart(fig_area_com)
                    
                    # Defining the charts for 4th row
                    # Defining 2 columns, first one takes up 30%, the other 70%
                    s_c1, s_c2 = st.columns((3,7))
                    
                    with s_c1:
                        # Caculating the count of filtered rows by sentiment
                        sentiment_counts_com = filtered_df['sentiment'].value_counts()
                        # Donut chart showing count & share of comments by sentiment 
                        fig_donut_com = px.pie(sentiment_counts_com, values=sentiment_counts_com.values, names=sentiment_counts_com.index, hole=0.5, )
                        fig_donut_com.update_traces(marker=dict(colors=[sentiment_colors[sentiment] for sentiment in sentiment_counts_com.index]))
                        fig_donut_com.update_layout(width=300, height=400, title = 'Distribution of sentiment')
                        # Displaying the donut chart in Streamlit
                        st.plotly_chart(fig_donut_com)
                    
                    with s_c2:
                        # Wordclouds for each sentiment type
                        sentiment_wordclouds_com = {}
                        for sentiment in filtered_df['sentiment'].unique():
                            text_com = ' '.join(filtered_df[filtered_df['sentiment'] == sentiment][text_column])
                            wordcloud_com = WordCloud(mask=None, width=1200, height=800, background_color='black').generate(text_com)
                            sentiment_wordclouds_com[sentiment] = wordcloud_com

                        # Plotting the word clouds
                        fig_wordclouds_com, axes = plt.subplots(1, 3, figsize=(18, 6))
                        for ax, (sentiment, wordcloud_com) in zip(axes, sentiment_wordclouds_com.items()):
                            ax.imshow(wordcloud_com, interpolation='bilinear', cmap=colormaps[sentiment.lower()])
                            ax.set_title(f'{sentiment.capitalize()}', color=sentiment_colors.get(sentiment, 'black'), fontsize=24)
                            ax.axis('off')
                            ax.set_aspect('equal')
                        fig_wordclouds_com.suptitle('Wordcloud across sentiment categories', fontsize=28)
                        # Displaying the wordclouds chart in Streamlit
                        st.pyplot(fig_wordclouds_com)

                    # Defining the charts for 5th row
                    # Getting then-gram charts from the function output
                    ngram_comment_plot = visualize_associated_ngrams(filtered_df, text_column, keyword)
                    # Displaying the ngram chart in Streamlit
                    st.pyplot(ngram_comment_plot, use_container_width=True)
            else:
                st.write(f"No {selected_view.lower()} matching the keyword.")
        else:
            st.error("Selected view not recognized or text column not found.")


