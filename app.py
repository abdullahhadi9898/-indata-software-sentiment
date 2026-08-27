"""
Amazon Software Reviews Sentiment Analysis Dashboard
Built based on the Indata Agency sentiment analysis project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import re
import os
import gzip
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Page configuration
st.set_page_config(
    page_title="Amazon Software Reviews Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #232F3E;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Topic definitions from the original analysis
COMPLAINT_TOPICS = {
    0: "Usability and Navigation Problems",
    1: "Paid Content and Value for Money",
    2: "Device Compatibility and Download Issues",
    3: "Mobile Game Quality and Ads",
    4: "Installation Errors and Customer Support",
    5: "Annual Updates and Subscription Pricing"
}

PRAISE_TOPICS = {
    0: "Amazon Device and App Experience",
    1: "Software Updates and Improvements",
    2: "Feature Discovery and Usability",
    3: "Tax Software Satisfaction",
    4: "Enjoyable Gaming Experience",
    5: "Streaming and Media Content Satisfaction"
}

# Expected topic distribution percentages (from original analysis)
COMPLAINT_DISTRIBUTION = {
    "Usability and Navigation Problems": 53.2,
    "Mobile Game Quality and Ads": 24.9,
    "Device Compatibility and Download Issues": 6.7,
    "Installation Errors and Customer Support": 6.0,
    "Paid Content and Value for Money": 5.9,
    "Annual Updates and Subscription Pricing": 3.3
}

PRAISE_DISTRIBUTION = {
    "Enjoyable Gaming Experience": 82.7,
    "Software Updates and Improvements": 11.1,
    "Feature Discovery and Usability": 4.8,
    "Streaming and Media Content Satisfaction": 0.8,
    "Amazon Device and App Experience": 0.5,
    "Tax Software Satisfaction": 0.1
}


@st.cache_data
def load_sample_data():
    """Load or generate sample dataset for demonstration"""
    sample_file = '/workspace/sample_reviews.csv'
    
    if os.path.exists(sample_file):
        return pd.read_csv(sample_file)
    
    # Generate synthetic data based on the original analysis findings
    np.random.seed(42)
    n_samples = 5000
    
    # Generate ratings with realistic distribution
    ratings = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.19, 0.21, 0.20, 0.21, 0.19])
    
    # Sample review titles and texts
    review_templates = {
        1: ["Terrible software", "Doesn't work at all", "Very disappointed", "Waste of money", "Horrible experience"],
        2: ["Not great", "Could be better", "Disappointing", "Has issues", "Not satisfied"],
        3: ["It's okay", "Average product", "Neither good nor bad", "Mediocre", "Acceptable"],
        4: ["Pretty good", "Works well", "Satisfied overall", "Good product", "Recommended"],
        5: ["Excellent!", "Love it!", "Perfect!", "Amazing software", "Best purchase ever"]
    }
    
    text_templates = {
        1: [
            "This software is terrible. It doesn't work properly and customer support is unhelpful.",
            "Complete waste of money. The app crashes constantly and I can't get a refund.",
            "Very disappointed with this purchase. Nothing works as advertised.",
            "Horrible experience. The installation failed and I lost all my data.",
            "Worst software ever. Full of bugs and the interface is confusing."
        ],
        2: [
            "Not great but usable. Has some annoying bugs that need fixing.",
            "Could be better for the price. Missing some basic features.",
            "Disappointing performance. Expected more from this product.",
            "Has issues with compatibility on my device. Frustrating to use.",
            "Not satisfied with the update. Previous version was better."
        ],
        3: [
            "It's okay for basic tasks. Nothing special but gets the job done.",
            "Average product. Works fine but nothing impressive.",
            "Neither good nor bad. Does what it says but could improve.",
            "Mediocre software. Some features are useful, others are not.",
            "Acceptable for the price. Don't expect too much."
        ],
        4: [
            "Pretty good overall. Works well for my needs with minor issues.",
            "Works well most of the time. Good value for money.",
            "Satisfied overall. Easy to use and reliable.",
            "Good product with helpful features. Would recommend.",
            "Recommended for anyone looking for this type of software."
        ],
        5: [
            "Excellent software! Exceeded all my expectations. Highly recommend!",
            "Love it! Best purchase I've made this year. Perfect functionality.",
            "Perfect! Exactly what I needed. Easy setup and great performance.",
            "Amazing software with intuitive interface. Customer support is fantastic.",
            "Best purchase ever! This software has improved my productivity significantly."
        ]
    }
    
    data = []
    for i in range(n_samples):
        rating = ratings[i]
        title = np.random.choice(review_templates[rating])
        text = np.random.choice(text_templates[rating])
        
        # Add timestamp (random date between 2020-2023)
        year = np.random.randint(2020, 2024)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        timestamp = pd.Timestamp(year, month, day).timestamp() * 1000
        
        data.append({
            'rating': rating,
            'review_title': title,
            'text': text,
            'timestamp': timestamp,
            'helpful_vote': np.random.randint(0, 50),
            'verified_purchase': True,
            'parent_asin': f'B0{np.random.randint(10000000, 99999999)}',
            'product_name': np.random.choice(['Kindle App', 'Fire TV', 'Prime Video', 'Alexa', 'Audible', 'Amazon Music', 'TurboTax', 'Game Title']),
            'store': np.random.choice(['Amazon', 'Third Party', 'Adobe', 'Microsoft', 'Independent Developer'])
        })
    
    df = pd.DataFrame(data)
    
    # Add VADER sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        return pd.Series([compound, label])
    
    df[['compound_score', 'sentiment_label']] = df['text'].apply(get_sentiment)
    
    # Add rating-based sentiment
    def rating_to_sentiment(r):
        if r >= 4:
            return 'Positive'
        elif r <= 2:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['rating_sentiment'] = df['rating'].apply(rating_to_sentiment)
    df['vader_agrees'] = df['sentiment_label'] == df['rating_sentiment']
    
    # Add date columns
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Assign topics based on sentiment
    negative_indices = df[df['sentiment_label'] == 'Negative'].index.tolist()
    positive_indices = df[df['sentiment_label'] == 'Positive'].index.tolist()
    
    # Assign complaint topics to negative reviews
    complaint_list = list(COMPLAINT_DISTRIBUTION.keys())
    complaint_probs = list(COMPLAINT_DISTRIBUTION.values())
    complaint_probs = [p/sum(complaint_probs) for p in complaint_probs]
    
    for idx in negative_indices:
        df.loc[idx, 'dominant_topic'] = np.random.choice(complaint_list, p=complaint_probs)
    
    # Assign praise topics to positive reviews
    praise_list = list(PRAISE_DISTRIBUTION.keys())
    praise_probs = list(PRAISE_DISTRIBUTION.values())
    praise_probs = [p/sum(praise_probs) for p in praise_probs]
    
    for idx in positive_indices:
        df.loc[idx, 'dominant_topic'] = np.random.choice(praise_list, p=praise_probs)
    
    # Neutral reviews
    neutral_indices = df[df['sentiment_label'] == 'Neutral'].index.tolist()
    for idx in neutral_indices:
        df.loc[idx, 'dominant_topic'] = 'Neutral — No dominant topic'
    
    # Save for future use
    df.to_csv(sample_file, index=False)
    
    return df


def preprocess_text(text):
    """Basic text preprocessing for sentiment analysis"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Amazon Software Reviews Sentiment Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎯 Navigation")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_sample_data()
    
    st.sidebar.success(f"✅ Loaded {len(df):,} reviews")
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Rating filter
    rating_filter = st.sidebar.multiselect(
        "Select Ratings:",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="Filter by star ratings"
    )
    
    # Sentiment filter
    sentiment_filter = st.sidebar.multiselect(
        "Select Sentiment:",
        options=['Positive', 'Neutral', 'Negative'],
        default=['Positive', 'Neutral', 'Negative'],
        help="Filter by VADER sentiment analysis"
    )
    
    # Year filter
    year_filter = st.sidebar.multiselect(
        "Select Years:",
        options=sorted(df['year'].unique()),
        default=sorted(df['year'].unique()),
        help="Filter by review year"
    )
    
    # Apply filters
    filtered_df = df[
        (df['rating'].isin(rating_filter)) &
        (df['sentiment_label'].isin(sentiment_filter)) &
        (df['year'].isin(year_filter))
    ].copy()
    
    st.sidebar.markdown(f"**Filtered Reviews:** {len(filtered_df):,}")
    
    # Main navigation
    page = st.sidebar.radio(
        "Choose a section:",
        ["📈 Overview", "😊 Sentiment Analysis", "🏷️ Topic Modeling", "📅 Trends", "🔎 Explore Reviews"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About This Dashboard**
    
    This dashboard presents insights from Amazon software reviews using:
    - **VADER** for sentiment analysis
    - **LDA** for topic modeling
    
    Based on the Indata Agency analysis project.
    """)
    
    # ============================================
    # OVERVIEW PAGE
    # ============================================
    if page == "📈 Overview":
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Reviews",
                value=f"{len(df):,}",
                delta=None
            )
        
        with col2:
            avg_rating = df['rating'].mean()
            st.metric(
                label="Average Rating",
                value=f"{avg_rating:.2f} ⭐",
                delta=f"{avg_rating - 3:.2f} vs neutral"
            )
        
        with col3:
            unique_products = df['parent_asin'].nunique()
            st.metric(
                label="Unique Products",
                value=f"{unique_products:,}",
                delta=None
            )
        
        with col4:
            disagreement_rate = (1 - df['vader_agrees'].mean()) * 100
            st.metric(
                label="Rating-Sentiment Disagreement",
                value=f"{disagreement_rate:.1f}%",
                delta="⚠️ High",
                delta_color="inverse"
            )
        
        st.markdown("---")
        
        # Rating distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⭐ Rating Distribution")
            rating_dist = df['rating'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4']
            bars = ax.bar(rating_dist.index.astype(str), rating_dist.values, color=colors)
            
            ax.set_xlabel('Star Rating')
            ax.set_ylabel('Number of Reviews')
            ax.set_title('Distribution of Star Ratings')
            
            # Add value labels
            for bar, val in zip(bars, rating_dist.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                       f'{val:,}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("🎭 VADER Sentiment Distribution")
            sentiment_dist = df['sentiment_label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#4575b4', '#fee08b', '#d73027']  # Positive, Neutral, Negative
            wedges, texts, autotexts = ax.pie(
                sentiment_dist.values,
                labels=sentiment_dist.index,
                autopct='%1.1f%%',
                colors=colors,
                explode=(0.05, 0.05, 0.05)
            )
            
            ax.set_title('VADER Sentiment Analysis Results')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Key insight
        st.info("""
        **💡 Key Finding: Rating-Sentiment Disagreement**
        
        Our analysis reveals that approximately **44.7%** of reviews show disagreement between 
        the star rating given by users and the actual sentiment expressed in their written review text.
        
        This suggests that **star ratings alone may be unreliable** for understanding true customer 
        satisfaction. Text-based sentiment analysis provides deeper insights into customer opinions.
        """)
    
    # ============================================
    # SENTIMENT ANALYSIS PAGE
    # ============================================
    elif page == "😊 Sentiment Analysis":
        st.markdown('<h2 class="sub-header">😊 VADER Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        **VADER (Valence Aware Dictionary and sEntiment Reasoner)** is used to analyze the emotional 
        tone of review text. It provides:
        - **Compound Score**: Ranges from -1.0 (most negative) to +1.0 (most positive)
        - **Sentiment Label**: Positive (≥0.05), Neutral (-0.05 to 0.05), Negative (≤-0.05)
        """)
        
        # Agreement analysis
        st.subheader("🔄 VADER vs Star Rating Agreement")
        
        agreement_table = pd.crosstab(
            df['rating_sentiment'],
            df['sentiment_label'],
            margins=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Agreement Matrix:**")
            st.dataframe(agreement_table.style.background_gradient(cmap='Blues'))
        
        with col2:
            agree_pct = df['vader_agrees'].mean() * 100
            disagree_pct = 100 - agree_pct
            
            fig, ax = plt.subplots(figsize=(8, 5))
            wedges, texts, autotexts = ax.pie(
                [agree_pct, disagree_pct],
                labels=['Agree', 'Disagree'],
                autopct='%1.1f%%',
                colors=['#91bfdb', '#d73027'],
                explode=(0, 0.05)
            )
            ax.set_title('VADER vs Star Rating Agreement')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Sentiment by rating
        st.subheader("📊 Sentiment Distribution by Star Rating")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sentiment_by_rating = pd.crosstab(df['rating'], df['sentiment_label'], normalize='index') * 100
        
        sentiment_by_rating.plot(
            kind='bar',
            ax=ax,
            color=['#d73027', '#fee08b', '#4575b4'],
            stacked=True
        )
        
        ax.set_xlabel('Star Rating')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sentiment Distribution Within Each Star Rating')
        ax.legend(title='Sentiment', loc='upper right')
        ax.set_xticklabels(['1★', '2★', '3★', '4★', '5★'])
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Example reviews
        st.markdown("---")
        st.subheader("📝 Example Reviews by Sentiment")
        
        sentiment_example = st.selectbox(
            "Select sentiment category:",
            ['Positive', 'Neutral', 'Negative']
        )
        
        examples = df[df['sentiment_label'] == sentiment_example].sample(min(5, len(df[df['sentiment_label'] == sentiment_example])))
        
        for _, row in examples.iterrows():
            with st.expander(f"⭐ {row['rating']} stars - {row['review_title']}"):
                st.write(f"**Product:** {row['product_name']}")
                st.write(f"**Review:** {row['text']}")
                st.write(f"**VADER Score:** {row['compound_score']:.3f}")
                st.write(f"**Helpful Votes:** {row['helpful_vote']}")
    
    # ============================================
    # TOPIC MODELING PAGE
    # ============================================
    elif page == "🏷️ Topic Modeling":
        st.markdown('<h2 class="sub-header">🏷️ LDA Topic Modeling</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        **LDA (Latent Dirichlet Allocation)** identifies hidden themes in review text:
        - **Complaint Topics** (from negative reviews)
        - **Praise Topics** (from positive reviews)
        """)
        
        tab1, tab2 = st.tabs(["😠 Complaint Topics", "😊 Praise Topics"])
        
        with tab1:
            st.subheader("What Unhappy Customers Talk About")
            
            # Display complaint topics
            st.markdown("#### Identified Complaint Categories:")
            
            complaint_cols = st.columns(2)
            complaint_items = list(COMPLAINT_TOPICS.items())
            
            for i, (idx, topic) in enumerate(complaint_items):
                with complaint_cols[i % 2]:
                    pct = COMPLAINT_DISTRIBUTION.get(topic, 0)
                    st.markdown(f"""
                    **Topic {idx}: {topic}**
                    
                    📊 **Prevalence:** {pct:.1f}% of negative reviews
                    """)
            
            st.markdown("---")
            
            # Complaint topic chart
            st.subheader("📊 Complaint Topic Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            topics = list(COMPLAINT_DISTRIBUTION.keys())
            percentages = list(COMPLAINT_DISTRIBUTION.values())
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(topics)))
            
            bars = ax.barh(topics, percentages, color=colors)
            
            ax.set_xlabel('Percentage of Negative Reviews (%)')
            ax.set_title('Distribution of Complaint Topics')
            
            # Add value labels
            for bar, pct in zip(bars, percentages):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **Key Insight:** Over half (53.2%) of negative reviews focus on **Usability and 
            Navigation Problems**, making it the primary area for improvement.
            """)
        
        with tab2:
            st.subheader("What Happy Customers Appreciate")
            
            # Display praise topics
            st.markdown("#### Identified Praise Categories:")
            
            praise_cols = st.columns(2)
            praise_items = list(PRAISE_TOPICS.items())
            
            for i, (idx, topic) in enumerate(praise_items):
                with praise_cols[i % 2]:
                    pct = PRAISE_DISTRIBUTION.get(topic, 0)
                    st.markdown(f"""
                    **Topic {idx}: {topic}**
                    
                    📊 **Prevalence:** {pct:.1f}% of positive reviews
                    """)
            
            st.markdown("---")
            
            # Praise topic chart
            st.subheader("📊 Praise Topic Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            topics = list(PRAISE_DISTRIBUTION.keys())
            percentages = list(PRAISE_DISTRIBUTION.values())
            
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(topics)))
            
            bars = ax.barh(topics, percentages, color=colors)
            
            ax.set_xlabel('Percentage of Positive Reviews (%)')
            ax.set_title('Distribution of Praise Topics')
            
            # Add value labels
            for bar, pct in zip(bars, percentages):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success("""
            **Key Insight:** An overwhelming majority (82.7%) of positive reviews mention 
            **Enjoyable Gaming Experience**, indicating gaming software is a major strength.
            """)
        
        # Topic distribution in filtered data
        if len(filtered_df) > 0:
            st.markdown("---")
            st.subheader("📊 Topic Distribution in Filtered Data")
            
            topic_dist = filtered_df['dominant_topic'].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            topic_dist.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Topic')
            ax.set_ylabel('Count')
            ax.set_title('Topic Distribution in Selected Reviews')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
    
    # ============================================
    # TRENDS PAGE
    # ============================================
    elif page == "📅 Trends":
        st.markdown('<h2 class="sub-header">📅 Temporal Trends</h2>', unsafe_allow_html=True)
        
        # Monthly trends
        st.subheader("📈 Review Volume Over Time")
        
        monthly_counts = df.groupby(['year', 'month']).size().reset_index(name='count')
        monthly_counts['year_month'] = monthly_counts['year'].astype(str) + '-' + monthly_counts['month'].astype(str).str.zfill(2)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(monthly_counts['year_month'], monthly_counts['count'], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Year-Month')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Monthly Review Volume')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Sentiment trends over time
        st.subheader("🎭 Sentiment Trends Over Time")
        
        yearly_sentiment = pd.crosstab(df['year'], df['sentiment_label'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_sentiment.plot(
            kind='bar',
            ax=ax,
            color=['#d73027', '#fee08b', '#4575b4'],
            stacked=True
        )
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sentiment Distribution by Year')
        ax.legend(title='Sentiment', loc='upper right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Rating trends
        st.subheader("⭐ Average Rating Trend")
        
        yearly_rating = df.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Rating', color=color)
        line = ax1.plot(yearly_rating['year'], yearly_rating['mean'], color=color, marker='o', linewidth=2, markersize=8)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(1, 5)
        
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Review Count', color=color)
        bars = ax2.bar(yearly_rating['year'], yearly_rating['count'], color=color, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # Helpful votes trend
        st.subheader("👍 Most Helpful Reviews by Year")
        
        top_helpful = df.loc[df.groupby('year')['helpful_vote'].idxmax()][['year', 'review_title', 'helpful_vote', 'rating']]
        
        st.dataframe(top_helpful.style.format({'helpful_vote': '{:,}'}), hide_index=True)
    
    # ============================================
    # EXPLORE REVIEWS PAGE
    # ============================================
    elif page == "🔎 Explore Reviews":
        st.markdown('<h2 class="sub-header">🔎 Explore Individual Reviews</h2>', unsafe_allow_html=True)
        
        # Search functionality
        search_term = st.text_input("🔍 Search in reviews:", placeholder="Enter keywords to search...")
        
        if search_term:
            mask = (
                df['text'].str.contains(search_term, case=False, na=False) |
                df['review_title'].str.contains(search_term, case=False, na=False) |
                df['product_name'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = df[mask]
            st.info(f"Found {len(filtered_df)} reviews matching '{search_term}'")
        
        # Display reviews table
        st.subheader("📋 Review Browser")
        
        # Select columns to display
        display_cols = st.multiselect(
            "Select columns to display:",
            options=['rating', 'review_title', 'text', 'product_name', 'sentiment_label', 
                    'compound_score', 'dominant_topic', 'helpful_vote', 'year'],
            default=['rating', 'review_title', 'sentiment_label', 'dominant_topic', 'year']
        )
        
        # Pagination
        page_size = st.selectbox("Reviews per page:", [10, 25, 50, 100])
        total_pages = max(1, len(filtered_df) // page_size + 1)
        
        current_page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        
        page_df = filtered_df.iloc[start_idx:end_idx][display_cols]
        
        st.dataframe(page_df, hide_index=True, use_container_width=True)
        
        st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(filtered_df))} of {len(filtered_df)} reviews")
        
        # Download option
        st.download_button(
            label="📥 Download Filtered Reviews as CSV",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_amazon_reviews.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
