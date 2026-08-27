# Amazon Software Reviews Sentiment Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing Amazon software reviews using VADER sentiment analysis and LDA topic modeling.

## 📊 Features

### Dashboard Sections

1. **📈 Overview**
   - Key metrics (total reviews, average rating, unique products)
   - Rating distribution visualization
   - VADER sentiment distribution
   - Rating-Sentiment disagreement analysis

2. **😊 Sentiment Analysis**
   - VADER vs Star Rating agreement matrix
   - Sentiment distribution by star rating
   - Example reviews by sentiment category

3. **🏷️ Topic Modeling**
   - **Complaint Topics** (from negative reviews):
     - Usability and Navigation Problems (53.2%)
     - Mobile Game Quality and Ads (24.9%)
     - Device Compatibility and Download Issues (6.7%)
     - Installation Errors and Customer Support (6.0%)
     - Paid Content and Value for Money (5.9%)
     - Annual Updates and Subscription Pricing (3.3%)
   
   - **Praise Topics** (from positive reviews):
     - Enjoyable Gaming Experience (82.7%)
     - Software Updates and Improvements (11.1%)
     - Feature Discovery and Usability (4.8%)
     - Streaming and Media Content Satisfaction (0.8%)
     - Amazon Device and App Experience (0.5%)
     - Tax Software Satisfaction (0.1%)

4. **📅 Trends**
   - Monthly review volume
   - Sentiment trends over time
   - Average rating trends
   - Most helpful reviews by year

5. **🔎 Explore Reviews**
   - Search functionality
   - Filterable review browser
   - CSV download option

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app.py
```

4. Open your browser and navigate to:
```
http://localhost:8501
```

## 📁 Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `indata_sentiment_analysis_.ipynb` - Original Jupyter notebook analysis
- `sample_reviews.csv` - Generated sample dataset (created on first run)

## 🔧 Configuration

The dashboard includes interactive filters in the sidebar:
- **Rating Filter**: Select specific star ratings (1-5)
- **Sentiment Filter**: Filter by VADER sentiment (Positive/Neutral/Negative)
- **Year Filter**: Select review years

## 📊 Methodology

### VADER Sentiment Analysis
- **Compound Score**: Ranges from -1.0 (most negative) to +1.0 (most positive)
- **Sentiment Labels**:
  - Positive: compound ≥ 0.05
  - Neutral: -0.05 < compound < 0.05
  - Negative: compound ≤ -0.05

### Key Finding
Approximately **44.7%** of reviews show disagreement between star ratings and text sentiment, indicating that star ratings alone may be unreliable for understanding true customer satisfaction.

## 📈 Based On

This dashboard is built based on the Indata Agency sentiment analysis project, which analyzed 75,593 Amazon software reviews using:
- VADER for sentiment scoring
- LDA (Latent Dirichlet Allocation) for topic modeling

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
