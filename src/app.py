from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from io import BytesIO
import base64
import nltk

nltk.download('vader_lexicon')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../Data/uploads'
app.config['DATA_FOLDER'] = '../Data'

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['DATA_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_existing_products():
    """Get list of already analyzed products from Data folder"""
    products = []
    for filename in os.listdir(app.config['DATA_FOLDER']):
        if filename.startswith('sentiment_') and filename.endswith('_reviews.csv'):
            product = filename[len('sentiment_'):-len('_reviews.csv')]
            products.append(product)
    return products

def classify_sentiment(score):
    """Classify sentiment based on compound score"""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def update_summary(product_name, df):
    """Update sentiment summary with new product data"""
    summary_path = os.path.join(app.config['DATA_FOLDER'], 'sentiment_summary.csv')
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create new summary entry
    new_entry = {
        'Product': product_name,
        'Positive': sentiment_counts.get('Positive', 0),
        'Neutral': sentiment_counts.get('Neutral', 0),
        'Negative': sentiment_counts.get('Negative', 0),
        'Pie_Chart': f"{product_name}_pie_chart.png"
    }
    
    # Update or create summary file
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = pd.DataFrame(columns=['Product', 'Positive', 'Neutral', 'Negative', 'Pie_Chart'])
    
    summary_df = pd.concat([summary_df, pd.DataFrame([new_entry])], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    
    # Generate pie chart
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    df['sentiment'].value_counts().plot.pie(
        autopct='%1.1f%%',
        colors=['#9CDEB7', '#F6846F', '#FFEB66'],
        startangle=90,
        ax=ax
    )
    ax.set_title(f'Sentiment Distribution for {product_name}')
    ax.set_ylabel('')
    
    plt.savefig(os.path.join(app.config['DATA_FOLDER'], new_entry['Pie_Chart']))
    plt.close(fig)  # Explicitly close the figure

@app.route('/')
def index():
    """Main page with product selection"""
    products = get_existing_products()
    return render_template('index.html', products=products)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle analysis requests"""
    # Check for existing product selection
    if 'product' in request.form and request.form['product'] != "":
        product = request.form['product']
        existing_products = get_existing_products()
        
        if product not in existing_products:
            return jsonify({"error": "Product not found or not analyzed yet"}), 400
            
        # Load existing analysis
        file_path = os.path.join(app.config['DATA_FOLDER'], f'sentiment_{product}_reviews.csv')
        df = pd.read_csv(file_path)
        results = df.where(pd.notnull(df), None).to_dict('records')
        plot_url = generate_plot(df)
        
        return jsonify({
            "results": results,
            "plot_url": plot_url,
            "message": "Loaded existing analysis"
        })

    # Handle new file upload
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate filename format
        if not file.filename.endswith('_reviews.csv'):
            return jsonify({"error": "Filename must follow [product]_reviews.csv format"}), 400
            
        product_name = file.filename[:-len('_reviews.csv')]
        existing_products = get_existing_products()
        
        if product_name in existing_products:
            return jsonify({"error": "Product already analyzed"}), 400

        # Process the file
        # In your analyze route's file processing section:
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Read CSV with explicit handling of missing values
            df = pd.read_csv(file_path).fillna('')
            
            if 'review_text' not in df.columns:
                return jsonify({"error": "CSV must contain 'review_text' column"}), 400
                
            # Perform sentiment analysis with NaN handling
            df['compound_score'] = df['review_text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['compound'] if str(x).strip() != '' else 0
            )
            df['sentiment'] = df['compound_score'].apply(classify_sentiment)
            
            # Replace any remaining NaN/NaT values
            df = df.replace([np.nan, pd.NaT], None)
            
            # Save results
            output_path = os.path.join(
                app.config['DATA_FOLDER'],
                f'sentiment_{product_name}_reviews.csv'
            )
            df.to_csv(output_path, index=False)
            
            # Update summary and generate charts
            update_summary(product_name, df)
            
            # Generate plot URL
            plot_url = generate_plot(df)
            
            return jsonify({
                "results": df.where(pd.notnull(df), None).to_dict('records'),
                "plot_url": plot_url,
                "message": "New analysis completed and saved"
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return jsonify({"error": "No valid request received"}), 400

def generate_plot(df):
    """Generate base64 encoded plot image"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    df['sentiment'].value_counts().plot(kind='bar', 
                                      color=['#9CDEB7', '#FFEB66', '#F6846F'],
                                      ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contributors')
def contributors():
    return render_template('contributors.html')

if __name__ == '__main__':
    app.run(debug=True)