import re
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import nltk
from nltk.corpus import stopwords
import requests
from dotenv import load_dotenv

# Load environment variables from .env file at module startup
load_dotenv()

# Download NLTK data (stopwords)
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Retrospectiva Grupo Camburou",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Name mapping dictionary (from notebook cell 10)
NAME_MAP = {
    'Alexandre Sadaka': ['Arex'],
    'Andre Stolar': ['Deco Stolar'],
    'Ariel Hacham': ['Hacham'],
    'Bruno Menache': ['Bruno Menache'],
    'Bruno Skorkowski': ['Bubis', 'Bruno Skorkowski'],
    'Bruno Stisin': ['Bruno Stisin', r'â\x80\x8eVocÃª', r'âVocÃª', r'âYou', 'âYou', '‎You', r'‎You'],
    'Daniel Farina': ['Dummyts', r'~â\x80¯Daniel Turkie Farina', 'Dani Faras'],
    'Daniel Mesnik': ['Mesnik'],
    'David Cohen': ['David Cohen'],
    'Felipe Getz': ['Getzinho'],
    'Gabriel Tkacz': ['Tkacz'],
    'Leon Grinberg': ['Leon Grimberg'],
    'Leonardo Mandelbaum': ['Leo Mandelbaum'],
    'Paulo Sutiak': ['Paulo Sutiak'],
    'Rafael Thalenberg': ['Rafael Thalenberg'],
    'Rafael Turecki': ['Turecki'],
    'Raphael Ulrych': ['Raphael Ulrych'],
    'Ricardo Breinis': ['Ricardo Breinis'],
    'Tiago Hudler': ['Ticaega'],
    'William Gottesmann': ['William', r'~â\x80¯William', r'~â¯William'],
    'Yuri Marchette': ['Yuri'],
    'IGNORE': ['', 'Meta AI', 'PDA - Camburou - DPA'],
}


def convert_google_drive_url(url):
    """Convert Google Drive sharing URL to direct download URL.
    
    Args:
        url: Google Drive sharing URL (e.g., https://drive.google.com/file/d/FILE_ID/view?usp=sharing)
    
    Returns:
        Direct download URL or original URL if not a Google Drive URL
    """
    import re
    # Pattern to match Google Drive file URLs
    pattern = r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)'
    match = re.search(pattern, url)
    if match:
        file_id = match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url


@st.cache_data
def load_and_parse_messages(chat_file_url):
    """Load and parse messages from _chat.txt file downloaded from URL.
    
    Args:
        chat_file_url: URL to download the chat file from
    
    Returns:
        tuple: (DataFrame, set of unrecognized names)
    """
    # Convert Google Drive sharing URL to direct download URL if needed
    download_url = convert_google_drive_url(chat_file_url)
    
    # Download the file from URL
    try:
        # Use a session to handle cookies (needed for Google Drive)
        session = requests.Session()
        response = session.get(download_url, timeout=30, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Check if we got HTML instead of the file (Google Drive sometimes returns HTML)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            # If it's HTML, we might have hit a virus scan warning page
            # Try to get the actual download link
            if 'download' in response.text.lower() or 'virus' in response.text.lower():
                # Extract the actual download link from the HTML
                import re
                download_match = re.search(r'href="(/uc\?export=download[^"]+)"', response.text)
                if download_match:
                    actual_url = 'https://drive.google.com' + download_match.group(1)
                    response = session.get(actual_url, timeout=30, allow_redirects=True)
                    response.raise_for_status()
                else:
                    raise Exception(
                        f"Google Drive returned an HTML page instead of the file. "
                        f"This might be a virus scan warning. Please try: "
                        f"1. Right-click the file in Google Drive > Get link > Change to 'Anyone with the link' > Viewer\n"
                        f"2. Or use the direct download URL format: https://drive.google.com/uc?export=download&id=FILE_ID"
                    )
    except requests.exceptions.RequestException as e:
        raise Exception(
            f"Error downloading '_chat.txt' from URL '{chat_file_url}': {str(e)}\n"
            f"Download URL used: '{download_url}'\n"
            f"Please check that the URL is correct and accessible."
        )
    
    # Decode the content using latin-1 encoding
    try:
        content = response.content.decode('latin-1')
        data_lines = content.splitlines(keepends=True)
    except UnicodeDecodeError as e:
        raise Exception(f"Error decoding downloaded file content: {str(e)}")
    
    if not data_lines:
        raise ValueError("The downloaded '_chat.txt' file is empty.")
    
    # Filter valid messages using regex
    parsed_data = [item for item in data_lines if use_regex(item)]
    
    if not parsed_data:
        raise ValueError("No valid messages found matching the expected format.")
    
    # Parse into DataFrame
    raw_df_data = {'date': [], 'raw_name': [], 'parsed_name': [], 'raw_message': []}
    unrecognized_names = set()
    
    for item in parsed_data:
        try:
            # Find the closing bracket of the date
            bracket_end = item.find(']')
            if bracket_end == -1:
                continue
            
            # Extract date: everything between [ and ]
            date_str = item[1:bracket_end]
            
            # Extract name: everything after ] and before the first :
            name_part = item[bracket_end + 1:].split(':', 1)
            if len(name_part) < 2:
                continue
            
            raw_name = name_part[0].strip()
            parsed_name = parse_name(raw_name)
            
            # Track unrecognized names (but not IGNORE entries)
            if parsed_name is None:
                unrecognized_names.add(raw_name)
            
            raw_df_data['date'].append(date_str)
            raw_df_data['raw_name'].append(raw_name)
            raw_df_data['parsed_name'].append(parsed_name)
            raw_df_data['raw_message'].append(item)
        except (IndexError, ValueError, AttributeError):
            # Skip malformed lines
            continue
    
    # Create DataFrame
    if not raw_df_data['date']:
        raise ValueError("No messages could be parsed from the file.")
    
    df = pd.DataFrame(raw_df_data)
    # Match notebook exactly: replace \n with empty string (not regex pattern)
    df['raw_message'] = df['raw_message'].str.replace(r'\n', '', regex=True)
    
    # Filter unwanted messages (same order as notebook)
    df = df.drop(df[df.apply(should_drop, axis=1)].index)
    # Filter out None and IGNORE entries
    df = df[df['parsed_name'].notna()]
    df = df[df['parsed_name'] != 'IGNORE']
    
    if df.empty:
        # Provide more debugging info
        debug_info = {
            'total_parsed': len(parsed_data),
            'total_after_parsing': len(raw_df_data['date']),
            'unique_raw_names': len(set(raw_df_data['raw_name'])),
            'sample_raw_names': list(set(raw_df_data['raw_name']))[:10]
        }
        raise ValueError(
            f"All messages were filtered out. Check name mapping and filter logic.\n"
            f"Debug: {debug_info}"
        )
    
    # Convert date - match notebook format exactly (notebook uses space, but actual string has comma)
    # Replace comma with space to match notebook format string
    df['date'] = df['date'].str.replace(', ', ' ', regex=False)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['date'])
    
    if df.empty:
        raise ValueError("All dates failed to parse. Check date format.")
    
    # Parse message content - match notebook exactly (uses .str[3], not [3:])
    df['parsed_message'] = df['raw_message'].str.split(':').str[3].str.strip()
    df['parsed_message'] = df['parsed_message'].str.encode('latin-1', errors='ignore').str.decode('utf-8', errors='ignore')
    df['parsed_message'] = df['parsed_message'].str.upper()
    
    # Extract year
    df['year'] = pd.DatetimeIndex(df['date']).year
    
    return df, unrecognized_names


def use_regex(input_text):
    """Check if input text matches the message pattern."""
    pattern = re.compile(r"\[[^\]]*\]", re.IGNORECASE)
    return pattern.match(input_text)


def should_drop(row):
    """Determine if a message should be dropped."""
    input_text = row['raw_message']
    return not((input_text != '.') and 
               (not re.match(r'.*[Kk]{2,}\Z', input_text)) and 
               (not re.match(r'(?i).*haha+.*', input_text)))


def parse_name(raw_name):
    """Map raw name to normalized name."""
    # Check IGNORE entry first
    if 'IGNORE' in NAME_MAP and raw_name in NAME_MAP['IGNORE']:
        return 'IGNORE'
    
    # Check other entries
    for key, value in NAME_MAP.items():
        if key != 'IGNORE' and raw_name in value:
            return key
    return None


def filter_data(df, selected_years, selected_persons):
    """Filter DataFrame based on selected years and persons."""
    filtered_df = df.copy()
    
    if selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
    
    if selected_persons:
        filtered_df = filtered_df[filtered_df['parsed_name'].isin(selected_persons)]
    
    return filtered_df


def calculate_metrics(df):
    """Calculate overview metrics."""
    total_messages = len(df)
    unique_participants = df['parsed_name'].nunique()
    avg_messages_per_person = total_messages / unique_participants if unique_participants > 0 else 0
    
    return {
        'total_messages': total_messages,
        'unique_participants': unique_participants,
        'avg_messages_per_person': avg_messages_per_person
    }


def group_by_time_period(df, period_type):
    """Group messages by different time periods.
    
    Args:
        df: DataFrame with 'date' column
        period_type: One of 'month', 'trimester', 'quarter', 'semester'
    
    Returns:
        DataFrame with 'period' and 'count' columns, sorted chronologically
    """
    df_copy = df.copy()
    
    if period_type == 'month':
        # Group by year-month
        df_copy['period'] = df_copy['date'].dt.to_period('M')
        period_counts = df_copy.groupby('period').size().reset_index(name='count')
        period_counts['period'] = period_counts['period'].astype(str)
        
    elif period_type == 'trimester':
        # Group by 4-month periods: Jan-Apr (T1), May-Aug (T2), Sep-Dec (T3)
        def get_trimester(date):
            month = date.month
            year = date.year
            if month <= 4:
                return f"{year} T1"
            elif month <= 8:
                return f"{year} T2"
            else:
                return f"{year} T3"
        
        df_copy['period'] = df_copy['date'].apply(get_trimester)
        period_counts = df_copy.groupby('period').size().reset_index(name='count')
        
    elif period_type == 'quarter':
        # Group by 3-month quarters: Q1 (Jan-Mar), Q2 (Apr-Jun), Q3 (Jul-Sep), Q4 (Oct-Dec)
        df_copy['period'] = df_copy['date'].dt.to_period('Q')
        period_counts = df_copy.groupby('period').size().reset_index(name='count')
        # Format as "2024 Q1" instead of "2024Q1"
        period_counts['period'] = period_counts['period'].astype(str).str.replace('Q', ' Q')
        
    elif period_type == 'semester':
        # Group by 6-month periods: Jan-Jun (S1), Jul-Dec (S2)
        def get_semester(date):
            month = date.month
            year = date.year
            if month <= 6:
                return f"{year} S1"
            else:
                return f"{year} S2"
        
        df_copy['period'] = df_copy['date'].apply(get_semester)
        period_counts = df_copy.groupby('period').size().reset_index(name='count')
        
    else:
        raise ValueError(f"Unknown period_type: {period_type}")
    
    # Sort chronologically
    # Extract year and period number for sorting
    def sort_key(period_str):
        # Check if it's month format "YYYY-MM" first
        if '-' in period_str and len(period_str.split('-')) == 2:
            # Month format: "2025-01"
            parts = period_str.split('-')
            year = int(parts[0])
            month = int(parts[1])
            return (year, month)
        
        # For other formats: "2024 Q1", "2024 T1", "2024 S1"
        parts = period_str.split()
        year = int(parts[0])
        if len(parts) > 1:
            period_id = parts[1]
            # Map period identifiers to numbers for sorting
            if period_id.startswith('T'):
                period_num = int(period_id[1])
            elif period_id.startswith('Q'):
                period_num = int(period_id[1])
            elif period_id.startswith('S'):
                period_num = int(period_id[1])
            else:
                period_num = 0
            return (year, period_num)
        else:
            return (year, 0)
    
    period_counts['sort_key'] = period_counts['period'].apply(sort_key)
    period_counts = period_counts.sort_values('sort_key')
    period_counts = period_counts.drop('sort_key', axis=1).reset_index(drop=True)
    
    return period_counts


def get_stopwords():
    """Get list of common stopwords in Portuguese and English using NLTK."""
    # Get Portuguese stopwords from NLTK
    try:
        pt_stopwords = set(stopwords.words('portuguese'))
        # Convert to uppercase to match existing logic
        pt_stopwords_upper = {word.upper() for word in pt_stopwords}
    except:
        pt_stopwords_upper = set()
    
    # Get English stopwords from NLTK
    try:
        en_stopwords = set(stopwords.words('english'))
        # Convert to uppercase to match existing logic
        en_stopwords_upper = {word.upper() for word in en_stopwords}
    except:
        en_stopwords_upper = set()
    
    # Combine both sets
    all_stopwords = pt_stopwords_upper.union(en_stopwords_upper)
    
    # Add some additional common words that might be missing
    additional_stopwords = {
        'KK', 'KKK', 'KKKK', 'KKKKK', 'HAHA', 'HAHAHA', 'RS', 'RSRS', 'RSRSRS'
    }
    
    return all_stopwords.union(additional_stopwords)


def generate_word_cloud(df):
    """Generate word cloud from messages."""
    # Combine all messages
    all_text = ' '.join(df['parsed_message'].dropna().astype(str))
    
    # Split into words and filter
    words = re.findall(r'\b\w+\b', all_text.upper())
    stopwords_set = get_stopwords()
    
    # Filter stopwords and short words
    filtered_words = [w for w in words if w not in stopwords_set and len(w) > 2]
    
    if not filtered_words:
        return None
    
    # Count word frequencies
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Generate word cloud with improved settings
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        max_words=200,
        colormap='plasma',
        relative_scaling=0.3,
        prefer_horizontal=0.7,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    return wordcloud


def convert_to_percentage(pivot_df):
    """Convert absolute values in pivot table to row percentages."""
    # Calculate row sums
    row_sums = pivot_df.sum(axis=1)
    # Convert to percentage (avoid division by zero)
    percentage_df = pivot_df.div(row_sums, axis=0) * 100
    percentage_df = percentage_df.fillna(0)
    return percentage_df


def convert_to_percentage_per_week(pivot_df):
    """Convert absolute values in pivot table to percentages relative to entire week."""
    total_sum = pivot_df.sum().sum()
    percentage_df = (pivot_df / total_sum * 100) if total_sum > 0 else pivot_df * 0
    percentage_df = percentage_df.fillna(0)
    return percentage_df


def get_most_common_word_per_hour_day(df):
    """Find the most common word for each hour/day combination."""
    stopwords_set = get_stopwords()
    results = []
    
    # Group by day and hour
    for day_name in df['day_name'].unique():
        day_data = df[df['day_name'] == day_name]
        for hour in sorted(day_data['hour'].unique()):
            hour_data = day_data[day_data['hour'] == hour]
            
            # Extract words from messages in this hour/day
            all_words = []
            for message in hour_data['parsed_message'].dropna():
                words = re.findall(r'\b\w+\b', str(message).upper())
                filtered_words = [w for w in words if w not in stopwords_set and len(w) > 2]
                all_words.extend(filtered_words)
            
            if all_words:
                # Count word frequencies
                word_counts = pd.Series(all_words).value_counts()
                most_common_word = word_counts.index[0]
                most_common_count = word_counts.iloc[0]
                results.append({
                    'day_name': day_name,
                    'hour': hour,
                    'most_common_word': most_common_word,
                    'count': most_common_count
                })
            else:
                results.append({
                    'day_name': day_name,
                    'hour': hour,
                    'most_common_word': '-',
                    'count': 0
                })
    
    return pd.DataFrame(results)


def main():
    st.title("Retrospectiva Grupo Camburou")
    st.markdown("---")
    
    # Read CHAT_FILE_URL from environment variable or Streamlit secrets (outside cached function)
    # First try .env file (via os.getenv), then fall back to Streamlit secrets
    chat_file_url = os.getenv('CHAT_FILE_URL', '').strip()
    
    # If not found in .env, try Streamlit secrets
    if not chat_file_url:
        try:
            # Check if st.secrets is available and contains CHAT_FILE_URL
            if hasattr(st, 'secrets') and 'CHAT_FILE_URL' in st.secrets:
                chat_file_url = str(st.secrets['CHAT_FILE_URL']).strip()
        except (AttributeError, KeyError, FileNotFoundError, TypeError):
            # st.secrets might not be available or CHAT_FILE_URL not in secrets
            chat_file_url = ''
    
    # Check if URL is set before calling cached function
    if not chat_file_url:
        st.error("CHAT_FILE_URL environment variable is not set.")
        st.info("Certifique-se de que a variável CHAT_FILE_URL está configurada no arquivo .env ou nas configurações de secrets do Streamlit")
        
        # Show debugging information
        st.markdown("### CHAT_FILE_URL Debugging")
        chat_file_url_value = os.getenv('CHAT_FILE_URL')
        chat_file_url_stripped = os.getenv('CHAT_FILE_URL', '').strip()
        
        # Check Streamlit secrets
        secrets_value = None
        secrets_stripped = ''
        try:
            secrets_value = st.secrets.get('CHAT_FILE_URL')
            secrets_stripped = str(secrets_value).strip() if secrets_value else ''
        except (AttributeError, KeyError, FileNotFoundError):
            pass
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**os.getenv('CHAT_FILE_URL'):**")
            if chat_file_url_value is None:
                st.error(f"`None` (not found in .env)")
            else:
                st.success(f"`{repr(chat_file_url_value)}`")
        with col2:
            st.write("**st.secrets.get('CHAT_FILE_URL'):**")
            if secrets_value is None:
                st.error(f"`None` (not found in secrets)")
            else:
                st.success(f"`{repr(secrets_value)}`")
        with col3:
            st.write("**Final value (after fallback):**")
            if not chat_file_url_stripped and not secrets_stripped:
                st.error(f"`{repr('')}` (not found in either)")
            elif chat_file_url_stripped:
                st.success(f"`{repr(chat_file_url_stripped)}` (from .env)")
            else:
                st.success(f"`{repr(secrets_stripped)}` (from secrets)")
        
        # Check for related environment variables
        env_vars = {k: v for k, v in os.environ.items()}
        chat_related = {k: v for k, v in env_vars.items() if 'CHAT' in k.upper() or 'URL' in k.upper()}
        if chat_related:
            st.write("**Environment variables containing 'CHAT' or 'URL':**")
            chat_df = pd.DataFrame([
                {"Variable": k, "Value": v} for k, v in sorted(chat_related.items())
            ])
            st.dataframe(chat_df, width='stretch', hide_index=True)
        
        # Check if CHAT_FILE_URL exists in os.environ
        if 'CHAT_FILE_URL' in env_vars:
            st.success(f"**Found in os.environ:** `CHAT_FILE_URL` = `{repr(env_vars['CHAT_FILE_URL'])}`")
            if chat_file_url_value is None:
                st.warning("💡 **Issue:** The variable exists in os.environ but `os.getenv('CHAT_FILE_URL')` returned `None`. This is likely a Streamlit caching issue.")
                st.info("**Solution:** Try clearing the Streamlit cache (Settings > Clear cache) or restarting the Streamlit server.")
            elif not chat_file_url_stripped:
                st.warning("💡 **Issue:** The variable exists but is empty or contains only whitespace after stripping.")
        else:
            # Check for case-insensitive match
            chat_file_url_variants = {k: v for k, v in env_vars.items() if k.upper() == 'CHAT_FILE_URL'}
            if chat_file_url_variants:
                variant_key = list(chat_file_url_variants.keys())[0]
                st.warning(f"**Found in os.environ with different case:** `{variant_key}` = `{repr(chat_file_url_variants[variant_key])}`")
                st.info("💡 **Tip:** Environment variable names are case-sensitive. Make sure your .env file uses exactly `CHAT_FILE_URL` (all uppercase).")
            else:
                st.error("**CHAT_FILE_URL not found in os.environ at all.**")
        
        st.markdown("---")
        
        # Always show all environment variables for debugging
        st.markdown("### All Environment Variables")
        st.write(f"**Total environment variables found: {len(env_vars)}**")
        
        env_df = pd.DataFrame([
            {"Variable": k, "Value": v} for k, v in sorted(env_vars.items())
        ])
        if not env_df.empty:
            st.dataframe(env_df, width='stretch', hide_index=True, height=400)
            
            # Also show as JSON for easier debugging
            with st.expander("View as JSON"):
                import json
                st.json(dict(sorted(env_vars.items())))
        else:
            st.warning("No environment variables found!")
        
        return
    
    # Load data
    try:
        with st.spinner("Carregando e analisando mensagens..."):
            df, unrecognized_names = load_and_parse_messages(chat_file_url)
    except ValueError as e:
        error_msg = str(e)
        st.error(f"Erro ao carregar dados: {error_msg}")
        st.info("Certifique-se de que a variável CHAT_FILE_URL está configurada no arquivo .env ou nas configurações de secrets do Streamlit")
        
        # Specific debugging for CHAT_FILE_URL
        st.markdown("### CHAT_FILE_URL Debugging")
        chat_file_url_value = os.getenv('CHAT_FILE_URL')
        chat_file_url_stripped = os.getenv('CHAT_FILE_URL', '').strip()
        
        # Check Streamlit secrets
        secrets_value = None
        secrets_stripped = ''
        try:
            if hasattr(st, 'secrets') and 'CHAT_FILE_URL' in st.secrets:
                secrets_value = st.secrets['CHAT_FILE_URL']
                secrets_stripped = str(secrets_value).strip() if secrets_value else ''
        except (AttributeError, KeyError, FileNotFoundError, TypeError):
            pass
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**os.getenv('CHAT_FILE_URL'):**")
            if chat_file_url_value is None:
                st.error(f"`None` (not found in .env)")
            else:
                st.success(f"`{repr(chat_file_url_value)}`")
        with col2:
            st.write("**st.secrets.get('CHAT_FILE_URL'):**")
            if secrets_value is None:
                st.error(f"`None` (not found in secrets)")
            else:
                st.success(f"`{repr(secrets_value)}`")
        with col3:
            st.write("**Final value (after fallback):**")
            if not chat_file_url_stripped and not secrets_stripped:
                st.error(f"`{repr('')}` (not found in either)")
            elif chat_file_url_stripped:
                st.success(f"`{repr(chat_file_url_stripped)}` (from .env)")
            else:
                st.success(f"`{repr(secrets_stripped)}` (from secrets)")
        
        # Check for related environment variables
        env_vars = {k: v for k, v in os.environ.items()}
        chat_related = {k: v for k, v in env_vars.items() if 'CHAT' in k.upper() or 'URL' in k.upper()}
        if chat_related:
            st.write("**Environment variables containing 'CHAT' or 'URL':**")
            chat_df = pd.DataFrame([
                {"Variable": k, "Value": v} for k, v in sorted(chat_related.items())
            ])
            st.dataframe(chat_df, width='stretch', hide_index=True)
        
        # Check if CHAT_FILE_URL exists in os.environ
        if 'CHAT_FILE_URL' in env_vars:
            # Exact case match found
            st.success(f"**Found in os.environ:** `CHAT_FILE_URL` = `{repr(env_vars['CHAT_FILE_URL'])}`")
            if chat_file_url_value is None:
                st.warning("💡 **Issue:** The variable exists in os.environ but `os.getenv('CHAT_FILE_URL')` returned `None`. This is likely a Streamlit caching issue.")
                st.info("**Solution:** Try clearing the Streamlit cache (Settings > Clear cache) or restarting the Streamlit server.")
            elif not chat_file_url_stripped:
                st.warning("💡 **Issue:** The variable exists but is empty or contains only whitespace after stripping.")
        else:
            # Check for case-insensitive match
            chat_file_url_variants = {k: v for k, v in env_vars.items() if k.upper() == 'CHAT_FILE_URL'}
            if chat_file_url_variants:
                variant_key = list(chat_file_url_variants.keys())[0]
                st.warning(f"**Found in os.environ with different case:** `{variant_key}` = `{repr(chat_file_url_variants[variant_key])}`")
                st.info("💡 **Tip:** Environment variable names are case-sensitive. Make sure your .env file uses exactly `CHAT_FILE_URL` (all uppercase).")
            else:
                st.error("**CHAT_FILE_URL not found in os.environ at all.**")
        
        st.markdown("---")
        
        # Always show all environment variables for debugging
        st.markdown("### All Environment Variables")
        st.write(f"**Total environment variables found: {len(env_vars)}**")
        
        env_df = pd.DataFrame([
            {"Variable": k, "Value": v} for k, v in sorted(env_vars.items())
        ])
        if not env_df.empty:
            st.dataframe(env_df, width='stretch', hide_index=True, height=400)
            
            # Also show as JSON for easier debugging
            with st.expander("View as JSON"):
                import json
                st.json(dict(sorted(env_vars.items())))
        else:
            st.warning("No environment variables found!")
        
        with st.expander("Debug - Error Details"):
            st.write("Detalhes do erro:")
            st.code(error_msg)
            st.write("**Note:** If CHAT_FILE_URL appears in the environment variables table above but os.getenv() returns None, this is likely a Streamlit caching issue. Try:")
            st.code("1. Clear Streamlit cache: Settings > Clear cache\n2. Restart the Streamlit server\n3. Check that .env file is in the correct location\n4. Alternatively, configure CHAT_FILE_URL in Streamlit secrets (.streamlit/secrets.toml for local or Streamlit Cloud secrets for remote)")
        return
    except Exception as e:
        error_msg = str(e)
        st.error(f"Erro ao baixar ou analisar dados: {error_msg}")
        st.info("Verifique se a URL no arquivo .env ou nas configurações de secrets do Streamlit está correta e acessível")
        
        # Show environment variables for any exception to help debug
        st.subheader("Environment Variables (Debug)")
        env_vars = {k: v for k, v in os.environ.items()}
        env_df = pd.DataFrame([
            {"Variable": k, "Value": v} for k, v in sorted(env_vars.items())
        ])
        st.dataframe(env_df, width='stretch', hide_index=True, height=400)
        
        st.exception(e)
        return
    
    # Display alert modal for unrecognized names
    if unrecognized_names:
        with st.container():
            st.error("⚠️ **Atenção: Mensagens com nomes não reconhecidos encontradas**")
            st.markdown(f"**{len(unrecognized_names)} nome(s) não reconhecido(s) encontrado(s):**")
            for name in sorted(unrecognized_names):
                st.markdown(f"- {name}")
            st.markdown("Essas mensagens foram excluídas da análise. Considere adicionar esses nomes ao mapeamento de nomes (NAME_MAP) no código.")
            st.markdown("---")
    
    # Check if DataFrame is empty
    if df.empty:
        st.error("Nenhum dado foi carregado. Por favor, verifique se o arquivo baixado da URL contém mensagens válidas.")
        st.info("Debug: O arquivo pode estar vazio ou a lógica de análise pode precisar de ajuste.")
        return
    
    # Get available years and persons (filter out NaN values)
    available_years = sorted([y for y in df['year'].unique().tolist() if pd.notna(y)], reverse=True) if not df.empty else []
    available_persons = sorted([p for p in df['parsed_name'].unique().tolist() if pd.notna(p)]) if not df.empty else []
    
    # Show debug info if no data
    if not available_years or not available_persons:
        st.error("Nenhum dado válido encontrado após a análise.")
        with st.expander("Debug"):
            st.write(f"Total de linhas após análise: {len(df)}")
            if len(df) > 0:
                st.write("Dados de exemplo:")
                st.dataframe(df.head())
            st.write("Qtd. de nomes brutos:")
            if 'raw_name' in df.columns:
                st.write(df['raw_name'].value_counts())
        return
    
    # Sidebar filters
    st.sidebar.header("Filtros")
    
    # Initialize session state for selections
    if 'selected_years_key' not in st.session_state:
        st.session_state.selected_years_key = available_years
    if 'selected_persons_key' not in st.session_state:
        st.session_state.selected_persons_key = available_persons
    
    # Filter session state to only include valid options (in case options changed)
    st.session_state.selected_years_key = [y for y in st.session_state.selected_years_key if y in available_years]
    st.session_state.selected_persons_key = [p for p in st.session_state.selected_persons_key if p in available_persons]
    
    # Years filter with select all buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Selecionar Todos", key="select_all_years_btn"):
            st.session_state.selected_years_key = available_years
            st.rerun()
    with col2:
        if st.button("Deselecionar Todos", key="deselect_all_years_btn"):
            st.session_state.selected_years_key = []
            st.rerun()
    
    selected_years = st.sidebar.multiselect(
        "Selecionar Anos",
        options=available_years,
        default=st.session_state.selected_years_key,
        help="Escolha quais anos incluir na análise",
        key="years_multiselect"
    )
    
    # Update session state from widget
    st.session_state.selected_years_key = selected_years
    
    # Persons filter with select all buttons
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("Selecionar Todos", key="select_all_persons_btn"):
            st.session_state.selected_persons_key = available_persons
            st.rerun()
    with col4:
        if st.button("Deselecionar Todos", key="deselect_all_persons_btn"):
            st.session_state.selected_persons_key = []
            st.rerun()
    
    selected_persons = st.sidebar.multiselect(
        "Selecionar Pessoas",
        options=available_persons,
        default=st.session_state.selected_persons_key,
        help="Escolha quais pessoas incluir na análise",
        key="persons_multiselect"
    )
    
    # Update session state from widget
    st.session_state.selected_persons_key = selected_persons
    
    # Add percentage toggle in sidebar (for both heatmaps and bar graphs)
    use_percentage = st.sidebar.checkbox("Usar porcentagem nos gráficos", value=True, key="graph_percentage")
    
    # Add heatmap normalization mode toggle
    heatmap_mode = st.sidebar.radio(
        "Modo de Normalização do Mapa de Calor",
        options=["Por Dia", "Por Semana"],
        index=1,  # Default to "Por Semana"
        help="Por Dia: cada célula é relativa ao seu dia da semana. Por Semana: cada célula é relativa à semana inteira.",
        key="heatmap_mode"
    )
    
    # Create tabs
    tab1, tab2 = st.tabs(["Overview", "Individual"])
    
    # Overview Tab
    with tab1:
        # Filter data
        if not selected_years or not selected_persons:
            st.warning("Por favor, selecione pelo menos um ano e uma pessoa para visualizar a análise.")
        else:
            filtered_df = filter_data(df, selected_years, selected_persons)
            
            # Overview Metrics
            st.header("Métricas Gerais")
            metrics = calculate_metrics(filtered_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Mensagens", f"{metrics['total_messages']:,}")
            with col2:
                st.metric("Participantes Únicos", metrics['unique_participants'])
            with col3:
                st.metric("Média de Mensagens por Pessoa", f"{metrics['avg_messages_per_person']:,.1f}")
            
            st.markdown("---")
            
            # Per-year message counts per person
            st.header("Mensagens por Pessoa por Ano")
    
            # Create grouped data
            year_person_counts = filtered_df.groupby(['year', 'parsed_name']).size().reset_index(name='count')
            
            # Stacked bar chart - create with original data order
            fig_stacked = px.bar(
                year_person_counts,
                x='year',
                y='count',
                color='parsed_name',
                title='Mensagens por Pessoa por Ano',
                labels={'year': 'Ano', 'count': 'Número de Mensagens', 'parsed_name': 'Pessoa'},
                barmode='stack'
            )
            # Set legendrank for alphabetical legend order without changing trace/visual order
            unique_names = sorted(year_person_counts['parsed_name'].unique())
            name_to_rank = {name: i for i, name in enumerate(unique_names)}
            for trace in fig_stacked.data:
                if trace.name in name_to_rank:
                    trace.legendrank = name_to_rank[trace.name]
            fig_stacked.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_stacked, width='stretch')
            
            # Total messages per year
            st.subheader("Total de Mensagens por Ano")
            year_totals = filtered_df.groupby('year').size().reset_index(name='count')
            
            # Calculate percentages
            total_all_years = year_totals['count'].sum()
            year_totals['percentage'] = (year_totals['count'] / total_all_years * 100) if total_all_years > 0 else 0
            
            # Use percentage or absolute value based on toggle
            if use_percentage:
                year_totals['display_value'] = year_totals['percentage']
                year_totals['text'] = year_totals['percentage'].apply(lambda x: f'{x:.1f}%')
                y_label = 'Porcentagem de Mensagens'
            else:
                year_totals['display_value'] = year_totals['count']
                year_totals['text'] = year_totals['count'].apply(lambda x: f'{x:,}')
                y_label = 'Número de Mensagens'
            
            fig_total = px.bar(
                year_totals,
                x='year',
                y='display_value',
                title='Total de Mensagens por Ano',
                labels={'year': 'Ano', 'display_value': y_label},
                color='display_value',
                color_continuous_scale='viridis',
                text='text'
            )
            fig_total.update_traces(textposition='inside', textfont_size=12)
            fig_total.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_total, width='stretch')
            
            st.markdown("---")
            
            # Messages by time period (month, trimester, quarter, semester)
            st.header("Mensagens por Período")
            
            # Dropdown for time period selection
            period_options = {
                'Mês': 'month',
                'Trimestre': 'trimester',
                'Trimestre (Q1-Q4)': 'quarter',
                'Semestre': 'semester'
            }
            selected_period_label = st.selectbox(
                "Selecionar Período",
                options=list(period_options.keys()),
                index=0,  # Default to 'Mês'
                help="Escolha o período de agrupamento das mensagens",
                key="time_period_select"
            )
            selected_period_type = period_options[selected_period_label]
            
            # Group messages by selected time period
            period_counts = group_by_time_period(filtered_df, selected_period_type)
            
            if not period_counts.empty:
                # Calculate percentages
                total_all_periods = period_counts['count'].sum()
                period_counts['percentage'] = (period_counts['count'] / total_all_periods * 100) if total_all_periods > 0 else 0
                
                # Use percentage or absolute value based on toggle
                if use_percentage:
                    period_counts['display_value'] = period_counts['percentage']
                    period_counts['text'] = period_counts['percentage'].apply(lambda x: f'{x:.1f}%')
                    y_label = 'Porcentagem de Mensagens'
                else:
                    period_counts['display_value'] = period_counts['count']
                    period_counts['text'] = period_counts['count'].apply(lambda x: f'{x:,}')
                    y_label = 'Número de Mensagens'
                
                # Create period label for title
                period_title_map = {
                    'month': 'Mês',
                    'trimester': 'Trimestre',
                    'quarter': 'Trimestre (Q1-Q4)',
                    'semester': 'Semestre'
                }
                period_title = period_title_map[selected_period_type]
                
                fig_period = px.bar(
                    period_counts,
                    x='period',
                    y='display_value',
                    title=f'Mensagens por {period_title}',
                    labels={'period': period_title, 'display_value': y_label},
                    color='display_value',
                    color_continuous_scale='viridis',
                    text='text'
                )
                fig_period.update_traces(textposition='inside', textfont_size=12)
                fig_period.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig_period, width='stretch')
            else:
                st.info("Não há dados disponíveis para o período selecionado.")
            
            st.markdown("---")
            
            # Top users per year
            st.header("Top 10 Usuários por Ano")
            
            for year in sorted(selected_years, reverse=True):
                year_data = filtered_df[filtered_df['year'] == year]
                if len(year_data) == 0:
                    continue
                
                year_counts = year_data['parsed_name'].value_counts().head(10).reset_index()
                year_counts.columns = ['Person', 'Count']
                
                # Calculate percentages
                total_year = year_counts['Count'].sum()
                year_counts['percentage'] = (year_counts['Count'] / total_year * 100) if total_year > 0 else 0
                
                # Use percentage or absolute value based on toggle
                if use_percentage:
                    year_counts['display_value'] = year_counts['percentage']
                    year_counts['text'] = year_counts['percentage'].apply(lambda x: f'{x:.1f}%')
                    x_label = 'Porcentagem de Mensagens'
                else:
                    year_counts['display_value'] = year_counts['Count']
                    year_counts['text'] = year_counts['Count'].apply(lambda x: f'{x:,}')
                    x_label = 'Número de Mensagens'
                
                st.subheader(year)
                fig_top = px.bar(
                    year_counts,
                    x='display_value',
                    y='Person',
                    orientation='h',
                    title=f'Top 10 Usuários Mais Ativos em {year}',
                    labels={'display_value': x_label, 'Person': 'Pessoa'},
                    color='display_value',
                    color_continuous_scale='viridis',
                    text='text'
                )
                fig_top.update_traces(textposition='inside', textfont_size=10)
                fig_top.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_top, width='stretch')
            
            st.markdown("---")
            
            # Most/Least active comparison per year
            st.header("Mais vs Menos Ativos por Ano")
            
            for year in sorted(selected_years, reverse=True):
                year_data = filtered_df[filtered_df['year'] == year]
                if len(year_data) == 0:
                    continue
                
                year_counts = year_data['parsed_name'].value_counts()
                if len(year_counts) == 0:
                    continue
                
                most_active = year_counts.idxmax()
                least_active = year_counts.idxmin()
                most_count = year_counts.max()
                least_count = year_counts.min()
                
                # Calculate percentages
                total_year = year_counts.sum()
                most_percentage = (most_count / total_year * 100) if total_year > 0 else 0
                least_percentage = (least_count / total_year * 100) if total_year > 0 else 0
                
                st.subheader(year)
                
                # Use percentage or absolute value based on toggle
                if use_percentage:
                    comparison_data = pd.DataFrame({
                        'Person': [most_active, least_active],
                        'Count': [most_count, least_count],
                        'DisplayValue': [most_percentage, least_percentage],
                        'Type': ['Mais Ativo', 'Menos Ativo']
                    })
                    comparison_data['text'] = comparison_data['DisplayValue'].apply(lambda x: f'{x:.1f}%')
                    y_label = 'Porcentagem de Mensagens'
                else:
                    comparison_data = pd.DataFrame({
                        'Person': [most_active, least_active],
                        'Count': [most_count, least_count],
                        'DisplayValue': [most_count, least_count],
                        'Type': ['Mais Ativo', 'Menos Ativo']
                    })
                    comparison_data['text'] = comparison_data['DisplayValue'].apply(lambda x: f'{x:,}')
                    y_label = 'Número de Mensagens'
                
                fig_compare = px.bar(
                    comparison_data,
                    x='Type',
                    y='DisplayValue',
                    color='Type',
                    text='text',
                    title=f'Usuários Mais e Menos Ativos em {year}',
                    labels={'DisplayValue': y_label, 'Type': 'Categoria'},
                    color_discrete_map={'Mais Ativo': '#571089', 'Menos Ativo': '#d55d92'}
                )
                fig_compare.update_traces(textposition='inside', textfont_size=12)
                fig_compare.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_compare, width='stretch')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Mais Ativo:** {most_active} ({most_count:,} mensagens)")
                with col2:
                    st.info(f"**Menos Ativo:** {least_active} ({least_count:,} mensagens)")
            
            st.markdown("---")
            
            # Interactive data table
            st.header("Qtd. de Mensagens por Pessoa por Ano")
            
            table_data = filtered_df.groupby(['year', 'parsed_name']).size().reset_index(name='count')
            table_data = table_data.sort_values(['year', 'count'], ascending=[False, False])
            table_data.columns = ['Ano', 'Pessoa', 'Contagem de Mensagens']
            
            st.dataframe(
                table_data,
                width='stretch',
                height=400,
                hide_index=True
            )
            
            # Download button
            # csv = table_data.to_csv(index=False)
            # st.download_button(
            #     label="Baixar dados como CSV",
            #     data=csv,
            #     file_name=f"whatsapp_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            #     mime="text/csv"
            # )
            
            st.markdown("---")
            
            # Additional analyses
            st.header("Análises Adicionais")
            
            # Messages per day of week
            st.subheader("Mensagens por Dia da Semana")
            filtered_df['day_of_week'] = filtered_df['date'].dt.dayofweek
            day_names = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
            filtered_df['day_name'] = filtered_df['day_of_week'].map(lambda x: day_names[x])
            
            day_counts = filtered_df['day_name'].value_counts().reindex(day_names).reset_index()
            day_counts.columns = ['Dia da Semana', 'Contagem']
            
            # Calculate percentages
            total_days = day_counts['Contagem'].sum()
            day_counts['percentage'] = (day_counts['Contagem'] / total_days * 100) if total_days > 0 else 0
            
            # Use percentage or absolute value based on toggle
            if use_percentage:
                day_counts['display_value'] = day_counts['percentage']
                day_counts['text'] = day_counts['percentage'].apply(lambda x: f'{x:.1f}%')
                y_label = 'Porcentagem de Mensagens'
            else:
                day_counts['display_value'] = day_counts['Contagem']
                day_counts['text'] = day_counts['Contagem'].apply(lambda x: f'{x:,}')
                y_label = 'Número de Mensagens'
            
            fig_day = px.bar(
                day_counts,
                x='Dia da Semana',
                y='display_value',
                title='Mensagens por Dia da Semana',
                labels={'display_value': y_label},
                color='display_value',
                color_continuous_scale='viridis',
                text='text'
            )
            fig_day.update_traces(textposition='inside', textfont_size=12)
            fig_day.update_layout(height=400)
            st.plotly_chart(fig_day, width='stretch')
            
            # Weekday x Hour heatmap
            st.subheader("Mapa de Calor: Dia da Semana x Hora do Dia")
            filtered_df['hour'] = filtered_df['date'].dt.hour
            
            # Create heatmap data
            heatmap_data = filtered_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
            
            # Create pivot table for heatmap
            heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)
            
            # Reindex to ensure correct weekday order
            heatmap_pivot = heatmap_pivot.reindex(day_names)
            
            # Ensure all 24 hours (0-23) are in columns, fill missing with 0
            all_hours = list(range(24))
            for hour in all_hours:
                if hour not in heatmap_pivot.columns:
                    heatmap_pivot[hour] = 0
            heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]
            
            # Convert to percentage if toggle is enabled
            if use_percentage:
                if heatmap_mode == "Por Semana":
                    heatmap_pivot = convert_to_percentage_per_week(heatmap_pivot)
                else:
                    heatmap_pivot = convert_to_percentage(heatmap_pivot)
                colorbar_title = "Porcentagem de Mensagens"
                text_template = '%{text:.1f}%'
            else:
                colorbar_title = "Quantidade de Mensagens"
                text_template = '%{text:.0f}'
            
            # Create heatmap using plotly
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='Viridis',
                text=heatmap_pivot.values,
                texttemplate=text_template,
                textfont={"size": 10},
                colorbar=dict(title=colorbar_title)
            ))
            
            fig_heatmap.update_layout(
                title='Mapa de Calor: Mensagens por Dia da Semana e Hora do Dia',
                xaxis_title='Hora do Dia',
                yaxis_title='Dia da Semana',
                height=500,
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    tickvals=list(range(24)),
                    ticktext=[str(h) for h in range(24)]
                )
            )
            
            st.plotly_chart(fig_heatmap, width='stretch')
            
            # Word cloud
            st.subheader("Nuvem de Palavras")
            wordcloud = generate_word_cloud(filtered_df)
            if wordcloud:
                fig_wc, ax = plt.subplots(figsize=(16, 8), dpi=100, facecolor='white')
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc)
                plt.close(fig_wc)
            else:
                st.info("Não há palavras suficientes para gerar a nuvem de palavras.")
            
            # Most common word per hour/day
            st.subheader("Palavra Mais Comum por Hora e Dia da Semana")
            
            common_words_df = get_most_common_word_per_hour_day(filtered_df)
            
            if not common_words_df.empty:
                # Ensure all 24 hours (0-23) are included
                all_hours = list(range(24))
                
                # Create pivot table for display
                pivot_table = common_words_df.pivot(index='day_name', columns='hour', values='most_common_word')
                pivot_table = pivot_table.reindex(day_names)
                # Ensure all hours are in columns, fill missing with '-'
                for hour in all_hours:
                    if hour not in pivot_table.columns:
                        pivot_table[hour] = '-'
                pivot_table = pivot_table[sorted(pivot_table.columns)]
                pivot_table = pivot_table.fillna('-')
                
                # Display as table
                st.write("**Tabela: Palavra Mais Comum**")
                display_table = pivot_table.copy()
                display_table.index.name = 'Dia da Semana'
                display_table.columns.name = 'Hora'
                st.dataframe(display_table, width='stretch', height=300)
                
                # Create heatmap with word labels
                st.write("**Mapa de Calor: Palavra Mais Comum**")
                
                # Create a numeric version for the heatmap (using word counts)
                count_pivot = common_words_df.pivot(index='day_name', columns='hour', values='count')
                count_pivot = count_pivot.reindex(day_names)
                # Ensure all hours are in columns, fill missing with 0
                for hour in all_hours:
                    if hour not in count_pivot.columns:
                        count_pivot[hour] = 0
                count_pivot = count_pivot[sorted(count_pivot.columns)]
                count_pivot = count_pivot.fillna(0)
                
                # Convert to percentage if toggle is enabled (reuse the same toggle)
                if use_percentage:
                    if heatmap_mode == "Por Semana":
                        count_pivot = convert_to_percentage_per_week(count_pivot)
                    else:
                        count_pivot = convert_to_percentage(count_pivot)
                    colorbar_title = "Porcentagem da Frequência da Palavra"
                else:
                    colorbar_title = "Frequência da Palavra"
                
                # Create custom text for each cell - align with pivot table structure
                text_matrix = []
                for day in day_names:
                    row_text = []
                    for hour in sorted(count_pivot.columns):
                        if day in pivot_table.index and hour in pivot_table.columns:
                            word = pivot_table.loc[day, hour]
                            if pd.notna(word) and word != '-':
                                row_text.append(str(word))
                            else:
                                row_text.append("")
                        else:
                            row_text.append("")
                    text_matrix.append(row_text)
                
                # Create hover template based on mode
                if use_percentage:
                    hover_template = 'Dia: %{y}<br>Hora: %{x}<br>Palavra: %{text}<br>Porcentagem: %{z:.1f}%<extra></extra>'
                else:
                    hover_template = 'Dia: %{y}<br>Hora: %{x}<br>Palavra: %{text}<br>Frequência: %{z}<extra></extra>'
                
                fig_word_heatmap = go.Figure(data=go.Heatmap(
                    z=count_pivot.values,
                    x=count_pivot.columns,
                    y=count_pivot.index,
                    colorscale='Viridis',
                    text=text_matrix,
                    texttemplate='%{text}',
                    textfont={"size": 9},
                    colorbar=dict(title=colorbar_title),
                    hovertemplate=hover_template
                ))
                
                fig_word_heatmap.update_layout(
                    title='Palavra Mais Comum por Dia da Semana e Hora do Dia',
                    xaxis_title='Hora do Dia',
                    yaxis_title='Dia da Semana',
                    height=500,
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=1,
                        tickvals=list(range(24)),
                        ticktext=[str(h) for h in range(24)]
                    )
                )
                
                st.plotly_chart(fig_word_heatmap, width='stretch')
            else:
                st.info("Não há dados suficientes para analisar palavras por hora e dia.")
            
            # Total messages per person (pie chart)
            st.subheader("Total de Mensagens por Pessoa")
            person_totals = filtered_df['parsed_name'].value_counts().reset_index()
            person_totals.columns = ['Pessoa', 'Contagem']
            
            fig_pie = px.pie(
                person_totals,
                values='Contagem',
                names='Pessoa',
                title='Distribuição de Mensagens por Pessoa',
                hole=0.4
            )
            # Set legendrank for alphabetical legend order without changing trace/visual order
            unique_names = sorted(person_totals['Pessoa'].unique())
            name_to_rank = {name: i for i, name in enumerate(unique_names)}
            for trace in fig_pie.data:
                if trace.name in name_to_rank:
                    trace.legendrank = name_to_rank[trace.name]
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=600)
            st.plotly_chart(fig_pie, width='stretch')
            
            # Message trends over time
            st.subheader("Tendências de Mensagens ao Longo do Tempo")
            filtered_df['date_only'] = filtered_df['date'].dt.date
            daily_counts = filtered_df.groupby('date_only').size().reset_index(name='count')
            daily_counts.columns = ['Data', 'Contagem']
            daily_counts = daily_counts.sort_values('Data', ascending=False)
            
            fig_trend = px.line(
                daily_counts,
                x='Data',
                y='Contagem',
                title='Tendência de Qtd. de Mensagens ao Longo do Tempo',
                labels={'Contagem': 'Número de Mensagens', 'Data': 'Data'},
                markers=True
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, width='stretch')
    
    # Individual Tab
    with tab2:
        if not selected_years:
            st.warning("Por favor, selecione pelo menos um ano na barra lateral para visualizar a análise individual.")
        else:
            # Filter available persons based on selected years
            year_filtered_df = df[df['year'].isin(selected_years)]
            available_persons_filtered = sorted([p for p in year_filtered_df['parsed_name'].unique().tolist() if pd.notna(p)])
            
            if not available_persons_filtered:
                st.warning("Nenhuma pessoa encontrada para os anos selecionados.")
            else:
                # User selector
                selected_user = st.selectbox(
                    "Selecionar Pessoa",
                    options=available_persons_filtered,
                    help="Escolha uma pessoa para visualizar sua análise individual",
                    key="individual_user_select"
                )
                
                if selected_user:
                    # Filter data to single user
                    individual_df = filter_data(df, selected_years, [selected_user])
                    
                    if individual_df.empty:
                        st.info(f"Nenhuma mensagem encontrada para {selected_user} nos anos selecionados.")
                    else:
                        # Individual Metrics
                        st.header("Métricas Individuais")
                        total_messages = len(individual_df)
                        messages_per_year = individual_df.groupby('year').size()
                        avg_per_year = messages_per_year.mean() if len(messages_per_year) > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Mensagens", f"{total_messages:,}")
                        with col2:
                            st.metric("Anos com Mensagens", len(messages_per_year))
                        with col3:
                            st.metric("Média por Ano", f"{avg_per_year:,.1f}")
                        
                        st.markdown("---")
                        
                        # Messages per Year
                        st.header("Mensagens por Ano")
                        year_totals = individual_df.groupby('year').size().reset_index(name='count')
                        
                        # Calculate percentages
                        total_all_years = year_totals['count'].sum()
                        year_totals['percentage'] = (year_totals['count'] / total_all_years * 100) if total_all_years > 0 else 0
                        
                        # Use percentage or absolute value based on toggle
                        if use_percentage:
                            year_totals['display_value'] = year_totals['percentage']
                            year_totals['text'] = year_totals['percentage'].apply(lambda x: f'{x:.1f}%')
                            y_label = 'Porcentagem de Mensagens'
                        else:
                            year_totals['display_value'] = year_totals['count']
                            year_totals['text'] = year_totals['count'].apply(lambda x: f'{x:,}')
                            y_label = 'Número de Mensagens'
                        
                        fig_year = px.bar(
                            year_totals,
                            x='year',
                            y='display_value',
                            title=f'Mensagens por Ano - {selected_user}',
                            labels={'year': 'Ano', 'display_value': y_label},
                            color='display_value',
                            color_continuous_scale='viridis',
                            text='text'
                        )
                        fig_year.update_traces(textposition='inside', textfont_size=12)
                        fig_year.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_year, width='stretch')
                        
                        st.markdown("---")
                        
                        # Additional analyses
                        st.header("Análises Adicionais")
                        
                        # Messages per day of week
                        st.subheader("Mensagens por Dia da Semana")
                        individual_df['day_of_week'] = individual_df['date'].dt.dayofweek
                        day_names = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
                        individual_df['day_name'] = individual_df['day_of_week'].map(lambda x: day_names[x])
                        
                        day_counts = individual_df['day_name'].value_counts().reindex(day_names).reset_index()
                        day_counts.columns = ['Dia da Semana', 'Contagem']
                        
                        # Calculate percentages
                        total_days = day_counts['Contagem'].sum()
                        day_counts['percentage'] = (day_counts['Contagem'] / total_days * 100) if total_days > 0 else 0
                        
                        # Use percentage or absolute value based on toggle
                        if use_percentage:
                            day_counts['display_value'] = day_counts['percentage']
                            day_counts['text'] = day_counts['percentage'].apply(lambda x: f'{x:.1f}%')
                            y_label = 'Porcentagem de Mensagens'
                        else:
                            day_counts['display_value'] = day_counts['Contagem']
                            day_counts['text'] = day_counts['Contagem'].apply(lambda x: f'{x:,}')
                            y_label = 'Número de Mensagens'
                        
                        fig_day = px.bar(
                            day_counts,
                            x='Dia da Semana',
                            y='display_value',
                            title=f'Mensagens por Dia da Semana - {selected_user}',
                            labels={'display_value': y_label},
                            color='display_value',
                            color_continuous_scale='viridis',
                            text='text'
                        )
                        fig_day.update_traces(textposition='inside', textfont_size=12)
                        fig_day.update_layout(height=400)
                        st.plotly_chart(fig_day, width='stretch')
                        
                        # Weekday x Hour heatmap
                        st.subheader("Mapa de Calor: Dia da Semana x Hora do Dia")
                        individual_df['hour'] = individual_df['date'].dt.hour
                        
                        # Create heatmap data
                        heatmap_data = individual_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
                        
                        # Create pivot table for heatmap
                        heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)
                        
                        # Reindex to ensure correct weekday order
                        heatmap_pivot = heatmap_pivot.reindex(day_names)
                        
                        # Ensure all 24 hours (0-23) are in columns, fill missing with 0
                        all_hours = list(range(24))
                        for hour in all_hours:
                            if hour not in heatmap_pivot.columns:
                                heatmap_pivot[hour] = 0
                        heatmap_pivot = heatmap_pivot[sorted(heatmap_pivot.columns)]
                        
                        # Convert to percentage if toggle is enabled
                        if use_percentage:
                            if heatmap_mode == "Por Semana":
                                heatmap_pivot = convert_to_percentage_per_week(heatmap_pivot)
                            else:
                                heatmap_pivot = convert_to_percentage(heatmap_pivot)
                            colorbar_title = "Porcentagem de Mensagens"
                            text_template = '%{text:.1f}%'
                        else:
                            colorbar_title = "Quantidade de Mensagens"
                            text_template = '%{text:.0f}'
                        
                        # Create heatmap using plotly
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_pivot.values,
                            x=heatmap_pivot.columns,
                            y=heatmap_pivot.index,
                            colorscale='Viridis',
                            text=heatmap_pivot.values,
                            texttemplate=text_template,
                            textfont={"size": 10},
                            colorbar=dict(title=colorbar_title)
                        ))
                        
                        fig_heatmap.update_layout(
                            title=f'Mapa de Calor: Mensagens por Dia da Semana e Hora do Dia - {selected_user}',
                            xaxis_title='Hora do Dia',
                            yaxis_title='Dia da Semana',
                            height=500,
                            xaxis=dict(
                                tickmode='linear',
                                tick0=0,
                                dtick=1,
                                tickvals=list(range(24)),
                                ticktext=[str(h) for h in range(24)]
                            )
                        )
                        
                        st.plotly_chart(fig_heatmap, width='stretch')
                        
                        # Word cloud
                        st.subheader("Nuvem de Palavras")
                        wordcloud = generate_word_cloud(individual_df)
                        if wordcloud:
                            fig_wc, ax = plt.subplots(figsize=(16, 8), dpi=100, facecolor='white')
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            plt.tight_layout(pad=0)
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        else:
                            st.info("Não há palavras suficientes para gerar a nuvem de palavras.")
                        
                        # Most common word per hour/day
                        st.subheader("Palavra Mais Comum por Hora e Dia da Semana")
                        
                        common_words_df = get_most_common_word_per_hour_day(individual_df)
                        
                        if not common_words_df.empty:
                            # Ensure all 24 hours (0-23) are included
                            all_hours = list(range(24))
                            
                            # Create pivot table for display
                            pivot_table = common_words_df.pivot(index='day_name', columns='hour', values='most_common_word')
                            pivot_table = pivot_table.reindex(day_names)
                            # Ensure all hours are in columns, fill missing with '-'
                            for hour in all_hours:
                                if hour not in pivot_table.columns:
                                    pivot_table[hour] = '-'
                            pivot_table = pivot_table[sorted(pivot_table.columns)]
                            pivot_table = pivot_table.fillna('-')
                            
                            # Display as table
                            st.write("**Tabela: Palavra Mais Comum**")
                            display_table = pivot_table.copy()
                            display_table.index.name = 'Dia da Semana'
                            display_table.columns.name = 'Hora'
                            st.dataframe(display_table, width='stretch', height=300)
                            
                            # Create heatmap with word labels
                            st.write("**Mapa de Calor: Palavra Mais Comum**")
                            
                            # Create a numeric version for the heatmap (using word counts)
                            count_pivot = common_words_df.pivot(index='day_name', columns='hour', values='count')
                            count_pivot = count_pivot.reindex(day_names)
                            # Ensure all hours are in columns, fill missing with 0
                            for hour in all_hours:
                                if hour not in count_pivot.columns:
                                    count_pivot[hour] = 0
                            count_pivot = count_pivot[sorted(count_pivot.columns)]
                            count_pivot = count_pivot.fillna(0)
                            
                            # Convert to percentage if toggle is enabled (reuse the same toggle)
                            if use_percentage:
                                if heatmap_mode == "Por Semana":
                                    count_pivot = convert_to_percentage_per_week(count_pivot)
                                else:
                                    count_pivot = convert_to_percentage(count_pivot)
                                colorbar_title = "Porcentagem da Frequência da Palavra"
                            else:
                                colorbar_title = "Frequência da Palavra"
                            
                            # Create custom text for each cell - align with pivot table structure
                            text_matrix = []
                            for day in day_names:
                                row_text = []
                                for hour in sorted(count_pivot.columns):
                                    if day in pivot_table.index and hour in pivot_table.columns:
                                        word = pivot_table.loc[day, hour]
                                        if pd.notna(word) and word != '-':
                                            row_text.append(str(word))
                                        else:
                                            row_text.append("")
                                    else:
                                        row_text.append("")
                                text_matrix.append(row_text)
                            
                            # Create hover template based on mode
                            if use_percentage:
                                hover_template = 'Dia: %{y}<br>Hora: %{x}<br>Palavra: %{text}<br>Porcentagem: %{z:.1f}%<extra></extra>'
                            else:
                                hover_template = 'Dia: %{y}<br>Hora: %{x}<br>Palavra: %{text}<br>Frequência: %{z}<extra></extra>'
                            
                            fig_word_heatmap = go.Figure(data=go.Heatmap(
                                z=count_pivot.values,
                                x=count_pivot.columns,
                                y=count_pivot.index,
                                colorscale='Viridis',
                                text=text_matrix,
                                texttemplate='%{text}',
                                textfont={"size": 9},
                                colorbar=dict(title=colorbar_title),
                                hovertemplate=hover_template
                            ))
                            
                            fig_word_heatmap.update_layout(
                                title=f'Palavra Mais Comum por Dia da Semana e Hora do Dia - {selected_user}',
                                xaxis_title='Hora do Dia',
                                yaxis_title='Dia da Semana',
                                height=500,
                                xaxis=dict(
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=1,
                                    tickvals=list(range(24)),
                                    ticktext=[str(h) for h in range(24)]
                                )
                            )
                            
                            st.plotly_chart(fig_word_heatmap, width='stretch')
                        else:
                            st.info("Não há dados suficientes para analisar palavras por hora e dia.")
                        
                        # Message trends over time
                        st.subheader("Tendências de Mensagens ao Longo do Tempo")
                        individual_df['date_only'] = individual_df['date'].dt.date
                        daily_counts = individual_df.groupby('date_only').size().reset_index(name='count')
                        daily_counts.columns = ['Data', 'Contagem']
                        daily_counts = daily_counts.sort_values('Data', ascending=False)
                        
                        fig_trend = px.line(
                            daily_counts,
                            x='Data',
                            y='Contagem',
                            title=f'Tendência de Qtd. de Mensagens ao Longo do Tempo - {selected_user}',
                            labels={'Contagem': 'Número de Mensagens', 'Data': 'Data'},
                            markers=True
                        )
                        fig_trend.update_layout(height=400)
                        st.plotly_chart(fig_trend, width='stretch')


if __name__ == "__main__":
    main()

