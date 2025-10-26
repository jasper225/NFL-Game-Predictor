import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide"
)


class MinimalNFLCollector:
    """Lightweight NFL data collector using ESPN public API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_espn_scoreboard(self, season=2024, week=1):
        """Get game scores from ESPN API"""
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {'dates': season, 'seasontype': 2, 'week': week}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) != 2:
                    continue
                
                home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                
                if not home_team or not away_team:
                    continue
                
                game = {
                    'season': season,
                    'week': week,
                    'home_team': home_team.get('team', {}).get('abbreviation'),
                    'home_score': int(home_team.get('score', 0)),
                    'away_team': away_team.get('team', {}).get('abbreviation'),
                    'away_score': int(away_team.get('score', 0)),
                    'status': event.get('status', {}).get('type', {}).get('name'),
                }
                
                if game['status'] == 'STATUS_FINAL':
                    game['home_wins'] = 1 if game['home_score'] > game['away_score'] else 0
                    games.append(game)
            
            return pd.DataFrame(games)
            
        except Exception as e:
            logger.error(f"Error fetching scoreboard: {e}")
            return pd.DataFrame()
    
    def get_season_schedule(self, season=2024, weeks=None):
        """Get full season schedule"""
        if weeks is None:
            weeks = range(1, 19)
        
        all_games = []
        for week in weeks:
            games = self.get_espn_scoreboard(season, week)
            if not games.empty:
                all_games.append(games)
            time.sleep(0.3)
        
        if all_games:
            return pd.concat(all_games, ignore_index=True)
        return pd.DataFrame()
    
    def calculate_team_stats(self, schedule_df):
        """Calculate team statistics from schedule data"""
        if schedule_df.empty:
            return pd.DataFrame()
        
        teams = []
        team_abbrs = pd.concat([schedule_df['home_team'], schedule_df['away_team']]).unique()
        
        for team in team_abbrs:
            home_games = schedule_df[schedule_df['home_team'] == team]
            away_games = schedule_df[schedule_df['away_team'] == team]
            
            home_wins = home_games['home_wins'].sum()
            away_wins = len(away_games) - away_games['home_wins'].sum()
            total_wins = home_wins + away_wins
            total_games = len(home_games) + len(away_games)
            
            points_for = home_games['home_score'].sum() + away_games['away_score'].sum()
            points_against = home_games['away_score'].sum() + away_games['home_score'].sum()
            
            teams.append({
                'team': team,
                'games_played': total_games,
                'wins': total_wins,
                'losses': total_games - total_wins,
                'win_rate': total_wins / total_games if total_games > 0 else 0,
                'points_per_game': points_for / total_games if total_games > 0 else 0,
                'points_allowed_per_game': points_against / total_games if total_games > 0 else 0,
                'point_diff': points_for - points_against,
                'avg_margin': (points_for - points_against) / total_games if total_games > 0 else 0
            })
        
        return pd.DataFrame(teams)


class NFLPredictor:
    """NFL game prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.collector = MinimalNFLCollector()
    
    def collect_and_prepare_data(self, season=2024, weeks=None):
        """Collect real NFL data and prepare for training"""
        if weeks is None:
            # Get all completed weeks (adjust based on current week)
            weeks = range(1, 11)  # Adjust this as season progresses
        
        schedule = self.collector.get_season_schedule(season, weeks)
        team_stats = self.collector.calculate_team_stats(schedule)
        
        if schedule.empty or team_stats.empty:
            return pd.DataFrame()
        
        # Merge team stats with games
        training = schedule.merge(
            team_stats.add_suffix('_home'),
            left_on='home_team',
            right_on='team_home',
            how='left'
        )
        
        training = training.merge(
            team_stats.add_suffix('_away'),
            left_on='away_team',
            right_on='team_away',
            how='left'
        )
        
        # Create features
        training['win_rate_diff'] = training['win_rate_home'] - training['win_rate_away']
        training['ppg_diff'] = training['points_per_game_home'] - training['points_per_game_away']
        training['papg_diff'] = training['points_allowed_per_game_away'] - training['points_allowed_per_game_home']
        training['margin_diff'] = training['avg_margin_home'] - training['avg_margin_away']
        
        return training.dropna()
    
    def train_model(self, df):
        """Train prediction model"""
        feature_cols = [
            'win_rate_home', 'win_rate_away', 'win_rate_diff',
            'points_per_game_home', 'points_per_game_away', 'ppg_diff',
            'points_allowed_per_game_home', 'points_allowed_per_game_away', 'papg_diff',
            'avg_margin_home', 'avg_margin_away', 'margin_diff'
        ]
        
        X = df[feature_cols]
        y = df['home_wins']
        
        self.feature_columns = feature_cols
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy, feature_importance
    
    def predict_game(self, home_stats, away_stats):
        """Predict game outcome"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create feature vector
        features = {
            'win_rate_home': home_stats['win_rate'],
            'win_rate_away': away_stats['win_rate'],
            'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
            'points_per_game_home': home_stats['ppg'],
            'points_per_game_away': away_stats['ppg'],
            'ppg_diff': home_stats['ppg'] - away_stats['ppg'],
            'points_allowed_per_game_home': home_stats['papg'],
            'points_allowed_per_game_away': away_stats['papg'],
            'papg_diff': away_stats['papg'] - home_stats['papg'],
            'avg_margin_home': home_stats['margin'],
            'avg_margin_away': away_stats['margin'],
            'margin_diff': home_stats['margin'] - away_stats['margin']
        }
        
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'predicted_winner': 'Home' if prediction == 1 else 'Away',
            'home_win_probability': probability[1],
            'away_win_probability': probability[0],
            'confidence': max(probability)
        }


# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_trained = False
    st.session_state.accuracy = None
    st.session_state.team_stats = None

# App title
st.title("🏈 NFL Game Predictor")
st.markdown("**Predict NFL game outcomes using real data and machine learning**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    season = st.selectbox("Season", [2024, 2023], index=0)
    max_week = st.slider("Weeks to Include", 1, 18, 10)
    
    if st.button("🔄 Fetch Data & Train Model", type="primary", use_container_width=True):
        with st.spinner("Fetching real NFL data from ESPN..."):
            predictor = NFLPredictor()
            
            # Collect data
            df = predictor.collect_and_prepare_data(season=season, weeks=range(1, max_week + 1))
            
            if df.empty:
                st.error("❌ Could not fetch data. Check your internet connection.")
            else:
                st.info(f"✓ Collected {len(df)} games")
                
                # Train model
                accuracy, feature_importance = predictor.train_model(df)
                
                # Get team stats for predictions
                team_stats = predictor.collector.calculate_team_stats(df)
                
                st.session_state.predictor = predictor
                st.session_state.model_trained = True
                st.session_state.accuracy = accuracy
                st.session_state.team_stats = team_stats
                st.session_state.feature_importance = feature_importance
                
                st.success(f"✓ Model trained! Accuracy: {accuracy:.1%}")
    
    if st.session_state.model_trained:
        st.metric("Model Accuracy", f"{st.session_state.accuracy:.1%}")
        st.info("✅ Ready for predictions")
    else:
        st.warning("⚠️ Click above to fetch data and train model")

# Main content
if st.session_state.model_trained:
    tab1, tab2 = st.tabs(["🎯 Make Prediction", "📊 Model Insights"])
    
    with tab1:
        st.header("Predict Game Outcome")
        
        team_stats = st.session_state.team_stats
        teams = sorted(team_stats['team'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.selectbox("Select Home Team", teams, key="home")
            
            if home_team:
                home_data = team_stats[team_stats['team'] == home_team].iloc[0]
                st.metric("Win Rate", f"{home_data['win_rate']:.1%}")
                st.metric("Points Per Game", f"{home_data['points_per_game']:.1f}")
                st.metric("Points Allowed", f"{home_data['points_allowed_per_game']:.1f}")
        
        with col2:
            st.subheader("✈️ Away Team")
            away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team], key="away")
            
            if away_team:
                away_data = team_stats[team_stats['team'] == away_team].iloc[0]
                st.metric("Win Rate", f"{away_data['win_rate']:.1%}")
                st.metric("Points Per Game", f"{away_data['points_per_game']:.1f}")
                st.metric("Points Allowed", f"{away_data['points_allowed_per_game']:.1f}")
        
        if st.button("🔮 Predict Game", type="primary", use_container_width=True):
            home_stats = {
                'win_rate': home_data['win_rate'],
                'ppg': home_data['points_per_game'],
                'papg': home_data['points_allowed_per_game'],
                'margin': home_data['avg_margin']
            }
            
            away_stats = {
                'win_rate': away_data['win_rate'],
                'ppg': away_data['points_per_game'],
                'papg': away_data['points_allowed_per_game'],
                'margin': away_data['avg_margin']
            }
            
            prediction = st.session_state.predictor.predict_game(home_stats, away_stats)
            
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            winner_team = home_team if prediction['predicted_winner'] == 'Home' else away_team
            
            with col1:
                st.metric("Predicted Winner", winner_team)
            
            with col2:
                st.metric("Win Probability", f"{prediction['confidence']:.1%}")
            
            with col3:
                confidence_label = "High" if prediction['confidence'] > 0.7 else "Medium" if prediction['confidence'] > 0.55 else "Low"
                st.metric("Confidence", confidence_label)
            
            # Probability chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[home_team, away_team],
                    y=[prediction['home_win_probability'], prediction['away_win_probability']],
                    marker_color=['#013369', '#D50A0A'],
                    text=[f"{prediction['home_win_probability']:.1%}", 
                          f"{prediction['away_win_probability']:.1%}"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Win Probability",
                yaxis_title="Probability",
                yaxis=dict(tickformat='.0%'),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Performance")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Feature Importance")
            fig = px.bar(
                st.session_state.feature_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Team Statistics")
            st.dataframe(
                st.session_state.team_stats.sort_values('win_rate', ascending=False),
                use_container_width=True,
                height=400
            )

else:
    st.info("👈 Click 'Fetch Data & Train Model' in the sidebar to get started!")
    
    st.markdown("""
    ### How It Works
    
    1. **Fetch Real Data**: Pulls actual NFL game results from ESPN
    2. **Calculate Stats**: Computes team performance metrics
    3. **Train Model**: Builds ML model from historical games
    4. **Make Predictions**: Predicts outcomes for any matchup
    
    ### Features
    - ✅ Real NFL data (no fake samples!)
    - ✅ Current season statistics
    - ✅ Machine learning predictions
    - ✅ Probability estimates
    - ✅ No heavy dependencies needed
    
    ### Requirements
    ```
    pip install streamlit pandas scikit-learn plotly requests
    ```
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from ESPN API")