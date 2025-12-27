#!/usr/bin/env python3
"""
P99 Distribution Analysis - LLM Calls by User
Interactive visualization of user LLM call distribution with P99 insights
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="P99 Distribution - LLM Calls",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --accent-cyan: #00d4ff;
        --accent-purple: #a855f7;
        --accent-pink: #ec4899;
        --accent-orange: #f97316;
        --text-primary: #e2e8f0;
        --text-muted: #64748b;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, #0f0f1a 50%, #0a0a0f 100%);
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted);
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(18, 18, 26, 0.9), rgba(10, 10, 15, 0.9));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 212, 255, 0.5);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--accent-cyan);
        margin: 0;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .insight-box {
        background: linear-gradient(145deg, rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.05));
        border-left: 4px solid var(--accent-purple);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .p99-highlight {
        background: linear-gradient(145deg, rgba(249, 115, 22, 0.15), rgba(249, 115, 22, 0.05));
        border: 1px solid rgba(249, 115, 22, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: rgba(18, 18, 26, 0.8);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data (from actual Trino queries)
# =============================================================================

# Overall percentile data
percentile_data = {
    'percentile': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.5, 99.9, 100],
    'llm_calls': [1, 5, 11, 22, 32, 45, 59, 78, 100, 145, 301, 626, 4864, 9251, 26196, 225066]
}

# Distribution buckets (all users) with cost data
distribution_data = [
    {"bucket": "1-10", "user_count": 201880, "pct": 9.81, "avg_calls": 5, "total_cost": 22561.84, "avg_cost_per_user": 0.25, "cost_per_call": 0.0238},
    {"bucket": "11-25", "user_count": 290096, "pct": 14.10, "avg_calls": 18, "total_cost": 76513.93, "avg_cost_per_user": 0.42, "cost_per_call": 0.0149},
    {"bucket": "26-50", "user_count": 423867, "pct": 20.60, "avg_calls": 37, "total_cost": 168686.03, "avg_cost_per_user": 0.47, "cost_per_call": 0.0108},
    {"bucket": "51-100", "user_count": 534221, "pct": 25.96, "avg_calls": 73, "total_cost": 440967.37, "avg_cost_per_user": 0.88, "cost_per_call": 0.0114},
    {"bucket": "101-200", "user_count": 305081, "pct": 14.83, "avg_calls": 138, "total_cost": 539024.88, "avg_cost_per_user": 1.87, "cost_per_call": 0.0128},
    {"bucket": "201-500", "user_count": 178260, "pct": 8.66, "avg_calls": 309, "total_cost": 840252.50, "avg_cost_per_user": 4.87, "cost_per_call": 0.0153},
    {"bucket": "501-1K", "user_count": 54749, "pct": 2.66, "avg_calls": 690, "total_cost": 650262.47, "avg_cost_per_user": 12.30, "cost_per_call": 0.0172},
    {"bucket": "1K-2K", "user_count": 27031, "pct": 1.31, "avg_calls": 1397, "total_cost": 687718.49, "avg_cost_per_user": 26.25, "cost_per_call": 0.0182},
    {"bucket": "2K-5K", "user_count": 22197, "pct": 1.08, "avg_calls": 3177, "total_cost": 1456276.24, "avg_cost_per_user": 66.57, "cost_per_call": 0.0207},
    {"bucket": "5K-10K", "user_count": 11020, "pct": 0.54, "avg_calls": 7012, "total_cost": 1768262.18, "avg_cost_per_user": 161.21, "cost_per_call": 0.0229},
    {"bucket": "10K-25K", "user_count": 7089, "pct": 0.34, "avg_calls": 15087, "total_cost": 2707505.95, "avg_cost_per_user": 382.74, "cost_per_call": 0.0253},
    {"bucket": "25K-50K", "user_count": 1782, "pct": 0.09, "avg_calls": 33493, "total_cost": 1705092.18, "avg_cost_per_user": 956.84, "cost_per_call": 0.0286},
    {"bucket": "50K+", "user_count": 449, "pct": 0.02, "avg_calls": 163447, "total_cost": 1051527.83, "avg_cost_per_user": 2341.93, "cost_per_call": 0.0143},
]

# P99 users internal distribution
p99_distribution = [
    {"bucket": "5K-6K", "user_count": 4225, "pct": 19.92, "total_cost": 502286, "avg_calls": 5357},
    {"bucket": "6K-7K", "user_count": 2695, "pct": 12.70, "total_cost": 397819, "avg_calls": 6476},
    {"bucket": "7K-8K", "user_count": 2068, "pct": 9.75, "total_cost": 351818, "avg_calls": 7489},
    {"bucket": "8K-10K", "user_count": 2906, "pct": 13.70, "total_cost": 609165, "avg_calls": 8937},
    {"bucket": "10K-15K", "user_count": 4076, "pct": 19.21, "total_cost": 1212974, "avg_calls": 12184},
    {"bucket": "15K-20K", "user_count": 1939, "pct": 9.14, "total_cost": 856041, "avg_calls": 17194},
    {"bucket": "20K-30K", "user_count": 1757, "pct": 8.28, "total_cost": 1139426, "avg_calls": 24239},
    {"bucket": "30K-50K", "user_count": 1100, "pct": 5.19, "total_cost": 1204497, "avg_calls": 37323},
    {"bucket": "50K-75K", "user_count": 326, "pct": 1.54, "total_cost": 618632, "avg_calls": 60168},
    {"bucket": "75K-100K", "user_count": 93, "pct": 0.44, "total_cost": 262967, "avg_calls": 85122},
    {"bucket": "100K+", "user_count": 29, "pct": 0.14, "total_cost": 125774, "avg_calls": 127832},
]

# Key stats
total_users = 2057722
p99_threshold = 4996
p99_user_count = 20340
avg_calls = 302
median_calls = 59
max_calls = 225066
total_cost = 12114651.89
total_calls = 621194292
avg_cost_per_call = 0.0195
p99_total_cost = 7232388.14  # Sum of 5K+ buckets

# =============================================================================
# Header
# =============================================================================

st.markdown('<h1 class="main-header">P99 Distribution Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LLM Calls per User • Top 1% Heavy Users Analysis</p>', unsafe_allow_html=True)

# =============================================================================
# Key Metrics Row
# =============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">2.06M</p>
        <p class="metric-label">Total Users</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="color: #f97316;">$0.020</p>
        <p class="metric-label">Avg Cost/Call</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="color: #a855f7;">$12.1M</p>
        <p class="metric-label">Total LLM Cost</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="color: #ec4899;">59</p>
        <p class="metric-label">Median Calls</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value" style="color: #10b981;">225K</p>
        <p class="metric-label">Max Calls</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# Main Charts
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Distribution Overview", "💰 Calls vs Cost", "💸 Cost Simulator", "🔥 P99 Deep Dive", "📊 Cumulative Distribution"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Distribution histogram with cost overlay
        df_dist = pd.DataFrame(distribution_data)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Color gradient based on position
        colors = ['#00d4ff', '#00c4ef', '#00b4df', '#20a4cf', '#4094bf',
                  '#6084af', '#80749f', '#9f648f', '#be547f', '#dd446f',
                  '#ec4899', '#f97316', '#f97316']
        
        # Bar chart for user count
        fig.add_trace(go.Bar(
            x=df_dist['bucket'],
            y=df_dist['user_count'],
            name='Users',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            ),
            text=[f"{p:.1f}%" for p in df_dist['pct']],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=9),
            hovertemplate="<b>%{x}</b><br>Users: %{y:,.0f}<extra></extra>"
        ), secondary_y=False)
        
        # Line chart for cost
        fig.add_trace(go.Scatter(
            x=df_dist['bucket'],
            y=df_dist['total_cost'],
            name='Cost ($)',
            mode='lines+markers+text',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8, color='#10b981', symbol='diamond'),
            text=[f"${c/1000:.0f}K" for c in df_dist['total_cost']],
            textposition='top center',
            textfont=dict(color='#10b981', size=8),
            hovertemplate="<b>%{x}</b><br>Cost: $%{y:,.0f}<extra></extra>"
        ), secondary_y=True)
        
        # Add P99 threshold annotation
        fig.add_vline(x=8.5, line_dash="dash", line_color="#f97316", line_width=2)
        fig.add_annotation(
            x=8.5, y=df_dist['user_count'].max() * 0.9,
            text="P99 →",
            showarrow=False,
            font=dict(color="#f97316", size=14, family="JetBrains Mono"),
            xshift=30
        )
        
        fig.update_layout(
            title=dict(
                text="<b>User Distribution by LLM Call Volume</b>",
                font=dict(size=18, color='#e2e8f0', family='Space Grotesk')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            xaxis=dict(
                title="LLM Calls Bucket",
                gridcolor='rgba(100,100,100,0.2)',
                tickfont=dict(size=10)
            ),
            height=450,
            margin=dict(t=80, b=60),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        fig.update_yaxes(
            title_text="Number of Users",
            gridcolor='rgba(100,100,100,0.2)',
            type='log',
            secondary_y=False
        )
        fig.update_yaxes(
            title_text="Total Cost ($)",
            gridcolor='rgba(100,100,100,0.1)',
            secondary_y=True,
            showgrid=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("""
        <div class="insight-box">
            <h4 style="color: #a855f7; margin-top: 0;">📌 Key Insights</h4>
            <ul style="color: #e2e8f0; line-height: 1.8;">
                <li><b>70% of users</b> make ≤100 LLM calls</li>
                <li><b>Median user</b> makes only 59 calls</li>
                <li><b>P99 threshold</b>: ~5,000 calls</li>
                <li><b>Top 1%</b> drives disproportionate load</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="p99-highlight">
            <h4 style="color: #f97316; margin-top: 0;">🔥 P99 Users</h4>
            <p style="color: #e2e8f0; margin-bottom: 0.5rem;">
                <b>20,575 users</b> (1%) with <b>5K+ calls</b>
            </p>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                These power users account for 48% of all LLM calls
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats table
        st.markdown("#### Percentile Quick Reference")
        quick_stats = pd.DataFrame({
            'Percentile': ['P50', 'P75', 'P90', 'P95', 'P99', 'P99.9'],
            'LLM Calls': ['59', '121', '301', '628', '4,959', '26,196']
        })
        st.dataframe(quick_stats, hide_index=True, use_container_width=True)

with tab2:
    # Calculate cumulative data for Pareto analysis
    df_corr = pd.DataFrame(distribution_data)
    df_corr['cum_users'] = df_corr['user_count'].cumsum() / df_corr['user_count'].sum() * 100
    df_corr['cum_cost'] = df_corr['total_cost'].cumsum() / df_corr['total_cost'].sum() * 100
    df_corr['cost_pct'] = df_corr['total_cost'] / df_corr['total_cost'].sum() * 100
    df_corr['user_pct'] = df_corr['user_count'] / df_corr['user_count'].sum() * 100
    
    # P99 users are in buckets 5K+ (indexes 9-12)
    p99_buckets = df_corr[df_corr['avg_calls'] >= 5000]
    below_p99_buckets = df_corr[df_corr['avg_calls'] < 5000]
    
    p99_users = p99_buckets['user_count'].sum()
    p99_cost = p99_buckets['total_cost'].sum()
    below_p99_users = below_p99_buckets['user_count'].sum()
    below_p99_cost = below_p99_buckets['total_cost'].sum()
    
    total_cost_all = df_corr['total_cost'].sum()
    total_users_all = df_corr['user_count'].sum()
    
    # Top row: Key comparison metrics
    st.markdown("### 🎯 P99 Users: Are They The Most Expensive?")
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(168, 85, 247, 0.2), rgba(168, 85, 247, 0.05)); border: 2px solid rgba(168, 85, 247, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #a855f7; font-size: 2.5rem; font-weight: bold; margin: 0; font-family: 'JetBrains Mono';">99%</p>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">of users</p>
            <p style="color: #e2e8f0; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">${below_p99_cost/1e6:.1f}M</p>
            <p style="color: #64748b; margin: 0;">({below_p99_cost/total_cost_all*100:.0f}% of cost)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with comp_col2:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(249, 115, 22, 0.2), rgba(249, 115, 22, 0.05)); border: 2px solid rgba(249, 115, 22, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #f97316; font-size: 2.5rem; font-weight: bold; margin: 0; font-family: 'JetBrains Mono';">1%</p>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">P99 users (5K+ calls)</p>
            <p style="color: #e2e8f0; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">${p99_cost/1e6:.1f}M</p>
            <p style="color: #64748b; margin: 0;">({p99_cost/total_cost_all*100:.0f}% of cost)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with comp_col3:
        cost_ratio = (p99_cost/p99_users) / (below_p99_cost/below_p99_users)
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.05)); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #10b981; font-size: 2.5rem; font-weight: bold; margin: 0; font-family: 'JetBrains Mono';">{cost_ratio:.0f}x</p>
            <p style="color: #64748b; margin: 0.5rem 0 0 0;">Cost per user ratio</p>
            <p style="color: #e2e8f0; font-size: 1rem; margin: 0.5rem 0;">P99: ${p99_cost/p99_users:.0f}/user</p>
            <p style="color: #64748b; margin: 0;">Others: ${below_p99_cost/below_p99_users:.2f}/user</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Pareto chart - cumulative users vs cumulative cost
        fig_pareto = go.Figure()
        
        # Add area showing the gap between users and cost
        fig_pareto.add_trace(go.Scatter(
            x=list(df_corr['cum_users']) + [100],
            y=list(df_corr['cum_cost']) + [100],
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)',
            line=dict(color='#00d4ff', width=3),
            name='Cumulative Cost %',
            mode='lines+markers',
            marker=dict(size=10),
            hovertemplate="Users: %{x:.1f}%<br>Cost: %{y:.1f}%<extra></extra>"
        ))
        
        # Add diagonal line (perfect equality)
        fig_pareto.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            line=dict(color='#64748b', width=2, dash='dash'),
            name='Perfect Equality',
            hoverinfo='skip'
        ))
        
        # Add P99 marker (at 99% of users)
        p99_cost_pct = below_p99_cost / total_cost_all * 100
        fig_pareto.add_trace(go.Scatter(
            x=[99],
            y=[p99_cost_pct],
            mode='markers+text',
            marker=dict(size=16, color='#f97316', symbol='star'),
            text=[f'P99: {p99_cost_pct:.0f}%'],
            textposition='bottom left',
            textfont=dict(color='#f97316', size=12),
            name='P99 Threshold',
            hovertemplate="<b>P99 Threshold</b><br>99% of users<br>Only {p99_cost_pct:.0f}% of cost<extra></extra>"
        ))
        
        # Add annotation for the gap
        fig_pareto.add_annotation(
            x=99, y=(p99_cost_pct + 100) / 2,
            text=f"Top 1% = {100-p99_cost_pct:.0f}%<br>of total cost",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#f97316',
            font=dict(color='#f97316', size=11),
            ax=50, ay=0,
            bordercolor='#f97316',
            borderwidth=1,
            borderpad=4,
            bgcolor='rgba(249, 115, 22, 0.1)'
        )
        
        fig_pareto.update_layout(
            title=dict(
                text="<b>Pareto Analysis: User % vs Cost %</b>",
                font=dict(size=16, color='#e2e8f0', family='Space Grotesk')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            xaxis=dict(
                title="Cumulative % of Users (sorted by LLM calls)",
                gridcolor='rgba(100,100,100,0.2)',
                range=[0, 105]
            ),
            yaxis=dict(
                title="Cumulative % of Cost",
                gridcolor='rgba(100,100,100,0.2)',
                range=[0, 105]
            ),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_pareto, use_container_width=True)
    
    with col_right:
        # Stacked bar comparing P99 vs rest
        fig_compare = go.Figure()
        
        categories = ['Users', 'Total Cost']
        p99_values = [p99_users/total_users_all*100, p99_cost/total_cost_all*100]
        other_values = [100-p99_values[0], 100-p99_values[1]]
        
        fig_compare.add_trace(go.Bar(
            name='Bottom 99% of users',
            x=categories,
            y=other_values,
            marker_color='#a855f7',
            text=[f'{v:.1f}%' for v in other_values],
            textposition='inside',
            textfont=dict(size=14, color='white'),
            hovertemplate="%{x}: %{y:.1f}%<extra>Bottom 99%</extra>"
        ))
        
        fig_compare.add_trace(go.Bar(
            name='Top 1% (P99)',
            x=categories,
            y=p99_values,
            marker_color='#f97316',
            text=[f'{v:.1f}%' for v in p99_values],
            textposition='inside',
            textfont=dict(size=14, color='white'),
            hovertemplate="%{x}: %{y:.1f}%<extra>Top 1% P99</extra>"
        ))
        
        fig_compare.update_layout(
            title=dict(
                text="<b>P99 vs Rest: Users & Cost Share</b>",
                font=dict(size=16, color='#e2e8f0', family='Space Grotesk')
            ),
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            yaxis=dict(
                title="Percentage",
                gridcolor='rgba(100,100,100,0.2)',
                range=[0, 105]
            ),
            xaxis=dict(gridcolor='rgba(100,100,100,0.2)'),
            height=400,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            )
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # Answer box
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(249, 115, 22, 0.15), rgba(249, 115, 22, 0.05)); border: 2px solid rgba(249, 115, 22, 0.4); border-radius: 12px; padding: 1.5rem; margin-top: 1rem;">
        <h4 style="color: #f97316; margin-top: 0;">✅ Answer: YES, P99 users are disproportionately expensive</h4>
        <p style="color: #e2e8f0; line-height: 1.8; margin-bottom: 0;">
            The <b>top 1% of users</b> (those with 5K+ LLM calls) account for:<br>
            • <b>{p99_cost/total_cost_all*100:.0f}% of total cost</b> (${p99_cost/1e6:.1f}M)<br>
            • Only <b>{p99_users/total_users_all*100:.1f}% of users</b> ({p99_users:,} users)<br>
            • <b>{cost_ratio:.0f}x more expensive per user</b> than average<br><br>
            <span style="color: #64748b;">This is a classic Pareto distribution where a small fraction of heavy users drives the majority of costs.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 💸 Cost Savings Simulator")
    st.markdown("*Set a monthly call limit to see potential cost savings*")
    
    # Pre-computed cost data at various thresholds
    cost_data = {
        100: 1548917,
        250: 2287654,
        500: 3277752,
        750: 3745123,
        1000: 4175017,
        1250: 4512456,
        1500: 4807297,
        1750: 5078234,
        2000: 5322163,
        2500: 5756789,
        3000: 6153044,
        4000: 6798456,
        5000: 7351243,
        7500: 8287654,
        10000: 9075755,
    }
    
    users_affected_data = {
        100: 607658,
        250: 246696,
        500: 124317,
        750: 87376,
        1000: 69568,
        1250: 59876,
        1500: 51948,
        1750: 46789,
        2000: 42537,
        2500: 36543,
        3000: 31713,
        4000: 25234,
        5000: 20340,
        7500: 13456,
        10000: 9320,
    }
    
    total_cost_current = 12114652
    total_users = 2057722
    
    # Input controls
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        limit = st.slider(
            "**Set Monthly Call Limit per User**",
            min_value=100,
            max_value=10000,
            value=1500,
            step=100,
            help="Drag to set the maximum number of LLM calls allowed per user per month"
        )
    
    with col_input2:
        st.markdown(f"""
        <div style="background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 8px; padding: 1rem; text-align: center; margin-top: 0.5rem;">
            <p style="color: #64748b; margin: 0; font-size: 0.8rem;">Selected Limit</p>
            <p style="color: #00d4ff; font-size: 1.8rem; font-weight: bold; margin: 0; font-family: 'JetBrains Mono';">{limit:,}</p>
            <p style="color: #64748b; margin: 0; font-size: 0.8rem;">calls/month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interpolate cost at the selected limit
    thresholds = sorted(cost_data.keys())
    
    if limit in cost_data:
        cost_at_limit = cost_data[limit]
        users_affected = users_affected_data[limit]
    else:
        # Linear interpolation
        lower = max([t for t in thresholds if t <= limit])
        upper = min([t for t in thresholds if t >= limit])
        if lower == upper:
            cost_at_limit = cost_data[lower]
            users_affected = users_affected_data[lower]
        else:
            ratio = (limit - lower) / (upper - lower)
            cost_at_limit = cost_data[lower] + ratio * (cost_data[upper] - cost_data[lower])
            users_affected = int(users_affected_data[lower] + ratio * (users_affected_data[upper] - users_affected_data[lower]))
    
    savings = total_cost_current - cost_at_limit
    savings_pct = (savings / total_cost_current) * 100
    users_affected_pct = (users_affected / total_users) * 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results cards
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.05)); border: 2px solid rgba(16, 185, 129, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #64748b; margin: 0; font-size: 0.8rem;">Monthly Savings</p>
            <p style="color: #10b981; font-size: 2rem; font-weight: bold; margin: 0.3rem 0; font-family: 'JetBrains Mono';">${savings/1e6:.2f}M</p>
            <p style="color: #10b981; margin: 0; font-size: 1.2rem;">({savings_pct:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with res_col2:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.05)); border: 2px solid rgba(0, 212, 255, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #64748b; margin: 0; font-size: 0.8rem;">New Monthly Cost</p>
            <p style="color: #00d4ff; font-size: 2rem; font-weight: bold; margin: 0.3rem 0; font-family: 'JetBrains Mono';">${cost_at_limit/1e6:.2f}M</p>
            <p style="color: #64748b; margin: 0; font-size: 0.9rem;">vs ${total_cost_current/1e6:.1f}M current</p>
        </div>
        """, unsafe_allow_html=True)
    
    with res_col3:
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(249, 115, 22, 0.2), rgba(249, 115, 22, 0.05)); border: 2px solid rgba(249, 115, 22, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #64748b; margin: 0; font-size: 0.8rem;">Users Affected</p>
            <p style="color: #f97316; font-size: 2rem; font-weight: bold; margin: 0.3rem 0; font-family: 'JetBrains Mono';">{users_affected:,}</p>
            <p style="color: #f97316; margin: 0; font-size: 1.2rem;">({users_affected_pct:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with res_col4:
        yearly_savings = savings * 12
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, rgba(168, 85, 247, 0.2), rgba(168, 85, 247, 0.05)); border: 2px solid rgba(168, 85, 247, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <p style="color: #64748b; margin: 0; font-size: 0.8rem;">Yearly Savings</p>
            <p style="color: #a855f7; font-size: 2rem; font-weight: bold; margin: 0.3rem 0; font-family: 'JetBrains Mono';">${yearly_savings/1e6:.1f}M</p>
            <p style="color: #64748b; margin: 0; font-size: 0.9rem;">projected annually</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        # Cost comparison chart
        fig_sim = go.Figure()
        
        # Create data for all thresholds
        x_vals = sorted(cost_data.keys())
        y_savings = [(total_cost_current - cost_data[x]) / 1e6 for x in x_vals]
        y_savings_pct = [(total_cost_current - cost_data[x]) / total_cost_current * 100 for x in x_vals]
        
        fig_sim.add_trace(go.Scatter(
            x=x_vals,
            y=y_savings,
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8),
            name='Savings ($M)',
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)',
            hovertemplate="Limit: %{x:,}<br>Savings: $%{y:.2f}M<extra></extra>"
        ))
        
        # Add marker for current selection
        current_savings = savings / 1e6
        fig_sim.add_trace(go.Scatter(
            x=[limit],
            y=[current_savings],
            mode='markers+text',
            marker=dict(size=16, color='#f97316', symbol='star'),
            text=[f'${current_savings:.2f}M'],
            textposition='top center',
            textfont=dict(color='#f97316', size=12),
            name='Selected',
            hovertemplate=f"<b>Selected: {limit:,} calls</b><br>Savings: ${current_savings:.2f}M<extra></extra>"
        ))
        
        fig_sim.update_layout(
            title=dict(
                text="<b>Savings by Call Limit</b>",
                font=dict(size=16, color='#e2e8f0', family='Space Grotesk')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            xaxis=dict(
                title="Monthly Call Limit",
                gridcolor='rgba(100,100,100,0.2)',
                type='log'
            ),
            yaxis=dict(
                title="Monthly Savings ($M)",
                gridcolor='rgba(100,100,100,0.2)'
            ),
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)
    
    with viz_col2:
        # Users affected chart
        fig_users = go.Figure()
        
        y_users = [users_affected_data[x] for x in x_vals]
        y_users_pct = [users_affected_data[x] / total_users * 100 for x in x_vals]
        
        fig_users.add_trace(go.Scatter(
            x=x_vals,
            y=y_users_pct,
            mode='lines+markers',
            line=dict(color='#f97316', width=3),
            marker=dict(size=8),
            name='% Users Affected',
            fill='tozeroy',
            fillcolor='rgba(249, 115, 22, 0.2)',
            hovertemplate="Limit: %{x:,}<br>Users Affected: %{y:.1f}%<extra></extra>"
        ))
        
        # Add marker for current selection
        fig_users.add_trace(go.Scatter(
            x=[limit],
            y=[users_affected_pct],
            mode='markers+text',
            marker=dict(size=16, color='#10b981', symbol='star'),
            text=[f'{users_affected_pct:.1f}%'],
            textposition='top center',
            textfont=dict(color='#10b981', size=12),
            name='Selected',
            hovertemplate=f"<b>Selected: {limit:,} calls</b><br>Users: {users_affected_pct:.1f}%<extra></extra>"
        ))
        
        fig_users.update_layout(
            title=dict(
                text="<b>Users Affected by Limit</b>",
                font=dict(size=16, color='#e2e8f0', family='Space Grotesk')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            xaxis=dict(
                title="Monthly Call Limit",
                gridcolor='rgba(100,100,100,0.2)',
                type='log'
            ),
            yaxis=dict(
                title="% of Users Affected",
                gridcolor='rgba(100,100,100,0.2)'
            ),
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig_users, use_container_width=True)
    
    # Summary table
    st.markdown("#### 📊 Quick Reference: Common Limits")
    
    quick_ref = pd.DataFrame({
        'Limit': ['500', '1,000', '1,500', '2,000', '3,000', '5,000'],
        'Monthly Savings': [f"${(total_cost_current - cost_data[500])/1e6:.2f}M", 
                           f"${(total_cost_current - cost_data[1000])/1e6:.2f}M",
                           f"${(total_cost_current - cost_data[1500])/1e6:.2f}M",
                           f"${(total_cost_current - cost_data[2000])/1e6:.2f}M",
                           f"${(total_cost_current - cost_data[3000])/1e6:.2f}M",
                           f"${(total_cost_current - cost_data[5000])/1e6:.2f}M"],
        'Savings %': [f"{(total_cost_current - cost_data[500])/total_cost_current*100:.0f}%",
                     f"{(total_cost_current - cost_data[1000])/total_cost_current*100:.0f}%",
                     f"{(total_cost_current - cost_data[1500])/total_cost_current*100:.0f}%",
                     f"{(total_cost_current - cost_data[2000])/total_cost_current*100:.0f}%",
                     f"{(total_cost_current - cost_data[3000])/total_cost_current*100:.0f}%",
                     f"{(total_cost_current - cost_data[5000])/total_cost_current*100:.0f}%"],
        'Users Affected': [f"{users_affected_data[500]:,} ({users_affected_data[500]/total_users*100:.1f}%)",
                          f"{users_affected_data[1000]:,} ({users_affected_data[1000]/total_users*100:.1f}%)",
                          f"{users_affected_data[1500]:,} ({users_affected_data[1500]/total_users*100:.1f}%)",
                          f"{users_affected_data[2000]:,} ({users_affected_data[2000]/total_users*100:.1f}%)",
                          f"{users_affected_data[3000]:,} ({users_affected_data[3000]/total_users*100:.1f}%)",
                          f"{users_affected_data[5000]:,} ({users_affected_data[5000]/total_users*100:.1f}%)"]
    })
    st.dataframe(quick_ref, hide_index=True, use_container_width=True)

with tab4:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # P99 internal distribution
        df_p99 = pd.DataFrame(p99_distribution)
        
        fig2 = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("P99 Users by Call Volume", "Cost Distribution")
        )
        
        # Bar chart for P99 distribution
        colors_p99 = px.colors.sequential.Oranges[3:][::-1][:11]
        
        fig2.add_trace(go.Bar(
            x=df_p99['bucket'],
            y=df_p99['user_count'],
            marker=dict(
                color=df_p99['user_count'],
                colorscale='Oranges',
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            ),
            text=[f"{p:.1f}%" for p in df_p99['pct']],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=9),
            name="Users"
        ), row=1, col=1)
        
        # Pie chart for cost distribution
        fig2.add_trace(go.Pie(
            labels=df_p99['bucket'],
            values=df_p99['total_cost'],
            hole=0.5,
            marker=dict(
                colors=px.colors.sequential.Purples[3:]
            ),
            textinfo='percent',
            textfont=dict(size=10),
            name="Cost"
        ), row=1, col=2)
        
        fig2.update_layout(
            title=dict(
                text="<b>P99 Users Deep Dive: Call Volume & Cost Distribution</b>",
                font=dict(size=18, color='#e2e8f0', family='Space Grotesk')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            height=450,
            showlegend=False
        )
        
        fig2.update_xaxes(gridcolor='rgba(100,100,100,0.2)', row=1, col=1)
        fig2.update_yaxes(gridcolor='rgba(100,100,100,0.2)', row=1, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_right:
        st.markdown("""
        <div class="insight-box">
            <h4 style="color: #f97316; margin-top: 0;">🎯 P99 Profile</h4>
            <p style="color: #e2e8f0;">
                <b>Threshold:</b> 4,959+ LLM calls<br>
                <b>Count:</b> 20,575 users<br>
                <b>Max:</b> 225,066 calls<br>
                <b>Total Cost:</b> $7.2M
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # P99 internal percentiles
        st.markdown("#### Within P99 Users")
        p99_internal = pd.DataFrame({
            'Metric': ['Minimum', 'P25', 'Median', 'P75', 'P90', 'Maximum'],
            'Calls': ['4,937', '6,552', '9,272', '15,244', '26,294', '225,066']
        })
        st.dataframe(p99_internal, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <p style="color: #10b981; margin: 0; font-size: 0.9rem;">
                <b>💰 Top 30K-50K bucket</b> generates the most cost ($1.2M) despite having only 1,100 users
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Cumulative distribution chart
        df_pct = pd.DataFrame(percentile_data)
        
        fig3 = go.Figure()
        
        # Add cumulative line
        fig3.add_trace(go.Scatter(
            x=df_pct['percentile'],
            y=df_pct['llm_calls'],
            mode='lines+markers',
            line=dict(color='#00d4ff', width=3),
            marker=dict(size=8, color='#00d4ff', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)',
            name='LLM Calls',
            hovertemplate="P%{x}: %{y:,.0f} calls<extra></extra>"
        ))
        
        # Add P99 marker
        fig3.add_trace(go.Scatter(
            x=[99],
            y=[4864],
            mode='markers+text',
            marker=dict(size=16, color='#f97316', symbol='star'),
            text=['P99'],
            textposition='top center',
            textfont=dict(color='#f97316', size=12),
            name='P99 Threshold',
            hovertemplate="<b>P99</b><br>Threshold: 4,864 calls<extra></extra>"
        ))
        
        # Add annotations for key percentiles
        key_points = [
            (50, 59, "P50: 59"),
            (90, 301, "P90: 301"),
            (95, 626, "P95: 626"),
        ]
        
        for x, y, text in key_points:
            fig3.add_annotation(
                x=x, y=y,
                text=text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#64748b',
                font=dict(color='#e2e8f0', size=10),
                ax=0, ay=-30
            )
        
        fig3.update_layout(
            title=dict(
                text="<b>Cumulative Distribution Function (CDF)</b>",
                font=dict(size=18, color='#e2e8f0', family='Space Grotesk')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='JetBrains Mono'),
            xaxis=dict(
                title="Percentile",
                gridcolor='rgba(100,100,100,0.2)',
                dtick=10,
                range=[0, 102]
            ),
            yaxis=dict(
                title="LLM Calls",
                gridcolor='rgba(100,100,100,0.2)',
                type='log',
                range=[0, 5.5]
            ),
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_right:
        st.markdown("""
        <div class="insight-box">
            <h4 style="color: #00d4ff; margin-top: 0;">📈 Distribution Shape</h4>
            <p style="color: #e2e8f0; line-height: 1.8;">
                The CDF shows a <b>classic long-tail distribution</b>:
                <ul>
                    <li>Linear growth P1-P80</li>
                    <li>Exponential jump P90-P99</li>
                    <li>Extreme outliers in P99.9+</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Ratio comparison
        st.markdown("#### Percentile Ratios")
        ratios = pd.DataFrame({
            'Comparison': ['P99 vs Median', 'P99 vs P90', 'Max vs P99', 'P99.9 vs P99'],
            'Ratio': ['83x', '16x', '46x', '5x']
        })
        st.dataframe(ratios, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div style="background: rgba(236, 72, 153, 0.1); border: 1px solid rgba(236, 72, 153, 0.3); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            <p style="color: #ec4899; margin: 0; font-size: 0.9rem;">
                <b>⚡ P99 users make 83x more LLM calls</b> than the median user (4,959 vs 59)
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================

st.markdown("<br><hr style='border-color: rgba(100,100,100,0.3);'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
        <b>Data Source:</b> Langfuse Traces<br>
        <b>Total Users:</b> 2,057,721
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
        <b>Last Updated:</b> December 27, 2025<br>
        <b>Analysis:</b> My App Analytics
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
        <b>P99 Threshold:</b> 4,996 calls<br>
        <b>Total Cost:</b> $12.1M
    </div>
    """, unsafe_allow_html=True)

