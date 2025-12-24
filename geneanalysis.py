# app.py - GEO Expression Analysis Platform
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="GEO Expression Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS FOR BETTER STYLING
# ======================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stDownloadButton > button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        width: 100%;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 2px solid #2E86AB;
        color: #666;
        font-size: 0.9rem;
    }
    .plot-container {
        margin: 2rem 0;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .about-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #2E86AB;
    }
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 2px dashed #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# INITIALIZE SESSION STATE
# ======================================================
if 'expr_data' not in st.session_state:
    st.session_state.expr_data = None
if 'normalized_data' not in st.session_state:
    st.session_state.normalized_data = None
if 'de_results' not in st.session_state:
    st.session_state.de_results = None

# ======================================================
# DATA PROCESSING FUNCTIONS
# ======================================================
@st.cache_data(show_spinner=True)
def load_expression_data(file):
    """Load expression data from various file formats"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, index_col=0)
        elif file.name.endswith('.tsv'):
            df = pd.read_csv(file, sep='\t', index_col=0)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, index_col=0)
        else:
            st.error(f"Unsupported file format: {file.name}")
            return None
        
        # Basic data cleaning
        df = df.dropna(how='all', axis=1)
        df = df.loc[~df.index.duplicated(keep='first')]
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def normalize_data(df, method='log2'):
    """Normalize expression data"""
    if method == 'log2':
        return np.log2(df + 1)
    elif method == 'zscore':
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.T).T
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)
    else:
        return df

def differential_expression_analysis(expr_df, group_a, group_b, test='ttest'):
    """Perform differential expression analysis"""
    results = []
    
    for gene in expr_df.index:
        expr_a = expr_df.loc[gene, group_a].values
        expr_b = expr_df.loc[gene, group_b].values
        
        if len(expr_a) < 2 or len(expr_b) < 2:
            continue
        
        # Calculate means and fold change
        mean_a = np.mean(expr_a)
        mean_b = np.mean(expr_b)
        
        if mean_a == 0 and mean_b == 0:
            log2fc = 0
        elif mean_a == 0:
            log2fc = 10 if mean_b > 0 else -10
        else:
            log2fc = np.log2((mean_b + 1) / (mean_a + 1))
        
        # Statistical test
        if test == 'ttest':
            _, pval = ttest_ind(expr_a, expr_b, equal_var=False)
        elif test == 'mannwhitney':
            _, pval = mannwhitneyu(expr_a, expr_b, alternative='two-sided')
        
        results.append({
            'Gene': gene,
            'Mean_GroupA': mean_a,
            'Mean_GroupB': mean_b,
            'Log2FC': log2fc,
            'PValue': pval,
            'Abs_Log2FC': abs(log2fc)
        })
    
    if not results:
        return None
    
    # Create results dataframe
    result_df = pd.DataFrame(results)
    
    # Multiple testing correction
    result_df['PValue_Adj'] = multipletests(
        result_df['PValue'], 
        method='fdr_bh'
    )[1]
    
    # Get thresholds from session state
    pval_threshold = st.session_state.get('pval_threshold', 0.05)
    fc_threshold = st.session_state.get('fc_threshold', 2.0)
    
    # Mark significant genes
    result_df['Significant'] = (
        (result_df['PValue_Adj'] < pval_threshold) & 
        (result_df['Abs_Log2FC'] > np.log2(fc_threshold))
    )
    
    result_df['Neg_log10_PValue'] = -np.log10(result_df['PValue_Adj'])
    
    return result_df.sort_values('PValue_Adj')

# ======================================================
# VISUALIZATION FUNCTIONS
# ======================================================
def create_volcano_plot(de_results):
    """Create volcano plot"""
    # Prepare data
    de_results = de_results.copy()
    de_results['Color'] = 'gray'
    de_results.loc[(de_results['Significant']) & (de_results['Log2FC'] > 0), 'Color'] = 'red'
    de_results.loc[(de_results['Significant']) & (de_results['Log2FC'] < 0), 'Color'] = 'blue'
    
    pval_threshold = st.session_state.get('pval_threshold', 0.05)
    fc_threshold = st.session_state.get('fc_threshold', 2.0)
    
    # Create plot
    fig = px.scatter(
        de_results,
        x='Log2FC',
        y='Neg_log10_PValue',
        color='Color',
        hover_data=['Gene'],
        color_discrete_map={'red': 'red', 'blue': 'blue', 'gray': 'lightgray'},
        title='Volcano Plot - Differential Expression'
    )
    
    fig.update_layout(
        xaxis_title="Log2 Fold Change",
        yaxis_title="-log10(Adjusted P-value)",
        showlegend=False,
        height=500
    )
    
    # Add significance lines
    fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="black")
    fig.add_vline(x=np.log2(fc_threshold), line_dash="dash", line_color="green")
    fig.add_vline(x=-np.log2(fc_threshold), line_dash="dash", line_color="green")
    
    return fig

def create_pca_plot(expr_df):
    """Create PCA plot"""
    # Select top variable genes
    var_genes = expr_df.var(axis=1).sort_values(ascending=False)
    top_genes = var_genes.head(min(1000, len(var_genes))).index
    
    # Prepare data
    data_scaled = StandardScaler().fit_transform(expr_df.loc[top_genes].T)
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create dataframe
    pca_df = pd.DataFrame(
        pca_result,
        columns=['PC1', 'PC2', 'PC3'],
        index=expr_df.columns
    )
    
    # Calculate variance explained
    var_exp = pca.explained_variance_ratio_ * 100
    
    # Create 3D plot
    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        hover_name=pca_df.index,
        title=f'PCA Analysis (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%, PC3: {var_exp[2]:.1f}%)'
    )
    
    fig.update_layout(height=600)
    
    return fig, pca_df

def create_heatmap(expr_df, n_genes=50):
    """Create expression heatmap"""
    # Select top variable genes
    var_genes = expr_df.var(axis=1).sort_values(ascending=False)
    top_genes = var_genes.head(min(n_genes, len(var_genes))).index
    
    # Create heatmap
    fig = px.imshow(
        expr_df.loc[top_genes],
        color_continuous_scale='viridis',
        title=f'Expression Heatmap (Top {len(top_genes)} Variable Genes)',
        labels=dict(color="Expression Level"),
        aspect="auto"
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_distribution_plot(expr_df):
    """Create expression distribution plot"""
    expr_values_log = np.log2(expr_df.values.flatten() + 1)
    
    fig = px.histogram(
        x=expr_values_log,
        nbins=100,
        title='Expression Distribution',
        labels={'x': 'log2(Expression + 1)', 'y': 'Count'},
        color_discrete_sequence=['skyblue']
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

# ======================================================
# ABOUT SECTION FUNCTION
# ======================================================
def show_about_section():
    """Display the About section"""
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    
    st.markdown("""
    <h2 style='color: #2E86AB;'>🧬 Welcome to GEO Expression Analysis Platform</h2>
    
    <p style='font-size: 1.1rem;'>
    A comprehensive web-based tool for analyzing gene expression data from GEO datasets and other sources. 
    This platform is designed to help researchers perform sophisticated bioinformatics analyses 
    without requiring extensive programming experience.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='padding: 1rem; background: white; border-radius: 10px;'>
        <h4>📊 Key Features</h4>
        <ul>
        <li>Differential Expression Analysis</li>
        <li>Principal Component Analysis (PCA)</li>
        <li>Expression Heatmaps</li>
        <li>Statistical Testing</li>
        <li>Data Normalization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 1rem; background: white; border-radius: 10px;'>
        <h4>🔬 Applications</h4>
        <ul>
        <li>Cancer Research</li>
        <li>Drug Development</li>
        <li>Biomarker Discovery</li>
        <li>Functional Genomics</li>
        <li>Comparative Studies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='padding: 1rem; background: white; border-radius: 10px;'>
        <h4>📁 Supported Formats</h4>
        <ul>
        <li>CSV (.csv)</li>
        <li>Tab-separated (.tsv)</li>
        <li>Excel (.xlsx)</li>
        <li>Gene expression matrices</li>
        <li>Count matrices</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='margin-top: 2rem; padding: 1.5rem; background: white; border-radius: 10px;'>
    <h4>🚀 Getting Started</h4>
    <p>
    1. Upload your expression data using the uploader below<br>
    2. Perform quality control and normalization<br>
    3. Run differential expression analysis<br>
    4. Visualize and export your results
    </p>
    <p style='color: #A23B72; font-weight: bold;'>
    ⬇️ Use the file uploader below to begin your analysis.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# MAIN APPLICATION
# ======================================================
def main():
    # HEADER
    st.markdown('<h1 class="main-header">🧬 GEO Expression Analysis Platform</h1>', unsafe_allow_html=True)
    
    # SIDEBAR CONFIGURATION
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        
        # P-value threshold
        pval_threshold = st.slider(
            "P-value threshold",
            min_value=0.001,
            max_value=0.1,
            value=0.05,
            step=0.005,
            key="pval_slider"
        )
        st.session_state.pval_threshold = pval_threshold
        
        # Fold change threshold
        fc_threshold = st.slider(
            "Fold Change threshold",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key="fc_slider"
        )
        st.session_state.fc_threshold = fc_threshold
        
        st.markdown("---")
        st.markdown("### 📊 Current Status")
        
        if st.session_state.expr_data is not None:
            st.success("✅ Data loaded")
            st.write(f"Genes: {st.session_state.expr_data.shape[0]}")
            st.write(f"Samples: {st.session_state.expr_data.shape[1]}")
        else:
            st.info("No data loaded")
    
    # ======================================================
    # SHOW ABOUT SECTION AND FILE UPLOADER WHEN NO DATA IS LOADED
    # ======================================================
    if st.session_state.expr_data is None:
        # Show About section
        show_about_section()
        
        # Show file uploader section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">📤 Upload Your Data</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose expression matrix file",
                type=['csv', 'tsv', 'xlsx'],
                key="main_file_uploader",
                help="Upload your gene expression data matrix"
            )
            
            if uploaded_file is not None:
                with st.spinner("Loading data..."):
                    expr_data = load_expression_data(uploaded_file)
                    if expr_data is not None:
                        st.session_state.expr_data = expr_data
                        st.success(f"✅ Successfully loaded {expr_data.shape[0]} genes and {expr_data.shape[1]} samples")
                        st.rerun()
        
        with col2:
            st.markdown("""
            <div style='padding: 1.5rem; background: #f0f7ff; border-radius: 10px;'>
            <h4>📋 File Requirements</h4>
            <ul style='margin-bottom: 0;'>
            <li>Rows: Genes</li>
            <li>Columns: Samples</li>
            <li>First column: Gene IDs</li>
            <li>CSV, TSV, or Excel format</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Example data for testing
        with st.expander("📥 Need test data?"):
            st.markdown("""
            ### Create test data
            
            Copy this example CSV content to a file:
            
            ```csv
            Gene,Sample1,Sample2,Sample3,Sample4,Sample5,Sample6
            Gene1,10.5,12.3,8.7,22.1,9.8,20.5
            Gene2,5.2,6.8,4.9,15.3,5.1,16.2
            Gene3,8.9,10.2,7.5,18.9,8.3,19.7
            Gene4,12.1,14.5,10.8,25.3,11.9,24.1
            Gene5,3.8,4.2,3.1,9.5,3.5,8.9
            ```
            
            **Groups for analysis:**
            - Group A: Sample1, Sample2, Sample3
            - Group B: Sample4, Sample5, Sample6
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # ======================================================
        # MAIN CONTENT - TABS (only show when data is loaded)
        # ======================================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Data Upload & Preprocessing",
            "📊 Quality Control",
            "🔍 Differential Expression",
            "📈 Visualizations"
        ])
        
        # ======================================================
        # TAB 1: DATA UPLOAD & PREPROCESSING
        # ======================================================
        with tab1:
            st.markdown('<h2 class="sub-header">Data Upload & Preprocessing</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Data")
                st.info(f"**{st.session_state.expr_data.shape[0]}** genes × **{st.session_state.expr_data.shape[1]}** samples")
                
                # Show data preview
                with st.expander("View Data Preview (first 10 rows)"):
                    st.dataframe(
                        st.session_state.expr_data.head(10),
                        use_container_width=True,
                        height=300
                    )
                
                # Upload new file option
                st.markdown("#### Upload New Data")
                new_file = st.file_uploader(
                    "Upload a new file",
                    type=['csv', 'tsv', 'xlsx'],
                    key="tab1_file_uploader"
                )
                
                if new_file:
                    with st.spinner("Loading new data..."):
                        new_data = load_expression_data(new_file)
                        if new_data is not None:
                            st.session_state.expr_data = new_data
                            st.session_state.normalized_data = None
                            st.session_state.de_results = None
                            st.success("✅ New data loaded successfully!")
                            st.rerun()
            
            with col2:
                st.markdown("#### Data Normalization")
                
                norm_method = st.selectbox(
                    "Select normalization method:",
                    ["None (raw data)", "Log2 transform", "Z-score normalization"],
                    key="norm_select"
                )
                
                if st.button("Apply Normalization", type="primary", key="apply_norm"):
                    if norm_method == "Log2 transform":
                        st.session_state.normalized_data = normalize_data(
                            st.session_state.expr_data,
                            method='log2'
                        )
                    elif norm_method == "Z-score normalization":
                        st.session_state.normalized_data = normalize_data(
                            st.session_state.expr_data,
                            method='zscore'
                        )
                    else:
                        st.session_state.normalized_data = st.session_state.expr_data
                    
                    st.success(f"✅ Applied {norm_method}")
                    
                    if st.session_state.normalized_data is not None:
                        with st.expander("Normalized Data Preview"):
                            st.dataframe(
                                st.session_state.normalized_data.head(10),
                                use_container_width=True
                            )
                
                # Clear data button
                st.markdown("---")
                if st.button("🗑️ Clear All Data & Return to Home", type="secondary"):
                    st.session_state.expr_data = None
                    st.session_state.normalized_data = None
                    st.session_state.de_results = None
                    st.rerun()
        
        # ======================================================
        # TAB 2: QUALITY CONTROL
        # ======================================================
        with tab2:
            st.markdown('<h2 class="sub-header">Quality Control & Statistics</h2>', unsafe_allow_html=True)
            
            # Use normalized data if available
            expr_data = st.session_state.normalized_data if st.session_state.normalized_data is not None else st.session_state.expr_data
            
            # Summary statistics in cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Genes", f"{expr_data.shape[0]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Samples", f"{expr_data.shape[1]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                mean_expr = expr_data.values.mean()
                st.metric("Mean Expression", f"{mean_expr:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                missing = expr_data.isna().sum().sum()
                st.metric("Missing Values", f"{missing:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Distribution plot
            st.markdown("#### Expression Distribution")
            dist_fig = create_distribution_plot(expr_data)
            st.plotly_chart(dist_fig, use_container_width=True, key="dist_plot_tab2")
            
            # Correlation heatmap
            st.markdown("#### Sample Correlation")
            corr_matrix = expr_data.corr()
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale='viridis',
                title="Sample Correlation Matrix",
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_plot_tab2")
        
        # ======================================================
        # TAB 3: DIFFERENTIAL EXPRESSION
        # ======================================================
        with tab3:
            st.markdown('<h2 class="sub-header">Differential Expression Analysis</h2>', unsafe_allow_html=True)
            
            expr_data = st.session_state.normalized_data if st.session_state.normalized_data is not None else st.session_state.expr_data
            
            # Group selection
            st.markdown("#### Define Groups for Comparison")
            
            all_samples = expr_data.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Group A")
                group_a = st.multiselect(
                    "Select samples for Group A:",
                    all_samples,
                    key="group_a_select"
                )
                
                if group_a:
                    st.info(f"Selected {len(group_a)} samples")
            
            with col2:
                st.markdown("##### Group B")
                available_samples = [s for s in all_samples if s not in group_a]
                group_b = st.multiselect(
                    "Select samples for Group B:",
                    available_samples,
                    key="group_b_select"
                )
                
                if group_b:
                    st.info(f"Selected {len(group_b)} samples")
            
            # Analysis parameters
            st.markdown("#### Analysis Parameters")
            test_method = st.radio(
                "Statistical test:",
                ["T-test", "Mann-Whitney U test"],
                horizontal=True,
                key="test_select"
            )
            
            # Run analysis
            if st.button("Run Differential Expression Analysis", type="primary", key="run_de"):
                if len(group_a) >= 2 and len(group_b) >= 2:
                    with st.spinner("Performing analysis..."):
                        de_results = differential_expression_analysis(
                            expr_data,
                            group_a,
                            group_b,
                            test='ttest' if test_method == 'T-test' else 'mannwhitney'
                        )
                        
                        if de_results is not None:
                            st.session_state.de_results = de_results
                            st.success("✅ Analysis completed!")
                        else:
                            st.error("Analysis failed. Please check your data.")
                else:
                    st.warning("Please select at least 2 samples for each group")
            
            # Display results
            if st.session_state.de_results is not None:
                de_results = st.session_state.de_results
                
                # Summary statistics
                sig_genes = de_results[de_results['Significant']]
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Significant Genes", len(sig_genes))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    up_reg = len(sig_genes[sig_genes['Log2FC'] > 0])
                    st.metric("Up-regulated", up_reg)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col5:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    down_reg = len(sig_genes[sig_genes['Log2FC'] < 0])
                    st.metric("Down-regulated", down_reg)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Results table
                st.markdown("#### Top Differential Genes")
                
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Adjusted P-value", "Absolute Fold Change", "Fold Change"],
                    key="sort_select"
                )
                
                if sort_by == "Adjusted P-value":
                    sorted_results = de_results.sort_values('PValue_Adj')
                elif sort_by == "Absolute Fold Change":
                    sorted_results = de_results.sort_values('Abs_Log2FC', ascending=False)
                else:
                    sorted_results = de_results.sort_values('Log2FC', ascending=False)
                
                # Display table
                st.dataframe(
                    sorted_results.head(100),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv_data = de_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv_data,
                    file_name="differential_expression_results.csv",
                    mime="text/csv",
                    key="download_de"
                )
        
        # ======================================================
        # TAB 4: VISUALIZATIONS
        # ======================================================
        with tab4:
            st.markdown('<h2 class="sub-header">Advanced Visualizations</h2>', unsafe_allow_html=True)
            
            expr_data = st.session_state.normalized_data if st.session_state.normalized_data is not None else st.session_state.expr_data
            
            # Visualization selection
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Volcano Plot", "PCA Analysis", "Expression Heatmap", "Distribution Plot"],
                key="viz_select"
            )
            
            if viz_type == "Volcano Plot":
                if st.session_state.de_results is not None:
                    st.markdown("#### Volcano Plot")
                    volcano_fig = create_volcano_plot(st.session_state.de_results)
                    st.plotly_chart(volcano_fig, use_container_width=True, key="volcano_plot_tab4")
                    
                    # Download data
                    de_csv = st.session_state.de_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Volcano Plot Data (CSV)",
                        data=de_csv,
                        file_name="volcano_plot_data.csv",
                        mime="text/csv",
                        key="download_volcano"
                    )
                else:
                    st.info("Please run differential expression analysis in Tab 3 first")
            
            elif viz_type == "PCA Analysis":
                st.markdown("#### Principal Component Analysis")
                pca_fig, pca_df = create_pca_plot(expr_data)
                st.plotly_chart(pca_fig, use_container_width=True, key="pca_plot_tab4")
                
                # Download PCA data
                pca_csv = pca_df.to_csv()
                st.download_button(
                    label="📥 Download PCA Coordinates (CSV)",
                    data=pca_csv,
                    file_name="pca_coordinates.csv",
                    mime="text/csv",
                    key="download_pca"
                )
            
            elif viz_type == "Expression Heatmap":
                st.markdown("#### Expression Heatmap")
                
                n_genes = st.slider(
                    "Number of genes to display:",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    key="heatmap_slider_tab4"
                )
                
                heatmap_fig = create_heatmap(expr_data, n_genes=n_genes)
                st.plotly_chart(heatmap_fig, use_container_width=True, key="heatmap_plot_tab4")
                
                # Download heatmap data
                top_genes = expr_data.var(axis=1).sort_values(ascending=False).head(n_genes).index
                heatmap_data = expr_data.loc[top_genes]
                heatmap_csv = heatmap_data.to_csv()
                
                st.download_button(
                    label="📥 Download Heatmap Data (CSV)",
                    data=heatmap_csv,
                    file_name="heatmap_data.csv",
                    mime="text/csv",
                    key="download_heatmap"
                )
            
            elif viz_type == "Distribution Plot":
                st.markdown("#### Expression Distribution")
                dist_fig = create_distribution_plot(expr_data)
                st.plotly_chart(dist_fig, use_container_width=True, key="dist_plot_tab4")
                
                # Download distribution data
                expr_values = expr_data.values.flatten()
                expr_values_log = np.log2(expr_values + 1)
                dist_df = pd.DataFrame({
                    'log2_expression': expr_values_log,
                    'original_expression': expr_values
                })
                dist_csv = dist_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Distribution Data (CSV)",
                    data=dist_csv,
                    file_name="expression_distribution.csv",
                    mime="text/csv",
                    key="download_dist"
                )
    
    # ======================================================
    # FOOTER (always shown)
    # ======================================================
    st.markdown("---")
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown(f"© {datetime.now().year} GEO Expression Analysis Platform | Version 3.0 | Developed By : Sarah Ali ")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# RUN APPLICATION
# ======================================================
if __name__ == "__main__":
    main()