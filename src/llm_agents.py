import os
from typing import Dict, List, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from functools import lru_cache
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Prompt Templates
DATA_ANALYSIS_TEMPLATE = """Sen deneyimli bir veri bilimci ve ekonomist olarak, verilen veri setini kullanarak kapsamlı ve görsel analizler yapacaksın.

VERİ SETİ HAKKINDA:
- Toplam Ülke Sayısı: {total_countries}
- Yıl Aralığı: {year_range}
- Mevcut Metrikler: {metrics}
- Bölgeler: {regions}
- Global Mutluluk Ortalaması: {global_mean:.2f}
- En Mutlu Ülke: {happiest}
- En Mutsuz Ülke: {unhappiest}

TEMEL PRENSİPLER:

1. VERİ ANALİZİ
   - Temel istatistikler ve dağılımlar
   - Trend ve değişim analizleri
   - Bölgesel karşılaştırmalar
   - Faktör ilişkileri
   - Görsel analizler

2. İÇGÖRÜ GELİŞTİRME
   - Önemli bulguları vurgula
   - Nedensel ilişkileri açıkla
   - Karşılaştırmalı değerlendirmeler yap
   - Politika önerileri geliştir
   - Gelecek projeksiyonları sun

3. GÖRSEL SUNUM
   - Trend grafikleri [Trend: metrik]
   - Karşılaştırma grafikleri [Bar: metrik]
   - Korelasyon matrisleri [Scatter: metrik1 vs metrik2]
   - Dağılım grafikleri

4. RAPORLAMA YAPISI
   📊 Temel Bulgular
      - Önemli metrikler
      - Kritik değişimler
      - Aykırı değerler

   📈 Trend Analizi
      - Zaman serisi değişimleri
      - Büyüme oranları
      - Volatilite analizi

   🌍 Karşılaştırmalı Analiz
      - Bölgesel farklılıklar
      - Lider ve geride kalan ülkeler
      - Başarı faktörleri

   🔍 Faktör Analizi
      - Korelasyonlar
      - Nedensel ilişkiler
      - Etkileşim etkileri

   💡 Öneriler ve Projeksiyonlar
      - Politika önerileri
      - İyileştirme alanları
      - Gelecek tahminleri

Soru: {question}

Yanıtını verirken mutlaka veri setindeki gerçek değerleri kullan ve görsellerle destekle. Her sayısal değer ve trend veri setinden gelmeli."""

CAUSAL_ANALYSIS_TEMPLATE = """Sen deneyimli bir sosyal bilimci ve mutluluk araştırmacısısın.
Verilen soruyu, o spesifik durum veya ülke için analiz edeceksin.

TEMEL PRENSİPLER:

1. SORU ODAKLI YANITLAMA
   - Sorudaki spesifik ülke/bölge/duruma odaklan
   - Genel analizler yerine hedeflenen analiz yap
   - Sadece ilgili verileri kullan
   - Karşılaştırmaları sorulan bağlamda yap

2. SOMUT VERİLERLE DESTEKLEME
   - Spesifik ülke/bölgenin gerçek verilerini kullan
   - Yıllara göre değişimi göster
   - Global ortalama ile karşılaştır
   - En yakın komşu/benzer ülkelerle kıyasla

3. YAPILANDIRILMIŞ YANIT
   🎯 [ÜLKE/BÖLGE] ANALİZİ:
   - Güncel durum ve sıralama
   - Öne çıkan faktörler
   - Benzersiz özellikler

   📈 BAŞARI/BAŞARISIZLIK HİKAYESİ:
   - Kritik dönüm noktaları
   - Başarılı/başarısız politikalar
   - Toplumsal dinamikler

   💡 KARŞILAŞTIRMALI BAKIŞ:
   - Benzer ülkelerle kıyaslama
   - Bölgesel konum
   - İyi/kötü örnekler

GÖRSELLEŞTİRME:
Sadece şu durumlarda görsel kullan:
- Zaman içindeki önemli değişimleri göstermek için
- Benzer ülkelerle karşılaştırma yapmak için

Görsel isteklerini şu formatta yap:
[görsel X: <tip> <ülke/bölge> <metrik>]

Veri setindeki değişkenler:
{variables}

Soru: {question}

Not: 
- Yanıtın soru odaklı ve spesifik olmalı
- Her iddia veriyle desteklenmeli
- Maksimum 2 görsel kullanmalısın
- Hikayeleştirerek anlat"""

GENERAL_QA_TEMPLATE = """Sen deneyimli bir veri bilimci, ekonomist ve mutluluk araştırmacısısın.
Verilen veri setini kullanarak soruları detaylı ve anlamlı bir şekilde yanıtlayacaksın.
Analizlerini görsellerle destekleyebilirsin.

TEMEL PRENSİPLER:

1. VERİ ODAKLI YAKLAŞIM
   - Analizlerini veri setindeki gerçek verilere dayandır
   - Önemli sayısal bulguları vurgula
   - İstatistiksel analizler ve karşılaştırmalar yap
   - Anlamlı trendleri ve kalıpları belirle
   - Bulgularını görsellerle destekle

2. BÜTÜNCÜL DEĞERLENDİRME
   - Çoklu faktörleri incele
   - Farklı açılardan karşılaştırmalar yap
   - Zaman içindeki değişimleri analiz et
   - Bölgesel ve global bağlamı değerlendir
   - Karşılaştırmalı grafikler kullan

3. ANLAMLI İÇGÖRÜLER
   - Verilerden derin çıkarımlar yap
   - İlginç bulguları vurgula
   - Beklenmedik sonuçları açıkla
   - Önemli ilişkileri belirt
   - Görsel içgörüler sun

4. UZMAN YORUMLARI
   - Veri analizine dayanarak uzman görüşlerini ekle
   - Olası nedenleri ve sonuçları tartış
   - Gelecek projeksiyonları yap
   - Politika önerileri geliştir
   - Tahminlerini görselleştir

GÖRSELLEŞTİRME SEÇENEKLERİ:
📈 Trend Grafikleri
   - Zaman serisi analizleri
   - Büyüme eğrileri
   - Karşılaştırmalı trendler

📊 Karşılaştırma Grafikleri
   - Bar grafikleri
   - Kutu grafikleri
   - Radar grafikleri

🗺️ Coğrafi Görselleştirmeler
   - Bölgesel karşılaştırmalar
   - Küresel dağılımlar
   - Sıcaklık haritaları

📉 İlişki Grafikleri
   - Saçılım grafikleri
   - Korelasyon matrisleri
   - Ağaç haritaları

YANITLARKEN:
✓ Sorunun bağlamına uygun analiz yap
✓ Önemli bulguları vurgula
✓ Analizlerini görsellerle destekle
✓ Uzman yorumlarını ekle
✓ Gelecek öngörülerinde bulun

Soru: {question}
"""

class AgentType:
    DATA = "data"
    CAUSAL = "causal"
    QA = "qa"

class MultiAgentSystem:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = self._create_llm()
        self.agents = self._create_agents()
        
        # Streamlit session state'i başlat
        if 'last_question' not in st.session_state:
            st.session_state.last_question = None
            st.session_state.last_response = None
            st.session_state.response_count = 0
        
        # CSS stillerini uygula
        st.markdown("""
            <style>
                .stMarkdown {
                    text-align: left !important;
                }
                .stMarkdown p {
                    text-align: left !important;
                    margin-left: 0 !important;
                }
                .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                    text-align: left !important;
                    margin-left: 0 !important;
                }
                .stMarkdown ul, .stMarkdown ol {
                    text-align: left !important;
                    margin-left: 1em !important;
                    padding-left: 0 !important;
                }
                .stMarkdown li {
                    text-align: left !important;
                    margin-left: 0 !important;
                }
                div[data-testid="stMarkdownContainer"] {
                    text-align: left !important;
                }
                .element-container {
                    margin-left: 0 !important;
                }
            </style>
        """, unsafe_allow_html=True)
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Base LLM oluştur ve optimize et"""
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=False,
            max_output_tokens=2048,
            top_p=0.95,
            top_k=40,
            timeout=120,  # Timeout süresini artır
            retry_max_attempts=3,  # Yeniden deneme sayısını sınırla
            retry_min_wait=1,  # Minimum bekleme süresini azalt
            cache=False,  # Önbelleği devre dışı bırak
        )
    
    def _create_agents(self) -> Dict[str, Any]:
        """Tüm agent'ları oluştur"""
        return {
            AgentType.DATA: self._create_data_agent(),
            AgentType.CAUSAL: self._create_causal_agent(),
            AgentType.QA: self._create_qa_agent()
        }
    
    def _calculate_basic_stats(self, data: pd.Series) -> Dict[str, float]:
        """Temel istatistikleri hesapla"""
        stats = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
        
        # Varyasyon katsayısı
        stats['cv'] = (stats['std'] / stats['mean']) * 100 if stats['mean'] != 0 else 0
        
        # Z-skorları hesapla
        z_scores = (data - stats['mean']) / stats['std']
        stats['outliers'] = data[abs(z_scores) > 2].to_dict()
        
        # Yüzdelik dilimler
        stats['percentiles'] = {
            'p5': data.quantile(0.05),
            'p10': data.quantile(0.10),
            'p25': data.quantile(0.25),
            'p75': data.quantile(0.75),
            'p90': data.quantile(0.90),
            'p95': data.quantile(0.95)
        }
        
        # Dağılım özellikleri
        stats['distribution'] = {
            'normality': 'simetrik' if abs(stats['skewness']) < 0.5 else ('sağa çarpık' if stats['skewness'] > 0 else 'sola çarpık'),
            'peakedness': 'normal' if abs(stats['kurtosis']) < 0.5 else ('sivri' if stats['kurtosis'] > 0 else 'basık')
        }
        
        return stats
    
    def _calculate_trend_analysis(self, data: pd.DataFrame, metric: str) -> Dict[str, Any]:
        """Trend analizi yap"""
        yearly_data = data.groupby('year')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Yıllık değişim oranları
        yearly_data['change'] = yearly_data['mean'].pct_change()
        
        # CAGR hesapla
        years = len(yearly_data)
        if years > 1:
            cagr = ((yearly_data['mean'].iloc[-1] / yearly_data['mean'].iloc[0]) ** (1/(years-1)) - 1) * 100
        else:
            cagr = 0
            
        # Volatilite analizi
        volatility = {
            'std_dev': yearly_data['change'].std() * 100,
            'max_drawdown': yearly_data['change'].min() * 100,
            'max_increase': yearly_data['change'].max() * 100
        }
        
        # Trend analizi
        X = np.arange(len(yearly_data)).reshape(-1, 1)
        y = yearly_data['mean'].values
        trend_model = LinearRegression().fit(X, y)
        trend_direction = 'artış' if trend_model.coef_[0] > 0 else 'düşüş'
        trend_strength = trend_model.score(X, y)  # R-kare
        
        return {
            'yearly_stats': yearly_data.to_dict('records'),
            'cagr': cagr,
            'volatility': volatility,
            'trend': {
                'direction': trend_direction,
                'strength': trend_strength,
                'slope': float(trend_model.coef_[0])
            }
        }
    
    def _calculate_comparative_analysis(self, data: pd.DataFrame, metric: str, year: int = None) -> Dict[str, Any]:
        """Karşılaştırmalı analiz yap"""
        if year is None:
            year = data['year'].max()
            
        year_data = data[data['year'] == year]
        
        # Bölgesel analiz
        regional_stats = year_data.groupby('regional_indicator')[metric].agg([
            'mean', 'std', 'count', 'min', 'max', 'median',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ]).round(3)
        regional_stats.columns = ['mean', 'std', 'count', 'min', 'max', 'median', 'q1', 'q3']
        
        # Z-skorları hesapla
        regional_stats['z_score'] = (regional_stats['mean'] - regional_stats['mean'].mean()) / regional_stats['mean'].std()
        
        # Bölgesel karşılaştırmalar
        global_mean = year_data[metric].mean()
        regional_stats['vs_global'] = ((regional_stats['mean'] - global_mean) / global_mean * 100).round(2)
        
        # Bölgesel eşitsizlik göstergeleri
        regional_stats['gini'] = year_data.groupby('regional_indicator').apply(lambda x: (x[metric].sort_values().reset_index(drop=True) * pd.Series(range(len(x)), index=x[metric].sort_values().index)).sum() / (len(x) * x[metric].sum()) - (len(x) + 1) / (2 * len(x)))
        
        # En iyi/kötü ülkeler (bölgesel)
        top_by_region = {}
        bottom_by_region = {}
        trends_by_region = {}
        
        for region in regional_stats.index:
            region_data = year_data[year_data['regional_indicator'] == region]
            top_by_region[region] = region_data.nlargest(3, metric)[['country_name', metric]].to_dict('records')
            bottom_by_region[region] = region_data.nsmallest(3, metric)[['country_name', metric]].to_dict('records')
            
            # Bölgesel trendler
            region_trend = data[data['regional_indicator'] == region].groupby('year')[metric].mean()
            trends_by_region[region] = {
                'growth_rate': ((region_trend.iloc[-1] / region_trend.iloc[0]) - 1) * 100,
                'volatility': region_trend.std() / region_trend.mean() * 100
            }
        
        return {
            'regional_stats': regional_stats.to_dict('index'),
            'global_stats': {
                'mean': global_mean,
                'std': year_data[metric].std(),
                'cv': (year_data[metric].std() / global_mean * 100),
                'range': year_data[metric].max() - year_data[metric].min()
            },
            'top_by_region': top_by_region,
            'bottom_by_region': bottom_by_region,
            'trends_by_region': trends_by_region,
            'outliers': year_data[abs((year_data[metric] - global_mean) / year_data[metric].std()) > 2][
                ['country_name', metric, 'regional_indicator']
            ].to_dict('records')
        }
    
    def _calculate_factor_analysis(self, data: pd.DataFrame, target: str = 'life_ladder') -> Dict[str, Any]:
        """Faktör analizi yap"""
        # Faktör listesi
        factors = ['social_support', 'freedom_to_make_life_choices', 'generosity', 
                  'perceptions_of_corruption', 'gdp_per_capita', 'life_expectancy']
        
        # Korelasyon analizleri
        correlations = {}
        for factor in factors:
            if factor in data.columns:
                # Pearson korelasyonu
                pearson_corr = data[factor].corr(data[target])
                # Spearman korelasyonu
                spearman_corr = data[factor].corr(data[target], method='spearman')
                
                correlations[factor] = {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr
                }
        
        # Çoklu regresyon
        valid_factors = [f for f in factors if f in data.columns]
        if valid_factors:
            X = data[valid_factors]
            y = data[target]
            model = LinearRegression()
            model.fit(X, y)
            
            regression_results = {
                'r2': model.score(X, y),
                'coefficients': dict(zip(valid_factors, model.coef_)),
                'importance': dict(zip(
                    valid_factors,
                    np.abs(model.coef_) * np.std(X, axis=0)
                ))
            }
        else:
            regression_results = None
        
        # Faktör etkileşimleri
        interactions = {}
        for i, f1 in enumerate(valid_factors):
            for f2 in valid_factors[i+1:]:
                interaction_term = data[f1] * data[f2]
                corr_with_target = interaction_term.corr(data[target])
                interactions[f"{f1} x {f2}"] = corr_with_target
        
        return {
            'correlations': correlations,
            'regression': regression_results,
            'interactions': dict(sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True))
        }
    
    def _create_visualizations(self, data: pd.DataFrame, analysis_type: str, metric: str = None) -> Dict[str, Any]:
        """Analiz tipine göre görselleştirmeler oluştur"""
        visuals = {}
        
        if analysis_type == "trend":
            # MENA bölgesi için trend grafiği
            if metric == 'life_ladder':
                mena_data = data[data['regional_indicator'] == 'Middle East and North Africa']
                yearly_mena = mena_data.groupby('year')['life_ladder'].agg(['mean', 'std']).reset_index()
                
                fig = go.Figure()
                
                # MENA bölgesi trend çizgisi
                fig.add_trace(go.Scatter(
                    x=yearly_mena['year'],
                    y=yearly_mena['mean'],
                    mode='lines+markers',
                    name='MENA Bölgesi',
                    line=dict(color='#8dd3c7', width=3),
                    marker=dict(size=8)
                ))
                
                # Standart sapma aralığı
                fig.add_trace(go.Scatter(
                    x=yearly_mena['year'],
                    y=yearly_mena['mean'] + yearly_mena['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    name='Üst Sınır'
                ))
                
                fig.add_trace(go.Scatter(
                    x=yearly_mena['year'],
                    y=yearly_mena['mean'] - yearly_mena['std'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(141, 211, 199, 0.3)',
                    fill='tonexty',
                    showlegend=False,
                    name='Alt Sınır'
                ))
                
                # Global ortalama trend çizgisi
                yearly_global = data.groupby('year')['life_ladder'].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=yearly_global['year'],
                    y=yearly_global['life_ladder'],
                    mode='lines',
                    name='Global Ortalama',
                    line=dict(color='#bebada', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='MENA Bölgesi Mutluluk Trendi',
                    xaxis_title='Yıl',
                    yaxis_title='Mutluluk Skoru',
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0.5)'
                    )
                )
                
                visuals['trend'] = fig
            else:
                # Diğer metrikler için genel trend grafiği
                yearly_data = data.groupby('year')[metric].mean().reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_data['year'],
                    y=yearly_data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color='#8dd3c7', width=3)
                ))
                
                fig.update_layout(
                    title=f"{metric.replace('_', ' ').title()} Trend Analysis",
                    xaxis_title="Year",
                    yaxis_title=metric.replace('_', ' ').title(),
                    template="plotly_dark"
                )
                
                visuals['trend'] = fig
            
        elif analysis_type == "comparison":
            # Bölgesel karşılaştırma grafiği
            latest_year = data['year'].max()
            latest_data = data[data['year'] == latest_year]
            
            fig = px.box(
                latest_data,
                x='regional_indicator',
                y=metric,
                title=f"Regional Comparison of {metric.replace('_', ' ').title()}",
                template="plotly_dark"
            )
            
            visuals['comparison'] = fig
            
        elif analysis_type == "correlation":
            # Korelasyon matrisi
            factors = ['life_ladder', 'gdp_per_capita', 'social_support', 
                      'freedom_to_make_life_choices', 'generosity', 
                      'perceptions_of_corruption']
            
            corr_matrix = data[factors].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=[f.replace('_', ' ').title() for f in factors],
                y=[f.replace('_', ' ').title() for f in factors],
                title="Correlation Matrix",
                template="plotly_dark"
            )
            
            visuals['correlation'] = fig
            
        elif analysis_type == "causal":
            # Nedensel ağ grafiği
            factors = ['gdp_per_capita', 'social_support', 'freedom_to_make_life_choices',
                      'generosity', 'perceptions_of_corruption']
            
            # Faktörler arası korelasyonları hesapla
            correlations = []
            for i, f1 in enumerate(factors):
                for f2 in factors[i+1:]:
                    corr = data[f1].corr(data[f2])
                    if abs(corr) > 0.3:  # Sadece önemli korelasyonları göster
                        correlations.append((f1, f2, corr))
            
            # Nedensel ağ grafiği oluştur
            fig = go.Figure()
            
            # Düğümleri ekle
            for i, factor in enumerate(factors):
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[0],
                    mode='markers+text',
                    name=factor,
                    text=[factor.replace('_', ' ').title()],
                    marker=dict(size=20),
                    textposition="bottom center"
                ))
            
            # Bağlantıları ekle
            for f1, f2, corr in correlations:
                i1 = factors.index(f1)
                i2 = factors.index(f2)
                
                fig.add_trace(go.Scatter(
                    x=[i1, i2],
                    y=[0, 0],
                    mode='lines',
                    line=dict(
                        width=abs(corr) * 5,
                        color='red' if corr < 0 else 'green'
                    ),
                    name=f"{f1} - {f2} ({corr:.2f})"
                ))
            
            fig.update_layout(
                title="Causal Network Graph",
                showlegend=False,
                template="plotly_dark"
            )
            
            visuals['causal'] = fig
        
        return visuals
    
    def _format_analysis_results(self, results: Dict[str, Any]) -> str:
        """Analiz sonuçlarını formatla ve görselleştir"""
        output = []
        
        # Temel Analiz
        output.append("## 📊 Temel Bulgular")
        stats = results.get('basic_stats', {})
        if stats:
            output.append("\n### 📈 İstatistiksel Özet")
            output.append(f"- **Ortalama:** {stats.get('mean', 0):.2f}")
            output.append(f"- **Medyan:** {stats.get('median', 0):.2f}")
            output.append(f"- **Standart Sapma:** {stats.get('std', 0):.2f}")
            output.append(f"- **Değişim Aralığı:** {stats.get('min', 0):.2f} - {stats.get('max', 0):.2f}")
            
            # Dağılım özellikleri
            dist = stats.get('distribution', {})
            if dist:
                output.append("\n### 📊 Dağılım Analizi")
                output.append(f"- **Dağılım Tipi:** {dist.get('normality', 'bilinmiyor')}")
                output.append(f"- **Tepe Noktası:** {dist.get('peakedness', 'bilinmiyor')}")
                output.append(f"- **Değişkenlik Katsayısı:** %{stats.get('cv', 0):.1f}")
        
        # Bölgesel Analiz
        comp_analysis = results.get('comparative_analysis', {})
        if comp_analysis:
            output.append("\n## 🌍 Bölgesel Karşılaştırma")
            
            # Global istatistikler
            global_stats = comp_analysis.get('global_stats', {})
            output.append("\n### 🌐 Global Durum")
            output.append(f"- **Global Ortalama:** {global_stats.get('mean', 0):.2f}")
            output.append(f"- **Global Değişkenlik:** %{global_stats.get('cv', 0):.1f}")
            
            # Bölgesel sıralama
            regional_stats = comp_analysis.get('regional_stats', {})
            if regional_stats:
                output.append("\n### 📊 Bölgesel Sıralama")
                sorted_regions = sorted(
                    regional_stats.items(),
                    key=lambda x: x[1]['mean'],
                    reverse=True
                )
                
                # Bölgesel karşılaştırma grafiği
                fig = go.Figure()
                
                regions = [region for region, _ in sorted_regions]
                means = [stats['mean'] for _, stats in sorted_regions]
                stds = [stats['std'] for _, stats in sorted_regions]
                
                # Bar grafiği
                fig.add_trace(go.Bar(
                    x=regions,
                    y=means,
                    error_y=dict(type='data', array=stds),
                    name='Ortalama Değer',
                    marker_color='#8dd3c7'
                ))
                
                fig.update_layout(
                    title='Bölgelere Göre Karşılaştırmalı Analiz',
                    xaxis_title="Bölge",
                    yaxis_title="Değer",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(tickangle=-45),
                    showlegend=False,
                    height=500
                )
                
                # Grafiği doğrudan göster
                st.plotly_chart(fig, use_container_width=True)
                
                # Metin olarak da göster
                for region, stats in sorted_regions[:3]:
                    output.append(f"\n**{region}**")
                    output.append(f"- Ortalama: {stats['mean']:.2f}")
                    output.append(f"- Değişkenlik: %{stats['std']/stats['mean']*100:.1f}")
                    output.append(f"- Gini Katsayısı: {stats.get('gini', 0):.3f}")
        
        # Trend Analizi
        trends = results.get('trends_by_region', {})
        if trends:
            output.append("\n## 📈 Trend Analizi")
            
            # En yüksek büyüme gösteren bölgeler
            sorted_growth = sorted(
                [(region, data['growth_rate']) for region, data in trends.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            output.append("\n### 🚀 Büyüme Performansı")
            for region, growth in sorted_growth[:3]:
                output.append(f"- **{region}:** %{growth:.1f} büyüme")
        
        # Görselleştirmeleri ekle
        output.append("\n## 📊 Görsel Analizler")
        
        # Trend grafiği
        visuals = self._create_visualizations(self.df, "trend", results.get('metric'))
        if visuals.get('trend'):
            output.append("\n### 📈 Trend Analizi")
            st.plotly_chart(visuals['trend'], use_container_width=True)
        
        # Bölgesel karşılaştırma
        visuals = self._create_visualizations(self.df, "comparison", results.get('metric'))
        if visuals.get('comparison'):
            output.append("\n### 📊 Bölgesel Karşılaştırma")
            st.plotly_chart(visuals['comparison'], use_container_width=True)
        
        # Korelasyon matrisi
        visuals = self._create_visualizations(self.df, "correlation")
        if visuals.get('correlation'):
            output.append("\n### 🔍 Korelasyon Analizi")
            st.plotly_chart(visuals['correlation'], use_container_width=True)
        
        # Nedensel ağ grafiği
        if results.get('agent_type') == AgentType.CAUSAL:
            visuals = self._create_visualizations(self.df, "causal")
            if visuals.get('causal'):
                output.append("\n### 🔄 Nedensel İlişkiler")
                st.plotly_chart(visuals['causal'], use_container_width=True)
        
        # İçgörüler
        if results.get('insights'):
            output.append("\n## 🔍 Önemli İçgörüler")
            for i, insight in enumerate(results['insights'], 1):
                output.append(f"\n{i}. {insight}")
        
        # Metodoloji
        if results.get('methodology'):
            output.append("\n## ℹ️ Metodoloji Notları")
            for note in results['methodology']:
                output.append(f"- {note}")
        
        return "\n".join(output)
    
    def _create_data_agent(self) -> LLMChain:
        """Veri analizi agent'ı"""
        # Veri setinden temel istatistikleri hesapla
        latest_year = self.df['year'].max()
        latest_data = self.df[self.df['year'] == latest_year]
        
        # Veri seti özeti oluştur
        data_summary = {
            'total_countries': len(latest_data['country_name'].unique()),
            'year_range': f"{self.df['year'].min()} - {self.df['year'].max()}",
            'available_metrics': list(self.df.columns),
            'regions': list(latest_data['regional_indicator'].unique()),
            'global_happiness_mean': latest_data['life_ladder'].mean(),
            'happiest_country': latest_data.nlargest(1, 'life_ladder')['country_name'].iloc[0],
            'unhappiest_country': latest_data.nsmallest(1, 'life_ladder')['country_name'].iloc[0]
        }
        
        # Prompt template'i güncelle
        template = """Sen deneyimli bir veri bilimci ve ekonomist olarak, verilen veri setini kullanarak kapsamlı ve görsel analizler yapacaksın.

VERİ SETİ HAKKINDA:
- Toplam Ülke Sayısı: {total_countries}
- Yıl Aralığı: {year_range}
- Mevcut Metrikler: {metrics}
- Bölgeler: {regions}
- Global Mutluluk Ortalaması: {global_mean:.2f}
- En Mutlu Ülke: {happiest}
- En Mutsuz Ülke: {unhappiest}

TEMEL PRENSİPLER:

1. VERİ ANALİZİ
   - Temel istatistikler ve dağılımlar
   - Trend ve değişim analizleri
   - Bölgesel karşılaştırmalar
   - Faktör ilişkileri
   - Görsel analizler

2. İÇGÖRÜ GELİŞTİRME
   - Önemli bulguları vurgula
   - Nedensel ilişkileri açıkla
   - Karşılaştırmalı değerlendirmeler yap
   - Politika önerileri geliştir
   - Gelecek projeksiyonları sun

3. GÖRSEL SUNUM
   - Trend grafikleri [Trend: metrik]
   - Karşılaştırma grafikleri [Bar: metrik]
   - Korelasyon matrisleri [Scatter: metrik1 vs metrik2]
   - Dağılım grafikleri

4. RAPORLAMA YAPISI
   📊 Temel Bulgular
      - Önemli metrikler
      - Kritik değişimler
      - Aykırı değerler

   📈 Trend Analizi
      - Zaman serisi değişimleri
      - Büyüme oranları
      - Volatilite analizi

   🌍 Karşılaştırmalı Analiz
      - Bölgesel farklılıklar
      - Lider ve geride kalan ülkeler
      - Başarı faktörleri

   🔍 Faktör Analizi
      - Korelasyonlar
      - Nedensel ilişkiler
      - Etkileşim etkileri

   💡 Öneriler ve Projeksiyonlar
      - Politika önerileri
      - İyileştirme alanları
      - Gelecek tahminleri

Soru: {question}

Yanıtını verirken mutlaka veri setindeki gerçek değerleri kullan ve görsellerle destekle. Her sayısal değer ve trend veri setinden gelmeli."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
            partial_variables={
                "total_countries": str(data_summary['total_countries']),
                "year_range": data_summary['year_range'],
                "metrics": ", ".join(data_summary['available_metrics']),
                "regions": ", ".join(data_summary['regions']),
                "global_mean": data_summary['global_happiness_mean'],
                "happiest": data_summary['happiest_country'],
                "unhappiest": data_summary['unhappiest_country']
            }
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _create_causal_agent(self) -> LLMChain:
        """Nedensel analiz agent'ı"""
        variables = list(self.df.columns)
        
        # Sistem mesajını ve kullanıcı template'ini ayır
        system_message = """Sen bir sosyal bilimci ve mutluluk araştırmacısısın. 
Görevin, spesifik ülke veya bölgelerin mutluluk durumunu analiz etmek.
Yanıtların MUTLAKA şu yapıda olmalı:

1. MEVCUT DURUM
2. BAŞARI/BAŞARISIZLIK NEDENLERİ
3. ÖNEMLİ DETAYLAR

Genel istatistikler veya korelasyonlar yerine, 
sorulan ülke/bölgeye özel veriler ve analizler sunmalısın."""
        
        # Daha katı bir template yapısı
        template = """YANITINI KESİNLİKLE BU FORMATTA VER:

🎯 MEVCUT DURUM:
- {ülke/bölge} güncel mutluluk skoru: [DEĞER]
- Dünya sıralamasındaki yeri: [SIRA]
- Öne çıkan faktörler: [FAKTÖRLER]

📈 BAŞARI/BAŞARISIZLIK NEDENLERİ:
1. [İLK FAKTÖR]
   - Veri: [DEĞER]
   - Karşılaştırma: [KARŞILAŞTIRMA]

2. [İKİNCİ FAKTÖR]
   - Veri: [DEĞER]
   - Karşılaştırma: [KARŞILAŞTIRMA]

💡 ÖNEMLİ DETAYLAR:
- [DETAY 1]
- [DETAY 2]
- [DETAY 3]

Veri setindeki değişkenler: {variables}
Soru: {question}"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "variables"]
        )
        
        # LLM'i daha katı ayarlarla yapılandır
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,  # Yaratıcılığı minimumda tut
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,  # Sistem mesajını insan mesajına çevir
            max_output_tokens=2048,
            top_p=0.1,  # Daha deterministik yanıtlar
            top_k=1,    # En olası yanıtı seç
        )
        
        return LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True  # Debug için
        )
    
    def _create_qa_agent(self) -> LLMChain:
        """Genel soru-cevap agent'ı"""
        prompt = PromptTemplate(
            template=GENERAL_QA_TEMPLATE,
            input_variables=["question"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def route_question(self, question: str) -> str:
        """Soruyu uygun agent'a yönlendir"""
        # Küçük harfe çevir ve noktalama işaretlerini kaldır
        question_lower = ''.join(c.lower() for c in question if c.isalnum() or c.isspace())
        words = question_lower.split()
        
        # Nedensel analiz için spesifik kalıplar
        causal_patterns = [
            "neden bu kadar",
            "niye bu kadar",
            "niçin bu kadar",
            "neden bu denli",
            "neden böyle",
            "niye böyle",
            "nasıl bu kadar",
            "sebebi ne",
            "nedeni ne"
        ]
        
        # Veri analizi için spesifik kalıplar
        data_patterns = [
            "kaç tane",
            "ne kadar",
            "hangi ülke",
            "hangi bölge",
            "en mutlu",
            "en mutsuz",
            "karşılaştır",
            "kıyasla",
            "sırala",
            "göster",
            "trend",
            "değişim",
            "analiz",
            "fark",
            "arasında"
        ]
        
        # Önce nedensel analiz kalıplarını kontrol et
        for pattern in causal_patterns:
            if pattern in question_lower:
                return AgentType.CAUSAL
        
        # Sonra veri analizi kalıplarını kontrol et
        for pattern in data_patterns:
            if pattern in question_lower:
                return AgentType.DATA
        
        # Eğer hiçbir spesifik kalıp eşleşmezse, QA'ya yönlendir
        return AgentType.QA

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def _get_llm_response(self, agent: Any, **kwargs) -> str:
        """LLM yanıtlarını yönet ve görselleştirmeleri işle"""
        try:
            # LLM yanıtını al
            response = await agent.ainvoke(kwargs)
            
            # Yanıt kontrolü
            if not response or 'text' not in response:
                raise ValueError("LLM'den geçerli bir yanıt alınamadı")
                
            response_text = response['text']
            
            if not response_text or not isinstance(response_text, str):
                raise ValueError("LLM yanıtı boş veya geçersiz format")

            # Format kontrolü (Causal agent için)
            if isinstance(agent, LLMChain) and 'CAUSAL_ANALYSIS_TEMPLATE' in str(agent.prompt):
                # Ülke/bölge adını çıkar
                country_match = re.search(r'🎯\s*\[([^\]]+)\]', response_text)
                country_name = country_match.group(1) if country_match else "İlgili Ülke/Bölge"
                
                # Yanıtı yeniden formatla
                if not any(section in response_text for section in ['ANALİZİ:', 'HİKAYESİ:', 'BAKIŞ:']):
                    formatted_response = f"🎯 [{country_name}] ANALİZİ:\n"
                    formatted_response += "- Güncel durum analizi yapılıyor...\n\n"
                    formatted_response += "📈 BAŞARI/BAŞARISIZLIK HİKAYESİ:\n"
                    formatted_response += response_text + "\n\n"
                    formatted_response += "💡 KARŞILAŞTIRMALI BAKIŞ:\n"
                    formatted_response += "- Karşılaştırmalı analiz devam ediyor...\n"
                    
                    response_text = formatted_response

            # Görsel referanslarını işle
            try:
                processed_response = self._process_visualizations(response_text)
            except Exception as e:
                st.warning(f"Görselleştirmeler işlenirken bazı hatalar oluştu. Analiz devam ediyor.")
                processed_response = response_text
            
            return processed_response

        except Exception as e:
            st.error(f"LLM yanıtı alınırken hata: {str(e)}")
            return "Yanıt alınamadı. Lütfen tekrar deneyin."

    def _process_visualizations(self, response: str) -> str:
        """LLM yanıtındaki görsel referanslarını gerçek grafiklere dönüştür"""
        try:
            # Yanıt kontrolü
            if not response or not isinstance(response, str):
                return response

            # Köşeli parantez içindeki görsel referanslarını bul
            visual_refs = re.finditer(r'\[([^]]+)\]', response)
            
            for match in visual_refs:
                try:
                    visual_desc = match.group(1).strip().lower()
                    
                    # Görsel tipi ve metrik için varsayılan değerler
                    visual_type = None
                    metric = 'life_ladder'
                    metric2 = None
                    countries = []
                    
                    # Görsel tipini belirle
                    if any(keyword in visual_desc for keyword in ['trend', 'değişim', 'zaman']):
                        visual_type = 'trend'
                    elif any(keyword in visual_desc for keyword in ['bar', 'karşılaştırma', 'sütun']):
                        visual_type = 'bar'
                    elif any(keyword in visual_desc for keyword in ['saçılım', 'korelasyon', 'ilişki']):
                        visual_type = 'scatter'
                    
                    # Metrikleri belirle
                    metric_mapping = {
                        'gsyih': 'gdp_per_capita',
                        'gdp': 'gdp_per_capita',
                        'mutluluk': 'life_ladder',
                        'puan': 'life_ladder',
                        'skor': 'life_ladder',
                        'sosyal destek': 'social_support',
                        'özgürlük': 'freedom_to_make_life_choices',
                        'yolsuzluk': 'perceptions_of_corruption',
                        'cömertlik': 'generosity',
                        'yaşam beklentisi': 'life_expectancy',
                        'internet': 'internet_users_percent'
                    }
                    
                    # Metrikleri tespit et
                    for keyword, col in metric_mapping.items():
                        if keyword in visual_desc:
                            if metric == 'life_ladder':
                                metric = col
                            else:
                                metric2 = col
                                break
                    
                    # Ülkeleri belirle
                    country_mapping = {
                        'türkiye': 'Turkiye',
                        'kolombiya': 'Colombia',
                        'amerika': 'United States',
                        'abd': 'United States'
                    }
                    
                    for keyword, country in country_mapping.items():
                        if keyword in visual_desc:
                            countries.append(country)
                    
                    # Veri kontrolü
                    if not self.df[self.df['country_name'].isin(countries)].empty:
                        latest_year = self.df['year'].max()
                        
                        if visual_type == 'scatter':
                            latest_data = self.df[self.df['year'] == latest_year].copy()
                            
                            # Metrik kontrolü
                            if metric not in latest_data.columns or (metric2 and metric2 not in latest_data.columns):
                                st.warning(f"Bazı metrikler veri setinde bulunamadı: {metric}, {metric2}")
                                continue
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=latest_data[metric],
                                y=latest_data[metric2] if metric2 else latest_data['life_ladder'],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=latest_data['regional_indicator'].astype('category').cat.codes,
                                    colorscale='Viridis',
                                    showscale=True
                                ),
                                text=latest_data['country_name'],
                                hovertemplate='<b>%{text}</b><br>' +
                                            f'{metric}: %{{x}}<br>' +
                                            f'{metric2 if metric2 else "life_ladder"}: %{{y}}<br>'
                            ))
                            
                            fig.update_layout(
                                title=f"{metric.replace('_', ' ').title()} ve {(metric2 if metric2 else 'Mutluluk')} İlişkisi",
                                xaxis_title=metric.replace('_', ' ').title(),
                                yaxis_title=metric2.replace('_', ' ').title() if metric2 else 'Mutluluk Skoru',
                                template='plotly_dark',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif visual_type == 'trend':
                            if countries:
                                country_data = self.df[self.df['country_name'].isin(countries)]
                            else:
                                # Eğer ülke belirtilmemişse, global trend göster
                                country_data = self.df.groupby('year')[metric].mean().reset_index()
                                countries = ['Global Ortalama']
                            
                            if not country_data.empty:
                                fig = go.Figure()
                                
                                if len(countries) == 1 and countries[0] == 'Global Ortalama':
                                    fig.add_trace(go.Scatter(
                                        x=country_data['year'],
                                        y=country_data[metric],
                                        mode='lines+markers',
                                        name='Global Ortalama',
                                        line=dict(width=3),
                                        marker=dict(size=8)
                                    ))
                                
                                fig.update_layout(
                                    title=f"{''.join(countries)} {metric.replace('_', ' ').title()} Trendi",
                                    xaxis_title='Yıl',
                                    yaxis_title=metric.replace('_', ' ').title(),
                                    template='plotly_dark',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                        elif visual_type == 'bar':
                            latest_data = self.df[self.df['year'] == latest_year].copy()
                            
                            if not latest_data.empty:
                                # Top 10 ülkeyi seç
                                sorted_data = latest_data.nlargest(10, metric)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=sorted_data['country_name'],
                                    y=sorted_data[metric],
                                    marker_color=['#8dd3c7' if x in countries else '#bebada' for x in sorted_data['country_name']],
                                    text=sorted_data[metric].round(2),
                                    textposition='auto'
                                ))
                                
                                fig.update_layout(
                                    title=f"En Yüksek {metric.replace('_', ' ').title()} Değerine Sahip 10 Ülke",
                                    xaxis_title='Ülkeler',
                                    yaxis_title=metric.replace('_', ' ').title(),
                                    template='plotly_dark',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(tickangle=-45)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Bir görselleştirme işlenirken hata oluştu: {str(e)}")
                    continue
            
            return response

        except Exception as e:
            st.error(f"Görselleştirmeler işlenirken hata: {str(e)}")
            return response

    @st.cache_data(ttl=3600)  # 1 saat önbellek
    def _process_cached_response(self, response: str) -> Dict[str, Any]:
        """Yanıtları işle ve önbellekle"""
        try:
            # Yanıt işleme mantığı
            processed_response = {
                'text': response,
                'timestamp': pd.Timestamp.now()
            }
            return processed_response
        except Exception as e:
            st.error(f"Yanıt işlenirken hata: {str(e)}")
            return None

    async def get_answer(self, question: str) -> str:
        """
        Verilen soruyu uygun agent'a yönlendirir ve yanıt alır.
        """
        try:
            # Soruyu uygun agent'a yönlendir
            agent_type = self.route_question(question)
            st.info(f"Soru {agent_type} agent'ına yönlendirildi...")
            
            # Soru tipine göre özel değişkenleri hazırla
            kwargs = {
                'question': question
            }
            
            # Causal agent için ek değişkenler
            if agent_type == AgentType.CAUSAL:
                kwargs['variables'] = list(self.df.columns)
            
            # Data agent için ek değişkenler
            elif agent_type == AgentType.DATA:
                metric = self._determine_metric(question)
                if metric:
                    kwargs['metric'] = metric
            
            # LLM yanıtını al
            response = await self._get_llm_response(
                self.agents[agent_type],
                **kwargs
            )
            
            # Yanıt kontrolü
            if not response:
                return ""
            
            # Yanıtı string'e çevir
            if not isinstance(response, str):
                response = str(response)
            
            return response
            
        except Exception as e:
            st.error(f"Yanıt alınırken hata oluştu: {str(e)}")
            return ""

    def _determine_metric(self, question: str) -> Optional[str]:
        """Sorudan metrik belirle"""
        metric_keywords = {
            'sosyal destek': 'social_support',
            'özgürlük': 'freedom_to_make_life_choices',
            'yolsuzluk': 'perceptions_of_corruption',
            'cömertlik': 'generosity',
            'gdp': 'gdp_per_capita',
            'yaşam beklentisi': 'life_expectancy',
            'mutluluk': 'life_ladder',
            'skor': 'life_ladder',
            'puan': 'life_ladder',
            'doğum': 'fertility_rate',
            'doğum oranı': 'fertility_rate',
            'fertility': 'fertility_rate'
        }
        
        question_lower = question.lower()
        for keyword, col in metric_keywords.items():
            if keyword in question_lower and col in self.df.columns:
                return col
        
        return None

class ConversationManager:
    def __init__(self):
        self.conversation_history = []
        self.context = {}
    
    def add_to_history(self, question: str, answer: str, agent_type: str):
        """Soru ve cevabı geçmişe ekle"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'agent_type': agent_type,
            'timestamp': pd.Timestamp.now()
        })
    
    def get_relevant_context(self, question: str, max_history: int = 3) -> List[Dict]:
        """Soru ile ilgili geçmiş yanıtları bul"""
        # Basit kelime benzerliği kontrolü
        question_words = set(question.lower().split())
        
        relevant_history = []
        for entry in reversed(self.conversation_history):
            entry_words = set(entry['question'].lower().split())
            # Kelime kesişimi oranı
            similarity = len(question_words.intersection(entry_words)) / len(question_words)
            
            if similarity > 0.3:  # En az %30 benzerlik
                relevant_history.append(entry)
                
            if len(relevant_history) >= max_history:
                break
                
        return relevant_history 