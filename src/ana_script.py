import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import asyncio
import json
from scipy import stats

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")




@st.cache_data(ttl=3600)  # 1 saat önbellek
def load_data():
    """Veri setini yükle ve önbellekle"""
    try:
        df = pd.read_csv('cleaned_dataset.csv')
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None

def analyze_country_trend(df, country_name):
    """Ülke trend analizi yapar"""
    try:
        # Veri setinde ülkenin varlığını kontrol et
        if country_name not in df['country_name'].unique():
            st.error(f"{country_name} veri setinde bulunamadı!")
            return None, None
            
        country_data = df[df['country_name'] == country_name].sort_values('year')
        
        if len(country_data) == 0:
            st.error(f"{country_name} için veri bulunamadı!")
            return None, None
            
        # Trend grafiği
        fig = go.Figure()
        
        # Mutluluk skoru çizgisi
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['life_ladder'],
            mode='lines+markers',
            name='Mutluluk Skoru',
            line=dict(color='#8dd3c7', width=3),  # Soft turkuaz
            marker=dict(color='#8dd3c7', size=8)
        ))
        
        # Trend çizgisi
        z = np.polyfit(country_data['year'], country_data['life_ladder'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=p(country_data['year']),
            mode='lines',
            name='Trend',
            line=dict(dash='dash', color='#bebada', width=2)  # Soft mor
        ))
        
        fig.update_layout(
            title=f"{country_name} Mutluluk Trendi",
            xaxis_title="Yıl",
            yaxis_title="Mutluluk Skoru",
            showlegend=True,
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig, country_data
    except Exception as e:
        st.error(f"Veri analizi sırasında bir hata oluştu: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)  # 1 saat önbellek
def find_country(df, search_name):
    """Verilen isime en yakın ülkeyi bul ve önbellekle"""
    try:
        # Ülke isimleri sözlüğü
        country_mapping = {
            'türkiye': 'Turkiye',
            'turkey': 'Turkiye',
            'usa': 'United States',
            'united states': 'United States',
            'america': 'United States',
            'uk': 'United Kingdom',
            'britain': 'United Kingdom',
            'england': 'United Kingdom',
            'amerika': 'United States',
            'birleşik devletler': 'United States',
            'ingiltere': 'United Kingdom',
            'birleşik krallık': 'United Kingdom'
        }
        
        # Önce mapping'den kontrol et
        search_name_lower = search_name.lower()
        if search_name_lower in country_mapping:
            return country_mapping[search_name_lower]
        
        # Direkt eşleşme kontrolü
        all_countries = df['country_name'].unique()
        for country in all_countries:
            if country.lower() == search_name_lower:
                return country
        
        return None
    except Exception as e:
        st.error(f"Ülke aranırken hata oluştu: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # 1 saat önbellek
def preprocess_data(df):
    """Veri setini ön işle ve önbellekle"""
    try:
        # Ülke isimlerini standartlaştır
        country_mapping = {
            'Turkey': 'Turkiye',
            'Türkiye': 'Turkiye'
        }
        df['country_name'] = df['country_name'].replace(country_mapping)
        
        # Corruption değerlerini 0-1 arasına normalize et (eğer değilse)
        if df['perceptions_of_corruption'].max() > 1:
            df['perceptions_of_corruption'] = df['perceptions_of_corruption'] / df['perceptions_of_corruption'].max()
        
        # Yıl sütununu integer yap
        df['year'] = df['year'].astype(int)
        
        return df
    except Exception as e:
        st.error(f"Veri ön işleme sırasında hata oluştu: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # 1 saat önbellek
def prepare_common_answers(df):
    """En çok sorulan soruların cevaplarını hazırla ve önbellekle"""
    try:
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        # Faktör listesi
        factors = ['social_support', 'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption', 
                  'life_expectancy', 'education_expenditure_gdp', 'fertility_rate', 'internet_users_percent']
        
        answers = {
            # En mutlu ülkeler
            'happiest_country': latest_data.nlargest(1, 'life_ladder')['country_name'].iloc[0],
            'top_10_happiest': latest_data.nlargest(10, 'life_ladder')[['country_name', 'life_ladder']].to_dict('records'),
            
            # En mutsuz ülkeler
            'unhappiest_country': latest_data.nsmallest(1, 'life_ladder')['country_name'].iloc[0],
            'bottom_10_unhappiest': latest_data.nsmallest(10, 'life_ladder')[['country_name', 'life_ladder']].to_dict('records'),
            
            # Bölgesel ortalamalar
            'regional_averages': latest_data.groupby('regional_indicator')['life_ladder'].mean().to_dict(),
            
            # Faktör korelasyonları
            'factor_correlations': df[['life_ladder'] + factors].corr()['life_ladder'].to_dict(),
            
            # Global ortalama
            'global_average': latest_data['life_ladder'].mean(),
            
            # Yıllara göre global trend
            'yearly_trend': df.groupby('year')['life_ladder'].mean().to_dict()
        }
        
        return answers
    except Exception as e:
        st.error(f"Ortak cevaplar hazırlanırken hata oluştu: {str(e)}")
        return None

def analyze_top_n_countries(df, n, year=None):
    """En mutlu N ülkeyi analiz eder"""
    if year:
        # Belirli yıl için en mutlu N ülke
        year_data = df[df['year'] == year]
        top_n = year_data.nlargest(n, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
        
        st.write(f"### 🌟 {year} Yılının En Mutlu {n} Ülkesi")
    else:
        # Tüm yılların ortalamasına göre en mutlu N ülke
        # Önce her ülkenin ortalama mutluluk skorunu hesapla
        country_means = df.groupby('country_name')['life_ladder'].mean().reset_index()
        
        # En mutlu N ülkeyi seç
        top_n_countries = country_means.nlargest(n, 'life_ladder')['country_name'].tolist()
        
        # Bu ülkelerin verilerini al ve ortalamaları hesapla
        top_n_data = df[df['country_name'].isin(top_n_countries)].groupby(
            ['country_name', 'regional_indicator']
        ).agg({
            'life_ladder': 'mean'
        }).reset_index()
        
        # Sırala
        top_n = top_n_data.sort_values('life_ladder', ascending=False)
        
        st.write(f"### 🌟 Tüm Zamanların En Mutlu {n} Ülkesi")
        st.write("(Tüm yılların ortalamasına göre)")
    
    # Görselleştirme
    fig = px.bar(
        top_n,
        x='country_name',
        y='life_ladder',
        color='regional_indicator',
        title=f'En Mutlu {n} Ülke' + (f' ({year})' if year else ' (Tüm Yılların Ortalaması)'),
        labels={'country_name': 'Ülke', 'life_ladder': 'Mutluluk Skoru', 'regional_indicator': 'Bölge'},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel  # Soft renkler
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bölgesel analiz
    region_counts = top_n['regional_indicator'].value_counts()
    st.write("### 🌍 Bölgesel Dağılım Analizi")
    
    fig_pie = px.pie(
        values=region_counts.values,
        names=region_counts.index,
        title=f"En Mutlu {n} Ülkenin Bölgesel Dağılımı",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detaylı liste ve içgörüler
    st.write("### 📊 Detaylı Liste ve İçgörüler")
    for i, row in top_n.iterrows():
        st.write(f"{i + 1}. {row['country_name']}: {row['life_ladder']:.2f} ({row['regional_indicator']})")
    
    # Bölgesel içgörüler
    most_common_region = region_counts.index[0]
    region_percentage = (region_counts[most_common_region] / len(top_n)) * 100
    
    st.write("\n### 🔍 Önemli İçgörüler")
    st.write(f"• En mutlu {n} ülkenin {region_percentage:.1f}%'i {most_common_region} bölgesinde bulunuyor.")
    
    # Global karşılaştırma
    global_avg = df['life_ladder'].mean()
    top_n_avg = top_n['life_ladder'].mean()
    difference = top_n_avg - global_avg
    
    st.write(f"• Bu ülkelerin ortalama mutluluk skoru ({top_n_avg:.2f}), ")
    st.write(f"  global ortalamanın ({global_avg:.2f}) {difference:.2f} puan üstünde.")
    
    # Faktör analizi
    factors = ['social_support', 'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption']
    factor_means = df[factors].mean()
    top_n_factor_means = top_n.merge(df, on='country_name')[factors].mean()
    
    st.write("\n### 📈 Faktör Analizi")
    for factor in factors:
        diff = top_n_factor_means[factor] - factor_means[factor]
        st.write(f"• {factor}: Global ortalamadan {diff:.2f} puan farklı")
    
    return top_n

def analyze_bottom_n_countries(df, n, year=None):
    """En mutsuz N ülkeyi analiz eder"""
    if year:
        # Belirli yıl için en mutsuz N ülke
        year_data = df[df['year'] == year]
        bottom_n = year_data.nsmallest(n, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
        
        st.write(f"### 😢 {year} Yılının En Mutsuz {n} Ülkesi")
    else:
        # Tüm yılların ortalamasına göre en mutsuz N ülke
        # Önce her ülkenin ortalama mutluluk skorunu hesapla
        country_means = df.groupby('country_name')['life_ladder'].mean().reset_index()
        
        # En mutsuz N ülkeyi seç
        bottom_n_countries = country_means.nsmallest(n, 'life_ladder')['country_name'].tolist()
        
        # Bu ülkelerin verilerini al ve ortalamaları hesapla
        bottom_n_data = df[df['country_name'].isin(bottom_n_countries)].groupby(
            ['country_name', 'regional_indicator']
        ).agg({
            'life_ladder': 'mean'
        }).reset_index()
        
        # Sırala
        bottom_n = bottom_n_data.sort_values('life_ladder')
        
        st.write(f"### 😢 Tüm Zamanların En Mutsuz {n} Ülkesi")
        st.write("(Tüm yılların ortalamasına göre)")
    
    # Görselleştirme
    fig = px.bar(
        bottom_n,
        x='country_name',
        y='life_ladder',
        color='regional_indicator',
        title=f'En Mutsuz {n} Ülke' + (f' ({year})' if year else ' (Tüm Yılların Ortalaması)'),
        labels={'country_name': 'Ülke', 'life_ladder': 'Mutluluk Skoru', 'regional_indicator': 'Bölge'},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel  # Soft renkler
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bölgesel analiz
    region_counts = bottom_n['regional_indicator'].value_counts()
    st.write("### 🌍 Bölgesel Dağılım Analizi")
    
    fig_pie = px.pie(
        values=region_counts.values,
        names=region_counts.index,
        title=f"En Mutsuz {n} Ülkenin Bölgesel Dağılımı",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detaylı liste ve içgörüler
    st.write("### 📊 Detaylı Liste ve İçgörüler")
    for i, row in bottom_n.iterrows():
        st.write(f"{i + 1}. {row['country_name']}: {row['life_ladder']:.2f} ({row['regional_indicator']})")
    
    # Bölgesel içgörüler
    most_common_region = region_counts.index[0]
    region_percentage = (region_counts[most_common_region] / len(bottom_n)) * 100
    
    st.write("\n### 🔍 Önemli İçgörüler")
    st.write(f"• En mutsuz {n} ülkenin {region_percentage:.1f}%'i {most_common_region} bölgesinde bulunuyor.")
    
    # Global karşılaştırma
    global_avg = df['life_ladder'].mean()
    bottom_n_avg = bottom_n['life_ladder'].mean()
    difference = global_avg - bottom_n_avg
    
    st.write(f"• Bu ülkelerin ortalama mutluluk skoru ({bottom_n_avg:.2f}), ")
    st.write(f"  global ortalamanın ({global_avg:.2f}) {difference:.2f} puan altında.")
    
    # Faktör analizi
    factors = ['social_support', 'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption']
    factor_means = df[factors].mean()
    bottom_n_factor_means = bottom_n.merge(df, on='country_name')[factors].mean()
    
    st.write("\n### 📈 Faktör Analizi")
    for factor in factors:
        diff = bottom_n_factor_means[factor] - factor_means[factor]
        st.write(f"• {factor}: Global ortalamadan {diff:.2f} puan farklı")
    
    return bottom_n

def analyze_factor(df, factor_name):
    """Belirli bir faktörün mutluluk üzerindeki etkisini analiz eder"""
    try:
        # Korelasyon hesapla
        correlation = df['life_ladder'].corr(df[factor_name])
        
        # En son yıl verilerini al
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year]
        
        # En yüksek ve en düşük değerlere sahip ülkeler
        top_5 = latest_data.nlargest(5, factor_name)[['country_name', factor_name, 'life_ladder']]
        bottom_5 = latest_data.nsmallest(5, factor_name)[['country_name', factor_name, 'life_ladder']]
        
        # Faktör seviyelerine göre grupla
        df['factor_level'] = pd.qcut(df[factor_name], q=3, labels=['Düşük', 'Orta', 'Yüksek'])
        
        # Grup istatistiklerini hesapla ve MultiIndex'i düzleştir
        group_stats = df.groupby('factor_level').agg({
            'life_ladder': ['mean', 'count', 'std']
        }).round(3)
        
        # MultiIndex'i düzleştir
        group_stats_flat = pd.DataFrame({
            'factor_level': group_stats.index,
            'mean': group_stats[('life_ladder', 'mean')],
            'count': group_stats[('life_ladder', 'count')],
            'std': group_stats[('life_ladder', 'std')]
        })
        
        # Her seviye için örnek ülkeler
        example_countries = {}
        for level in ['Düşük', 'Orta', 'Yüksek']:
            level_data = df[df['factor_level'] == level]
            example_countries[level] = ', '.join(level_data['country_name'].unique()[:3])
        
        # Metrikleri göster
        st.write(f"### 📊 {factor_name.replace('_', ' ').title()} Analizi")
        
        # Ana metrikler
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Korelasyon", f"{correlation:.3f}")
        with col2:
            st.metric("Global Ortalama", f"{df[factor_name].mean():.3f}")
        with col3:
            st.metric("Standart Sapma", f"{df[factor_name].std():.3f}")
        
        # En iyi ve en kötü ülkeler
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"### 🏆 En Yüksek {factor_name.replace('_', ' ').title()}")
            fig_top = px.bar(
                top_5,
                x='country_name',
                y=factor_name,
                color='life_ladder',
                title=f"En Yüksek 5 Ülke ({latest_year})",
                template="plotly_dark",
                color_continuous_scale="Viridis"
            )
            fig_top.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
            # Tablo olarak göster
            st.write("#### Detaylı Bilgi")
            for _, row in top_5.iterrows():
                st.write(f"• {row['country_name']}: {row[factor_name]:.3f} (Mutluluk: {row['life_ladder']:.2f})")
        
        with col2:
            st.write(f"### 📉 En Düşük {factor_name.replace('_', ' ').title()}")
            fig_bottom = px.bar(
                bottom_5,
                x='country_name',
                y=factor_name,
                color='life_ladder',
                title=f"En Düşük 5 Ülke ({latest_year})",
                template="plotly_dark",
                color_continuous_scale="Viridis"
            )
            fig_bottom.update_layout(
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bottom, use_container_width=True)
            
            # Tablo olarak göster
            st.write("#### Detaylı Bilgi")
            for _, row in bottom_5.iterrows():
                st.write(f"• {row['country_name']}: {row[factor_name]:.3f} (Mutluluk: {row['life_ladder']:.2f})")
        
        # Scatter plot
        st.write("### 📊 Korelasyon Analizi")
        fig_scatter = px.scatter(
            latest_data, 
            x=factor_name, 
            y='life_ladder',
            title=f'{factor_name.replace("_", " ").title()} ve Mutluluk İlişkisi ({latest_year})',
            template="plotly_dark",
            trendline="ols",
            hover_data=['country_name']
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Grup karşılaştırma grafiği
        st.write("### 📊 Seviye Analizi")
        
        # Grafiği oluştur (düzleştirilmiş DataFrame'i kullan)
        fig_bar = px.bar(
            group_stats_flat,
            x='factor_level',
            y='mean',
            title=f'Faktör Seviyelerine Göre Ortalama Mutluluk Skoru',
            template="plotly_dark",
            error_y='std',
            labels={
                'factor_level': 'Faktör Seviyesi',
                'mean': 'Ortalama Mutluluk Skoru'
            }
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detaylı açıklama
        st.write("#### 📝 Grafik Açıklaması")
        st.write(f"""
        Bu grafik, {factor_name.replace('_', ' ').title()} faktörünün mutluluk üzerindeki etkisini gösterir:
        
        • **Düşük Seviye**: Ortalama mutluluk {group_stats_flat.loc[group_stats_flat['factor_level'] == 'Düşük', 'mean'].iloc[0]:.2f} 
          - Örnek ülkeler: {example_countries['Düşük']}
        
        • **Orta Seviye**: Ortalama mutluluk {group_stats_flat.loc[group_stats_flat['factor_level'] == 'Orta', 'mean'].iloc[0]:.2f}
          - Örnek ülkeler: {example_countries['Orta']}
        
        • **Yüksek Seviye**: Ortalama mutluluk {group_stats_flat.loc[group_stats_flat['factor_level'] == 'Yüksek', 'mean'].iloc[0]:.2f}
          - Örnek ülkeler: {example_countries['Yüksek']}
        
        Hata çubukları (beyaz çizgiler) her gruptaki değişkenliği gösterir. 
        Uzun hata çubukları, o gruptaki ülkeler arasında büyük farklılıklar olduğunu gösterir.
        """)
        
        # Seviyeler arası fark analizi
        highest_mean = group_stats_flat['mean'].max()
        lowest_mean = group_stats_flat['mean'].min()
        diff = highest_mean - lowest_mean
        
        st.write("#### 🔍 Seviyeler Arası Fark")
        st.write(f"""
        • En yüksek ve en düşük seviye arasında {diff:.2f} puanlık bir mutluluk farkı var.
        • Bu, {factor_name.replace('_', ' ')}'in mutluluk üzerinde önemli bir etkisi olduğunu gösterir.
        """)
        
        # İçgörüler
        st.write("### 🔍 Önemli İçgörüler")
        
        # Korelasyon yorumu
        if abs(correlation) > 0.5:
            strength = "güçlü"
        elif abs(correlation) > 0.3:
            strength = "orta düzeyde"
        else:
            strength = "zayıf"
            
        direction = "pozitif" if correlation > 0 else "negatif"
        
        # En yüksek ve en düşük ülkelerin mutluluk farkı
        top_happiness = top_5['life_ladder'].mean()
        bottom_happiness = bottom_5['life_ladder'].mean()
        happiness_diff = top_happiness - bottom_happiness
        
        st.write(f"1. {factor_name.replace('_', ' ').title()} ile mutluluk arasında {strength} {direction} korelasyon ({correlation:.3f}) bulunmaktadır.")
        st.write(f"2. En yüksek {factor_name.replace('_', ' ')} değerlerine sahip ülkeler, en düşük olanlara göre ortalama {happiness_diff:.2f} puan daha mutlu.")
        st.write(f"3. {top_5.iloc[0]['country_name']}, en yüksek {factor_name.replace('_', ' ')} değerine ({top_5.iloc[0][factor_name]:.3f}) sahip ülkedir.")
        
        # Öneriler
        st.write("### 💡 Politika Önerileri")
        if correlation > 0:
            st.write(f"1. {factor_name.replace('_', ' ').title()} seviyesini artırmaya yönelik politikalar mutluluğu artırabilir.")
            st.write(f"2. Özellikle {bottom_5['country_name'].iloc[0]} gibi düşük değerlere sahip ülkelere odaklanılmalı.")
            st.write(f"3. {top_5['country_name'].iloc[0]}'nin başarılı uygulamaları örnek alınabilir.")
        else:
            st.write(f"1. {factor_name.replace('_', ' ').title()} seviyesini optimize etmeye yönelik politikalar geliştirilmeli.")
            st.write(f"2. {factor_name.replace('_', ' ').title()} ile mutluluk arasındaki negatif ilişkinin nedenleri araştırılmalı.")
            st.write(f"3. Ülkelerin sosyo-ekonomik koşullarına göre özelleştirilmiş stratejiler geliştirilmeli.")
        
        return True
    except Exception as e:
        st.error(f"Faktör analizi sırasında hata oluştu: {str(e)}")
        return False

@st.cache_data(ttl=3600)  # 1 saat önbellek
def process_llm_response(response, df):
    """LLM yanıtını işle ve önbellekle"""
    try:
        response = str(response).strip()
        
        # Yanıt boş değilse işle
        if response:
            # Markdown formatındaki başlıkları işle
            sections = response.split('\n')
            for section in sections:
                section = section.strip()
                if section:
                    # Görsel referanslarını kontrol et
                    visual_patterns = [
                        ('[Görsel:', ']'),
                        ('[Trend Grafiği', ']'),
                        ('[Bar Grafiği', ']'),
                        ('[Grafik:', ']'),
                        ('[Grafik', ']')
                    ]
                    
                    is_visual = False
                    for start_pattern, end_pattern in visual_patterns:
                        if start_pattern in section:
                            is_visual = True
                            # Görsel referansını bul
                            start_idx = section.find(start_pattern)
                            end_idx = section.find(end_pattern, start_idx)
                            if end_idx != -1:
                                # Görsel açıklamasını al
                                visual_desc = section[start_idx + len(start_pattern):end_idx].strip()
                                
                                # Önceki metni göster
                                if start_idx > 0:
                                    st.write(section[:start_idx])
                                
                                # Görsel tipini ve içeriğini analiz et
                                countries = []
                                metric = None
                                
                                # Ülkeleri tespit et
                                if 'türkiye' in section.lower():
                                    countries.append('Turkiye')
                                if 'almanya' in section.lower():
                                    countries.append('Germany')
                                
                                # Metriği tespit et
                                if 'mutluluk' in section.lower():
                                    metric = 'life_ladder'
                                elif 'gsyih' in section.lower() or 'gdp' in section.lower():
                                    metric = 'gdp_per_capita'
                                elif 'yaşam beklentisi' in section.lower():
                                    metric = 'life_expectancy'
                                
                                # Grafik tipini belirle
                                is_trend = any(keyword in section.lower() for keyword in ['trend', 'değişim', 'zaman'])
                                is_bar = any(keyword in section.lower() for keyword in ['bar', 'sütun', 'karşılaştırma'])
                                
                                if countries and metric:
                                    if is_trend or not is_bar:  # Varsayılan olarak trend grafiği kullan
                                        fig = go.Figure()
                                        
                                        for country in countries:
                                            country_data = df[df['country_name'] == country]
                                            fig.add_trace(go.Scatter(
                                                x=country_data['year'],
                                                y=country_data[metric],
                                                mode='lines+markers',
                                                name=country,
                                                line=dict(width=3),
                                                marker=dict(size=8)
                                            ))
                                        
                                        fig.update_layout(
                                            title=f"{', '.join(countries)} {metric.replace('_', ' ').title()} Trendi",
                                            xaxis_title='Yıl',
                                            yaxis_title=metric.replace('_', ' ').title(),
                                            template='plotly_dark',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            hovermode='x unified'
                                        )
                                    else:  # Bar grafiği
                                        latest_year = df['year'].max()
                                        latest_data = df[
                                            (df['country_name'].isin(countries)) & 
                                            (df['year'] == latest_year)
                                        ]
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Bar(
                                            x=latest_data['country_name'],
                                            y=latest_data[metric],
                                            text=latest_data[metric].round(2),
                                            textposition='auto'
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"{', '.join(countries)} {metric.replace('_', ' ').title()} Karşılaştırması",
                                            xaxis_title='Ülkeler',
                                            yaxis_title=metric.replace('_', ' ').title(),
                                            template='plotly_dark',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            showlegend=False
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Kalan metni göster
                                if end_idx + 1 < len(section):
                                    st.write(section[end_idx + 1:])
                            break
                    
                    if not is_visual:
                        if section.startswith('#'):
                            # Başlık seviyesini belirle
                            level = len(section.split()[0])  # '#' sayısı
                            title = section.lstrip('#').strip()
                            if level == 1:
                                st.title(title)
                            elif level == 2:
                                st.header(title)
                            elif level == 3:
                                st.subheader(title)
                            else:
                                st.markdown(f"**{title}**")
                        else:
                            # Normal metin
                            st.write(section)
        else:
            st.warning("Yanıt boş veya geçersiz format.")
            
        return None  # Return None instead of True to prevent displaying return value
            
    except Exception as e:
        st.error(f"Yanıt işlenirken hata oluştu: {str(e)}")
        st.warning("Ham yanıt:")
        st.code(response)
        return None

def validate_data_types(data):
    """JSON verilerinin tiplerini kontrol et"""
    try:
        # Temel analiz kontrolü
        assert isinstance(data['temel_analiz']['mutluluk_skoru'], (int, float))
        assert isinstance(data['temel_analiz']['global_fark'], (int, float))
        assert isinstance(data['temel_analiz']['siralama'], int)
        assert isinstance(data['temel_analiz']['yuzdelik'], (int, float))
        
        # Bölgesel analiz kontrolü
        assert isinstance(data['bolgesel_analiz']['bolge_ortalamasi'], (int, float))
        assert isinstance(data['bolgesel_analiz']['bolge_siralamasi'], int)
        
        # Trend analiz kontrolü
        assert isinstance(data['trend_analiz']['son_5_yil_degisim'], (int, float))
        assert isinstance(data['trend_analiz']['yillik_ortalama'], (int, float))
        
        # İçgörüler ve öneriler kontrolü
        assert len(data['icgorular']) >= 3
        assert len(data['oneriler']) >= 2
        
    except AssertionError as e:
        st.error("Veri tipleri beklenen formatta değil!")
        raise

def display_analysis_results(data):
    """Analiz sonuçlarını görselleştir"""
    # Temel Analiz
    st.write("### 📊 Temel Analiz")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Mutluluk Skoru", 
            f"{data['temel_analiz']['mutluluk_skoru']:.2f}",
            f"{data['temel_analiz']['global_fark']:+.2f}"
        )
    with col2:
        st.metric(
            "Global Sıralama", 
            f"{data['temel_analiz']['siralama']}.",
            f"{data['temel_analiz']['yuzdelik']:.1f}%"
        )
    
    # Bölgesel Analiz
    st.write("### 🌍 Bölgesel Analiz")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bölge Ortalaması", f"{data['bolgesel_analiz']['bolge_ortalamasi']:.2f}")
    with col2:
        st.metric("Bölge Sıralaması", f"{data['bolgesel_analiz']['bolge_siralamasi']}.")
    
    # Trend Analizi
    st.write("### 📈 Trend Analizi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Son 5 Yıl Değişim",
            f"{abs(data['trend_analiz']['son_5_yil_degisim']):.2f}",
            f"{'↑' if data['trend_analiz']['son_5_yil_degisim'] > 0 else '↓'}"
        )
    with col2:
        st.metric("Yıllık Ortalama", f"{data['trend_analiz']['yillik_ortalama']:.2f}")
    
    # İçgörüler
    st.write("### 🔍 Önemli İçgörüler")
    for i, insight in enumerate(data['icgorular'], 1):
        st.write(f"{i}. {insight}")
    
    # Öneriler
    st.write("### 💡 Öneriler")
    for i, suggestion in enumerate(data['oneriler'], 1):
        st.write(f"{i}. {suggestion}")

def analyze_time_series(country_data):
    """Gelişmiş zaman serisi analizi yapar"""
    try:
        # Trend analizi
        X = country_data['year'].values.reshape(-1, 1)
        y = country_data['life_ladder'].values
        
        # Son 3 yıllık hareketli ortalama
        country_data['moving_avg'] = country_data['life_ladder'].rolling(window=3).mean()
        
        # Yıllık değişim oranı
        country_data['yearly_change'] = country_data['life_ladder'].pct_change() * 100
        
        # Büyüme hızı (CAGR)
        years = len(country_data) - 1
        if years > 0:
            total_growth = (country_data['life_ladder'].iloc[-1] / country_data['life_ladder'].iloc[0]) - 1
            cagr = (1 + total_growth) ** (1/years) - 1
        else:
            cagr = 0
            
        return {
            'moving_avg': country_data['moving_avg'],
            'yearly_changes': country_data['yearly_change'],
            'cagr': cagr * 100
        }
    except Exception as e:
        st.error(f"Zaman serisi analizi sırasında hata: {str(e)}")
        return None

def analyze_factors_correlation(country_data, df):
    """Faktörler arası korelasyon analizi yapar"""
    try:
        factors = ['social_support', 'freedom_to_make_life_choices', 
                  'generosity', 'perceptions_of_corruption',
                  'gdp_per_capita', 'life_expectancy']
        
        # Mevcut faktörleri kontrol et
        available_factors = [f for f in factors if f in country_data.columns]
        
        # Korelasyon matrisi
        corr_matrix = country_data[available_factors + ['life_ladder']].corr()
        
        # Faktörlerin mutluluk ile korelasyonu
        happiness_corr = corr_matrix['life_ladder'].drop('life_ladder')
        
        # Global korelasyonlarla karşılaştırma
        global_corr = df[available_factors + ['life_ladder']].corr()['life_ladder'].drop('life_ladder')
        
        return {
            'local_correlations': happiness_corr,
            'global_correlations': global_corr
        }
    except Exception as e:
        st.error(f"Faktör korelasyonu analizi sırasında hata: {str(e)}")
        return None

def predict_happiness(country_data):
    """Gelecek yıl için mutluluk tahmini yapar"""
    try:
        # En az 5 yıllık veri gerekli
        if len(country_data) < 5:
            return None
            
        # Son 5 yılın verilerini al
        recent_data = country_data.sort_values('year').tail(5)
        
        # Basit doğrusal regresyon
        X = recent_data['year'].values.reshape(-1, 1)
        y = recent_data['life_ladder'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Gelecek yıl tahmini
        next_year = recent_data['year'].max() + 1
        prediction = model.predict([[next_year]])[0]
        
        # R-kare skoru
        r2_score = model.score(X, y)
        
        return {
            'next_year': next_year,
            'prediction': prediction,
            'confidence': r2_score,
            'trend': 'artış' if model.coef_[0] > 0 else 'düşüş'
        }
    except Exception as e:
        st.error(f"Tahmin analizi sırasında hata: {str(e)}")
        return None

def find_similar_countries(df, country_data, latest_data):
    """Benzer özelliklere sahip ülkeleri bulur"""
    try:
        # Karşılaştırma faktörleri
        factors = ['social_support', 'freedom_to_make_life_choices', 
                  'generosity', 'perceptions_of_corruption',
                  'gdp_per_capita', 'life_expectancy']
        
        # Mevcut faktörleri kontrol et
        available_factors = [f for f in factors if f in df.columns]
        
        if not available_factors:
            return None
            
        # Son yıl verilerini al
        latest_year = df['year'].max()
        year_data = df[df['year'] == latest_year].copy()
        
        # Hedef ülkenin verilerini al
        target_country = year_data[year_data['country_name'] == latest_data['country_name']]
        
        if len(target_country) == 0:
            return None
        
        # Öklid mesafesini hesapla
        scaler = StandardScaler()
        
        # Verileri normalize et
        normalized_data = scaler.fit_transform(year_data[available_factors])
        normalized_df = pd.DataFrame(normalized_data, columns=available_factors, index=year_data.index)
        
        # Hedef ülkenin normalize edilmiş değerleri
        target_values = normalized_df.loc[target_country.index[0]]
        
        # Her ülke için mesafeyi hesapla
        distances = []
        for idx, row in normalized_df.iterrows():
            if year_data.iloc[idx]['country_name'] != latest_data['country_name']:
                distance = np.sqrt(((row - target_values) ** 2).sum())
                distances.append({
                    'country': year_data.iloc[idx]['country_name'],
                    'distance': distance,
                    'happiness': year_data.iloc[idx]['life_ladder']
                })
        
        # En yakın 5 ülkeyi bul
        similar_countries = sorted(distances, key=lambda x: x['distance'])[:5]
        
        return similar_countries
    except Exception as e:
        st.error(f"Benzer ülke analizi sırasında hata: {str(e)}")
        return None

def analyze_specific_country(df, country_name):
    try:
        country_data = df[df['country_name'].str.contains(country_name, case=False, na=False)]
        
        if len(country_data) == 0:
            st.error(f"{country_name} için veri bulunamadı!")
            return None
            
        latest_data = country_data.sort_values('year', ascending=False).iloc[0]
        
        # Başlık ve Açıklama - Daha kompakt
        st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 0.8rem; border-radius: 10px; margin-bottom: 0.8rem; text-align: center;'>
            <h1 style='margin: 0; color: #8dd3c7; font-size: 2rem;'>{country_name}</h1>
            <p style='margin: 0.3rem 0 0 0; color: #bebada; font-size: 1rem;'>Mutluluk Analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Ana metrikler ve trend grafiği yan yana
        col1, col2 = st.columns([0.25, 0.75])
        
        with col1:
            # Temel metrikler - Daha kompakt
            st.markdown(f"""
            <div style='background-color: #2E2E2E; padding: 0.6rem; border-radius: 8px; text-align: center; margin-bottom: 0.4rem;'>
                <h4 style='margin: 0; color: #8dd3c7; font-size: 0.9rem;'>Son Mutluluk Skoru</h4>
                <h2 style='margin: 0.2rem 0; color: white; font-size: 1.4rem;'>{latest_data['life_ladder']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color: #2E2E2E; padding: 0.6rem; border-radius: 8px; text-align: center; margin-bottom: 0.4rem;'>
                <h4 style='margin: 0; color: #8dd3c7; font-size: 0.9rem;'>Ortalama Skor</h4>
                <h2 style='margin: 0.2rem 0; color: white; font-size: 1.4rem;'>{country_data['life_ladder'].mean():.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            change = latest_data['life_ladder'] - country_data.iloc[-1]['life_ladder']
            change_color = '#4CAF50' if change >= 0 else '#FF5252'
            st.markdown(f"""
            <div style='background-color: #2E2E2E; padding: 0.6rem; border-radius: 8px; text-align: center;'>
                <h4 style='margin: 0; color: #8dd3c7; font-size: 0.9rem;'>Toplam Değişim</h4>
                <h2 style='margin: 0.2rem 0; color: {change_color}; font-size: 1.4rem;'>{change:+.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Trend grafiği
            fig = go.Figure()
            
            # Ülkenin mutluluk skoru
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data['life_ladder'],
                mode='lines+markers',
                name=f'{country_name}',
                line=dict(color='#8dd3c7', width=3),
                marker=dict(size=8)
            ))
            
            # Bölge ortalaması
            if 'regional_indicator' in latest_data:
                region = latest_data['regional_indicator']
                region_data = df[df['regional_indicator'] == region].groupby('year')['life_ladder'].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=region_data['year'],
                    y=region_data['life_ladder'],
                    mode='lines',
                    name=f'Bölge Ortalaması',
                    line=dict(color='#bebada', width=2, dash='dash')
                ))
            
            # Global ortalama
            global_data = df.groupby('year')['life_ladder'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=global_data['year'],
                y=global_data['life_ladder'],
                mode='lines',
                name='Global Ortalama',
                line=dict(color='#fb8072', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title=dict(
                    text="Mutluluk Trendi",
                    font=dict(size=20)
                ),
                height=400,
                xaxis_title="Yıl",
                yaxis_title="Mutluluk Skoru",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(0,0,0,0.5)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Bölgesel Sıralama Grafiği
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='margin: 0; color: #8dd3c7;'>📊 Bölgesel Sıralama</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Bölgedeki tüm ülkelerin son yıl verilerini al
        region_data = df[df['year'] == latest_data['year']]
        region_data = region_data[region_data['regional_indicator'] == region]
        region_data = region_data.sort_values('life_ladder', ascending=False)  # En mutlu ülkeler üstte
        
        # Renk listesi oluştur (seçili ülke turkuaz, diğerleri gri)
        colors = ['#808080' if x != country_name else '#8dd3c7' for x in region_data['country_name']]
        
        # Yatay bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=region_data['country_name'],
            x=region_data['life_ladder'],
            orientation='h',
            marker_color=colors,
            text=region_data['life_ladder'].round(2),
            textposition='auto',
        ))
        
        fig.update_layout(
            height=max(400, len(region_data) * 25),  # Ülke sayısına göre dinamik yükseklik
            xaxis_title="Mutluluk Skoru",
            yaxis_title=None,  # Y ekseni başlığını kaldır
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                title_font=dict(size=12),
            ),
            yaxis=dict(
                showgrid=False,
                title_font=dict(size=12),
            ),
            hoverlabel=dict(
                bgcolor='rgba(0,0,0,0.8)',
                font_size=12,
            ),
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Faktör grupları - Tab sistemi
        st.markdown("### 📊 Faktör Analizleri")
        
        # Faktör grupları
        factor_groups = {
            'Ekonomik Faktörler 💰': {
                'gdp_per_capita': 'Kişi Başı GDP',
                'education_expenditure_gdp': 'Eğitim Harcamaları'
            },
            'Sosyal Faktörler 👥': {
                'social_support': 'Sosyal Destek',
                'freedom_to_make_life_choices': 'Özgürlük',
                'generosity': 'Cömertlik'
            },
            'Yaşam Kalitesi 🌟': {
                'life_expectancy': 'Yaşam Beklentisi',
                'internet_users_percent': 'İnternet Kullanımı'
            },
            'Diğer Faktörler ⚖️': {
                'perceptions_of_corruption': 'Yolsuzluk Algısı'
            }
        }

        # Ana faktör grupları için tabs
        tabs = st.tabs(list(factor_groups.keys()))
        
        for tab, (group_name, factors) in zip(tabs, factor_groups.items()):
            with tab:
                # Mevcut faktörleri kontrol et
                available_factors = {k: v for k, v in factors.items() if k in country_data.columns}
                
                if available_factors:
                    # Her faktör grubu için alt sekmeler
                    factor_tabs = st.tabs(list(available_factors.values()))
                    
                    for factor_tab, (factor_code, factor_name) in zip(factor_tabs, available_factors.items()):
                        with factor_tab:
                            # Grafik ve metrikler yan yana
                            col1, col2 = st.columns([0.7, 0.3])
                            
                            with col1:
                                fig = go.Figure()
                                
                                # Ülke verisi
                                fig.add_trace(go.Scatter(
                                    x=country_data['year'],
                                    y=country_data[factor_code],
                                    mode='lines+markers',
                                    name=f'{country_name}',
                                    line=dict(color='#8dd3c7', width=3),
                                    marker=dict(size=8)
                                ))
                                
                                # Bölge ortalaması
                                if 'regional_indicator' in latest_data:
                                    region_data = df[df['regional_indicator'] == region].groupby('year')[factor_code].mean().reset_index()
                                    fig.add_trace(go.Scatter(
                                        x=region_data['year'],
                                        y=region_data[factor_code],
                                        mode='lines',
                                        name=f'Bölge Ortalaması',
                                        line=dict(color='#bebada', width=2, dash='dash')
                                    ))
                                
                                # Global ortalama
                                global_data = df.groupby('year')[factor_code].mean().reset_index()
                                fig.add_trace(go.Scatter(
                                    x=global_data['year'],
                                    y=global_data[factor_code],
                                    mode='lines',
                                    name='Global Ortalama',
                                    line=dict(color='#fb8072', width=2, dash='dot')
                                ))
                                
                                fig.update_layout(
                                    height=400,
                                    xaxis_title="Yıl",
                                    yaxis_title=factor_name,
                                    template="plotly_dark",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(l=40, r=40, t=40, b=40),
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01,
                                        bgcolor='rgba(0,0,0,0.5)'
                                    )
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Metrikler - Daha kompakt
                                latest_value = latest_data[factor_code]
                                global_avg = df[df['year'] == latest_data['year']][factor_code].mean()
                                region_avg = df[(df['year'] == latest_data['year']) & 
                                              (df['regional_indicator'] == region)][factor_code].mean()
                                
                                st.markdown(f"""
                                <div style='background-color: #2E2E2E; padding: 0.6rem; border-radius: 8px; text-align: center; margin-bottom: 0.4rem;'>
                                    <h4 style='margin: 0; color: #8dd3c7; font-size: 0.9rem;'>Son Değer</h4>
                                    <h2 style='margin: 0.2rem 0; color: white; font-size: 1.4rem;'>{latest_value:.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                diff_global = latest_value - global_avg
                                diff_color = '#4CAF50' if diff_global >= 0 else '#FF5252'
                                st.markdown(f"""
                                <div style='background-color: #2E2E2E; padding: 0.6rem; border-radius: 8px; text-align: center; margin-bottom: 0.4rem;'>
                                    <h4 style='margin: 0; color: #8dd3c7; font-size: 0.9rem;'>Global Fark</h4>
                                    <h2 style='margin: 0.2rem 0; color: {diff_color}; font-size: 1.4rem;'>{diff_global:+.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                diff_region = latest_value - region_avg
                                diff_color = '#4CAF50' if diff_region >= 0 else '#FF5252'
                                st.markdown(f"""
                                <div style='background-color: #2E2E2E; padding: 0.6rem; border-radius: 8px; text-align: center;'>
                                    <h4 style='margin: 0; color: #8dd3c7; font-size: 0.9rem;'>Bölgesel Fark</h4>
                                    <h2 style='margin: 0.2rem 0; color: {diff_color}; font-size: 1.4rem;'>{diff_region:+.2f}</h2>
                                </div>
                                """, unsafe_allow_html=True)

        # Önemli İçgörüler bölümü
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='margin: 0; color: #8dd3c7;'>🔍 Önemli İçgörüler</h3>
        </div>
        """, unsafe_allow_html=True)

        # Mutluluk Trendi Analizi
        trend_change = latest_data['life_ladder'] - country_data.iloc[-1]['life_ladder']
        avg_happiness = country_data['life_ladder'].mean()
        global_avg = df['life_ladder'].mean()
        region = latest_data['regional_indicator']
        region_avg = df[df['regional_indicator'] == region]['life_ladder'].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background-color: #2E2E2E; padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h4 style='color: #8dd3c7; margin: 0;'>📈 Trend Analizi</h4>
                <ul style='margin: 0.5rem 0;'>""", unsafe_allow_html=True)
            
            years_span = len(country_data)
            if abs(trend_change) < 0.01:
                st.markdown(f"<li>Son {years_span} yılda mutluluk skorunda önemli bir değişim gözlenmedi</li>", unsafe_allow_html=True)
            elif trend_change > 0:
                st.markdown(f"<li>Son {years_span} yılda mutluluk skoru {trend_change:.2f} puan artış gösterdi</li>", unsafe_allow_html=True)
            else:
                st.markdown(f"<li>Son {years_span} yılda mutluluk skoru {abs(trend_change):.2f} puan düşüş gösterdi</li>", unsafe_allow_html=True)
            
            if avg_happiness > global_avg:
                st.markdown(f"<li>Ortalama mutluluk ({avg_happiness:.2f}), global ortalamanın ({global_avg:.2f}) üzerinde</li>", unsafe_allow_html=True)
            else:
                st.markdown(f"<li>Ortalama mutluluk ({avg_happiness:.2f}), global ortalamanın ({global_avg:.2f}) altında</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background-color: #2E2E2E; padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem;'>
                <h4 style='color: #8dd3c7; margin: 0;'>🌍 Bölgesel Karşılaştırma</h4>
                <ul style='margin: 0.5rem 0;'>""", unsafe_allow_html=True)
            
            if avg_happiness > region_avg:
                st.markdown(f"<li>{region} bölgesindeki ülkeler arasında ortalamanın üzerinde</li>", unsafe_allow_html=True)
            else:
                st.markdown(f"<li>{region} bölgesindeki ülkeler arasında ortalamanın altında</li>", unsafe_allow_html=True)
            
            # Bölgedeki sıralama
            region_countries = df[df['regional_indicator'] == region]['country_name'].unique()
            region_rank = df[df['year'] == latest_data['year']]
            region_rank = region_rank[region_rank['regional_indicator'] == region]
            region_rank = region_rank.sort_values('life_ladder', ascending=False)
            country_rank = list(region_rank['country_name']).index(country_name) + 1
            total_countries = len(region_rank)
            
            st.markdown(f"<li>Bölgesinde {total_countries} ülke arasında {country_rank}. sırada</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)

        # Faktör Analizi
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='margin: 0; color: #8dd3c7;'>📊 Faktör Analizi</h3>
        </div>
        """, unsafe_allow_html=True)

        # En güçlü ve en zayıf faktörleri bul
        factors_latest = {}
        for group in factor_groups.values():
            for factor_code, factor_name in group.items():
                if factor_code in latest_data:
                    factor_value = latest_data[factor_code]
                    factor_global = df[df['year'] == latest_data['year']][factor_code].mean()
                    factors_latest[factor_name] = (factor_value - factor_global) / factor_global * 100

        # Faktörleri sırala
        sorted_factors = sorted(factors_latest.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # En güçlü ve en zayıf 2 faktörü göster
        st.markdown("""
        <div style='background-color: #2E2E2E; padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem;'>
            <ul style='margin: 0.5rem 0;'>""", unsafe_allow_html=True)
            
        for i, (factor_name, diff) in enumerate(sorted_factors[:2]):
            if diff > 0:
                st.markdown(f"<li>{factor_name} global ortalamanın %{abs(diff):.1f} üzerinde (güçlü yön)</li>", unsafe_allow_html=True)
            else:
                st.markdown(f"<li>{factor_name} global ortalamanın %{abs(diff):.1f} altında (gelişim alanı)</li>", unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)

        # Öneriler
        st.markdown("""
        <div style='background-color: #2E2E2E; padding: 0.8rem; border-radius: 8px;'>
            <h4 style='color: #8dd3c7; margin: 0;'>💡 Öneriler</h4>
            <ul style='margin: 0.5rem 0;'>""", unsafe_allow_html=True)

        # Zayıf faktörler için öneriler
        weak_factors = [f for f, d in sorted_factors if d < 0][:2]
        for factor in weak_factors:
            st.markdown(f"<li>{factor} alanında iyileştirmeler yapılabilir</li>", unsafe_allow_html=True)

        # Trend bazlı öneriler
        if trend_change < 0:
            st.markdown("<li>Son yıllardaki düşüş trendini tersine çevirmek için kapsamlı bir eylem planı geliştirilebilir</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        return None
        
    except Exception as e:
        st.error(f"Ülke analizi sırasında hata: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # 1 saat önbellek
async def get_answer(question, df):
    """LLM yanıtı al ve işle"""
    try:
        # Soruyu küçük harfe çevir ve Türkçe karakterleri normalize et
        question_lower = question.lower()
        tr_to_en = str.maketrans("çğıöşüİ", "cgiosui")
        question_normalized = question_lower.translate(tr_to_en)
        
        # Multi-agent sistemini kullan
        from llm_agents import MultiAgentSystem, ConversationManager
        
        # Singleton pattern ile conversation manager'ı oluştur
        if 'conversation_manager' not in st.session_state:
            st.session_state.conversation_manager = ConversationManager()
        
        # Multi-agent sistemini oluştur
        multi_agent = MultiAgentSystem(df)
        
        # Geçmiş bağlamı kontrol et
        relevant_history = st.session_state.conversation_manager.get_relevant_context(question)
        if relevant_history:
            with st.expander("Benzer Sorular", expanded=False):
                for entry in relevant_history:
                    st.write(f"Soru: {entry['question']}")
                    st.write(f"Yanıt: {entry['answer'][:200]}...")
                    st.write("---")
        
        # Soruyu yanıtla
        try:
            # Agent'dan yanıt al
            answer = await multi_agent.get_answer(question)
            
            # Yanıtı geçmişe ekle
            if answer:  # Boş yanıt değilse ekle
                st.session_state.conversation_manager.add_to_history(
                    question=question,
                    answer=answer,
                    agent_type=multi_agent.route_question(question)
                )
            
            return answer
            
        except Exception as e:
            st.error(f"Yanıt alınırken hata oluştu: {str(e)}")
            return None
        
    except Exception as e:
        st.error(f"Analiz sırasında hata: {str(e)}")
        return None

def main():
    # Sayfa konfigürasyonu
    st.set_page_config(
        page_title="Global Mutluluk Analizi",
        page_icon="🌍",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    /* Ana Tema ve Renkler */
    :root {
        --primary-color: #8dd3c7;
        --secondary-color: #bebada;
        --background-dark: #1a1a2e;
        --background-light: rgba(255, 255, 255, 0.05);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --accent-color: #fb8072;
    }

    /* Genel Stiller */
    .stApp {
        background: linear-gradient(135deg, var(--background-dark) 0%, #16213e 100%);
    }
    
    /* Ana wrapper için margin */
    .main-wrapper {
        margin-top: 5rem !important;
    }
    
    /* Ana container genişliği ve ortalama */
    .block-container {
        max-width: 70% !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin: 0 auto !important;
    }

    /* Streamlit elementlerini ortala */
    .element-container, .stMarkdown {
        width: 100% !important;
        margin: 0 auto !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Dashboard Container */
    .dashboard-container {
        background: var(--background-light);
        border-radius: 15px;
        padding: 2.5rem 3.5rem;
        margin: 0.5rem auto;
        width: 100%;
        box-sizing: border-box;
    }

    /* Tab Stilleri */
    .stTabs {
        background: var(--background-light);
        border-radius: 10px;
        padding: 2rem;
        margin: 0 auto;
        width: 100%;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        justify-content: center;
        padding: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre;
        background-color: transparent;
        border-radius: 5px;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0 2rem;
    }

    /* Başlık Stilleri */
    .dashboard-title {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-align: center;
    }

    /* Filtre Bar */
    .filter-container {
        background: var(--background-light);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem auto;
        width: 100%;
        box-sizing: border-box;
    }

    /* Navigation Buttons Container */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
        padding: 0 !important;
        margin: 0.5rem 0 !important;
    }

    /* Streamlit Bileşen Override */
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding: 0 !important;
    }

    /* Tab İçerik Alanı */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 2rem !important;
    }

    /* Kart Stilleri */
    .metric-card {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
        text-align: center;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* Grafik Container */
    .chart-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem auto;
        width: 100%;
        box-sizing: border-box;
    }

    /* Metin Stilleri */
    .section-title {
        color: var(--primary-color);
        font-size: 1.4rem;
        font-weight: 500;
        margin: 1.5rem 0;
        text-align: center;
    }

    /* Gizlenecek Elementler */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* Responsive Düzenlemeler */
    @media (max-width: 1200px) {
        .block-container {
            max-width: 85% !important;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            max-width: 95% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        # Veri yükleme ve işleme
        df = load_data()
        if df is None:
            st.error("Veri yüklenemedi! Lütfen 'cleaned_dataset.csv' dosyasının varlığını kontrol edin.")
            return

        df = preprocess_data(df)
        if df is None:
            st.error("Veri işlenemedi!")
            return

        # Session state başlangıcı
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Ana-Sayfa'

        # Ana wrapper
        st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)
        
        # Navigasyon - Ortalanmış
        st.markdown('<div style="display: flex; justify-content: center; gap: 1rem; margin: 1rem 0;">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📊 Ana Sayfa", key="home_btn", use_container_width=True):
                st.session_state.current_page = 'Ana-Sayfa'
                st.rerun()
        with col2:
            if st.button("💬 Soru & Cevap", key="qa_btn", use_container_width=True):
                st.session_state.current_page = 'Soru-Cevap'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Ana içerik
        if st.session_state.current_page == 'Ana-Sayfa':
            # Başlık
            st.markdown('<h1 class="dashboard-title">Dünya Mutluluk Analizi</h1>', unsafe_allow_html=True)
            
            # Filtre Bar - Ortalanmış
            with st.container():
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    years = sorted(df['year'].unique())
                    selected_year = st.selectbox('Yıl Seçin', ['Tümü'] + list(years), index=0)
                
                with col2:
                    regions = sorted(df['regional_indicator'].unique())
                    selected_region = st.selectbox('Bölge Seçin', ['Tümü'] + list(regions))
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Tab Sistemi - Ortalanmış
            tab1, tab2, tab3 = st.tabs(["🌍 Genel Bakış", "📈 Trend Analizi", "🔍 Faktör Analizi"])
            
            with tab1:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Dünya Mutluluk Haritası</h3>', unsafe_allow_html=True)
                
                # Harita verilerini hazırla
                if selected_year != 'Tümü':
                    map_data = df[df['year'] == selected_year].copy()
                    year_text = str(selected_year)
                else:
                    # Tüm yılların ortalamasını al
                    map_data = df.groupby('country_name')['life_ladder'].mean().reset_index()
                    year_text = "Tüm Yıllar"

                # Ülke isimlerini harita için uygun formata dönüştür
                country_name_mapping = {
                    'Turkiye': 'Turkey',
                    'United States': 'United States of America',
                    'Congo (Brazzaville)': 'Republic of Congo',
                    'Congo (Kinshasa)': 'Democratic Republic of the Congo',
                    'Palestinian Territories': 'Palestine',
                    'Taiwan Province of China': 'Taiwan',
                    'Hong Kong S.A.R. of China': 'Hong Kong',
                    'Czechia': 'Czech Republic',
                    'North Macedonia': 'Macedonia',
                    'Eswatini': 'Swaziland'
                }
                
                map_data['country_name'] = map_data['country_name'].replace(country_name_mapping)

                # Harita görselleştirmesi
                fig = go.Figure(data=go.Choropleth(
                    locations=map_data['country_name'],
                    locationmode='country names',
                    z=map_data['life_ladder'],
                    text=map_data['country_name'],
                    colorscale=[
                        [0, 'rgb(255,50,50)'],     # Kırmızı (en düşük)
                        [0.5, 'rgb(255,255,200)'],  # Açık sarı (orta)
                        [1, 'rgb(50,150,50)']      # Yeşil (en yüksek)
                    ],
                    colorbar_title="Mutluluk<br>Skoru",
                    hovertemplate='<b>%{text}</b><br>Mutluluk Skoru: %{z:.2f}<extra></extra>'
                ))

                # Harita düzeni
                fig.update_layout(
                    title=dict(
                        text=f"Dünya Mutluluk Haritası ({year_text})",
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=20, color='white')
                    ),
                    geo=dict(
                        showframe=False,
                        showcoastlines=True,
                        projection_type='equirectangular',
                        coastlinecolor='rgba(255, 255, 255, 0.5)',
                        showland=True,
                        landcolor='rgba(200, 200, 200, 0.1)',
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=600,
                    margin=dict(l=0, r=0, t=50, b=0)
                )

                # Haritayı göster
                st.plotly_chart(fig, use_container_width=True)

                # İstatistikler
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "En Mutlu Ülke",
                        f"{map_data.nlargest(1, 'life_ladder')['country_name'].iloc[0]}",
                        f"{map_data.nlargest(1, 'life_ladder')['life_ladder'].iloc[0]:.2f}"
                    )
                with col2:
                    st.metric(
                        "Global Ortalama",
                        f"{map_data['life_ladder'].mean():.2f}",
                        f"±{map_data['life_ladder'].std():.2f} std"
                    )
                with col3:
                    st.metric(
                        "En Mutsuz Ülke",
                        f"{map_data.nsmallest(1, 'life_ladder')['country_name'].iloc[0]}",
                        f"{map_data.nsmallest(1, 'life_ladder')['life_ladder'].iloc[0]:.2f}"
                    )

                # Bölümler arası boşluk
                st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

                # Bölgesel Mutluluk Ortalamaları
                st.markdown("### 🌍 Bölgesel Mutluluk Ortalamaları")
                
                # Bölgesel ortalamaları hesapla
                if selected_year != 'Tümü':
                    regional_avg = df[df['year'] == selected_year].groupby('regional_indicator')['life_ladder'].mean().reset_index()
                    year_text = str(selected_year)
                else:
                    regional_avg = df.groupby('regional_indicator')['life_ladder'].mean().reset_index()
                    year_text = "Tüm Yıllar"
                
                # Ortalamalara göre sırala
                regional_avg = regional_avg.sort_values('life_ladder', ascending=True)
                
                # Bar chart oluştur
                fig_regional = go.Figure()
                
                # Renk skalası oluştur
                colors = [
                    f'rgb({int(255 - (i * (255-50)/(len(regional_avg)-1)))}, '
                    f'{int(50 + (i * (150-50)/(len(regional_avg)-1)))}, 50)'
                    for i in range(len(regional_avg))
                ]
                
                fig_regional.add_trace(go.Bar(
                    y=regional_avg['regional_indicator'],
                    x=regional_avg['life_ladder'],
                    orientation='h',
                    marker_color=colors,
                    text=regional_avg['life_ladder'].round(2),
                    textposition='auto'
                ))
                
                fig_regional.update_layout(
                    title=dict(
                        text=f"Bölgesel Mutluluk Ortalamaları ({year_text})",
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=20, color='white')
                    ),
                    xaxis_title="Mutluluk Skoru",
                    yaxis_title=None,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=False,
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128, 128, 128, 0.2)'
                    ),
                    yaxis=dict(
                        showgrid=False
                    )
                )
                
                st.plotly_chart(fig_regional, use_container_width=True)
                
                # Bölgesel içgörüler
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **En Mutlu Bölge**: {regional_avg.iloc[-1]['regional_indicator']}
                    - Ortalama Skor: {regional_avg.iloc[-1]['life_ladder']:.2f}
                    """)
                with col2:
                    st.warning(f"""
                    **En Mutsuz Bölge**: {regional_avg.iloc[0]['regional_indicator']}
                    - Ortalama Skor: {regional_avg.iloc[0]['life_ladder']:.2f}
                    """)

                # Bölümler arası boşluk
                st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

                # En Mutlu ve En Mutsuz 10 Ülke
                st.markdown("### 🌟 En Mutlu ve En Mutsuz 10 Ülke")
                
                # Verileri hazırla
                if selected_year != 'Tümü':
                    top_10 = df[df['year'] == selected_year].nlargest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    bottom_10 = df[df['year'] == selected_year].nsmallest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    comparison_title = f"En Mutlu ve En Mutsuz 10 Ülke ({selected_year})"
                else:
                    # Tüm yılların ortalamasını al
                    avg_happiness = df.groupby('country_name')['life_ladder'].mean().reset_index()
                    # Bölge bilgisini ekle (en son yılın bölge bilgisini kullan)
                    latest_year = df['year'].max()
                    region_info = df[df['year'] == latest_year][['country_name', 'regional_indicator']].drop_duplicates()
                    avg_happiness = avg_happiness.merge(region_info, on='country_name')
                    
                    top_10 = avg_happiness.nlargest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    bottom_10 = avg_happiness.nsmallest(10, 'life_ladder')[['country_name', 'life_ladder', 'regional_indicator']]
                    comparison_title = "En Mutlu ve En Mutsuz 10 Ülke (Tüm Yılların Ortalaması)"

                # Görselleştirme için iki sütun oluştur
                col1, col2 = st.columns(2)
                
                with col1:
                    # En mutlu 10 ülke grafiği
                    fig_top = go.Figure()
                    fig_top.add_trace(go.Bar(
                        y=top_10['country_name'],
                        x=top_10['life_ladder'],
                        orientation='h',
                        marker_color='rgb(50,150,50)',  # Yeşil
                        text=top_10['life_ladder'].round(2),
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                    ))
                    
                    fig_top.update_layout(
                        title=dict(
                            text="En Mutlu 10 Ülke",
                            x=0.5,
                            y=0.95,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16, color='white')
                        ),
                        xaxis_title="Mutluluk Skoru",
                        yaxis_title=None,
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0),
                        showlegend=False,
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        ),
                        yaxis=dict(
                            showgrid=False,
                            autorange="reversed"  # En yüksek değeri en üstte göster
                        )
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
                
                with col2:
                    # En mutsuz 10 ülke grafiği
                    fig_bottom = go.Figure()
                    fig_bottom.add_trace(go.Bar(
                        y=bottom_10['country_name'],
                        x=bottom_10['life_ladder'],
                        orientation='h',
                        marker_color='rgb(255,50,50)',  # Kırmızı
                        text=bottom_10['life_ladder'].round(2),
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Mutluluk Skoru: %{x:.2f}<extra></extra>'
                    ))
                    
                    fig_bottom.update_layout(
                        title=dict(
                            text="En Mutsuz 10 Ülke",
                            x=0.5,
                            y=0.95,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=16, color='white')
                        ),
                        xaxis_title="Mutluluk Skoru",
                        yaxis_title=None,
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0),
                        showlegend=False,
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)'
                        ),
                        yaxis=dict(
                            showgrid=False
                        )
                    )
                    st.plotly_chart(fig_bottom, use_container_width=True)
                
                # Bölgesel dağılım analizi
                top_regions = top_10['regional_indicator'].value_counts()
                bottom_regions = bottom_10['regional_indicator'].value_counts()
                
                # İçgörüler
                st.markdown("#### 🔍 Önemli İçgörüler")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **En Mutlu 10 Ülke Analizi**:
                    - Ortalama Mutluluk: {top_10['life_ladder'].mean():.2f}
                    - En Yaygın Bölge: {top_regions.index[0]} ({top_regions.iloc[0]} ülke)
                    - En Yüksek Skor: {top_10['life_ladder'].max():.2f} ({top_10.iloc[0]['country_name']})
                    """)
                
                with col2:
                    st.warning(f"""
                    **En Mutsuz 10 Ülke Analizi**:
                    - Ortalama Mutluluk: {bottom_10['life_ladder'].mean():.2f}
                    - En Yaygın Bölge: {bottom_regions.index[0]} ({bottom_regions.iloc[0]} ülke)
                    - En Düşük Skor: {bottom_10['life_ladder'].min():.2f} ({bottom_10.iloc[-1]['country_name']})
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">Mutluluk Trend Analizi</h3>', unsafe_allow_html=True)
                
                # Global trend analizi
                st.markdown("### 📈 Global Mutluluk Trendi")
                
                # Yıllara göre global ortalama
                global_trend = df.groupby('year')['life_ladder'].agg(['mean', 'std']).reset_index()
                
                fig_global = go.Figure()
                
                # Ortalama çizgisi
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'],
                    mode='lines+markers',
                    name='Global Ortalama',
                    line=dict(color='#8dd3c7', width=3),
                    marker=dict(size=8)
                ))
                
                # Standart sapma aralığı
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'] + global_trend['std'],
                    mode='lines',
                    name='Standart Sapma',
                    line=dict(color='rgba(141, 211, 199, 0.2)', width=0),
                    showlegend=False
                ))
                
                fig_global.add_trace(go.Scatter(
                    x=global_trend['year'],
                    y=global_trend['mean'] - global_trend['std'],
                    mode='lines',
                    name='Standart Sapma',
                    line=dict(color='rgba(141, 211, 199, 0.2)', width=0),
                    fill='tonexty'
                ))
                
                fig_global.update_layout(
                    title="Global Mutluluk Trendi ve Değişkenlik",
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_global, use_container_width=True)
                
                # Bölgesel trend analizi
                st.markdown("### 🌍 Bölgesel Mutluluk Trendleri")
                
                # Bölgelere göre yıllık ortalamalar
                regional_trend = df.groupby(['year', 'regional_indicator'])['life_ladder'].mean().reset_index()
                
                fig_regional = go.Figure()
                
                for region in regional_trend['regional_indicator'].unique():
                    region_data = regional_trend[regional_trend['regional_indicator'] == region]
                    fig_regional.add_trace(go.Scatter(
                        x=region_data['year'],
                        y=region_data['life_ladder'],
                        mode='lines+markers',
                        name=region,
                        marker=dict(size=6)
                    ))
                
                fig_regional.update_layout(
                    title="Bölgesel Mutluluk Trendleri",
                    xaxis_title="Yıl",
                    yaxis_title="Mutluluk Skoru",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig_regional, use_container_width=True)
                
                # Trend analizi içgörüleri
                st.markdown("### 🔍 Trend Analizi İçgörüleri")
                
                # Global trend istatistikleri
                total_change = global_trend['mean'].iloc[-1] - global_trend['mean'].iloc[0]
                avg_change = total_change / (len(global_trend) - 1)
                volatility = global_trend['std'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Toplam Değişim",
                        f"{total_change:+.2f}",
                        "2005'ten günümüze"
                    )
                
                with col2:
                    st.metric(
                        "Yıllık Ortalama Değişim",
                        f"{avg_change:+.2f}",
                        "Her yıl için"
                    )
                
                with col3:
                    st.metric(
                        "Ortalama Değişkenlik",
                        f"{volatility:.2f}",
                        "Standart sapma"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="section-title">📊 Faktör Analizi Sekmesi</h3>', unsafe_allow_html=True)
                
                st.markdown("### 1. Mutluluk ile Faktörler Arasındaki Korelasyon (Heatmap)")
                
                # Faktör listesi
                factors = ['life_ladder', 'gdp_per_capita', 'social_support', 
                          'freedom_to_make_life_choices', 'internet_users_percent',
                          'life_expectancy']
                
                # Faktör isimleri
                factor_names = {
                    'life_ladder': 'Mutluluk',
                    'gdp_per_capita': 'GDP',
                    'social_support': 'Sosyal Destek',
                    'freedom_to_make_life_choices': 'Özgürlük',
                    'internet_users_percent': 'İnternet Kullanımı',
                    'life_expectancy': 'Yaşam Beklentisi'
                }
                
                # Korelasyon matrisi
                corr_matrix = df[factors].corr()
                
                # Heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=[factor_names[f] for f in factors],
                    y=[factor_names[f] for f in factors],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig_corr.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    margin=dict(l=50, r=50, t=30, b=50)
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)

                # 2. Faktörlerin Etkisi (Scatter Plots)
                st.markdown("### 2. Faktörlerin Etkisi (Scatter Plots)")

                # Scatter plot için faktörler
                scatter_factors = ['gdp_per_capita', 'internet_users_percent', 'freedom_to_make_life_choices']
                
                for factor in scatter_factors:
                    # Önce scatter plot'u oluştur (trend çizgisi olmadan)
                    fig_scatter = px.scatter(
                        df,
                        x=factor,
                        y='life_ladder',
                        title=f"{factor_names[factor]} vs Mutluluk",
                        labels={
                            factor: factor_names[factor],
                            'life_ladder': 'Mutluluk Skoru'
                        },
                        template="plotly_dark"
                    )
                    
                    # Scatter noktalarının stilini ayarla
                    fig_scatter.data[0].marker.update(
                        color='#FFA500',  # Turuncu renk
                        size=6,
                        opacity=0.6,
                        line=dict(color='#ffffff', width=1)
                    )
                    
                    # Trend çizgisini ayrı bir trace olarak ekle
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df[factor], df['life_ladder'])
                    line_x = np.array([df[factor].min(), df[factor].max()])
                    line_y = slope * line_x + intercept
                    
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode='lines',
                            name='Trend',
                            line=dict(color='#ff0000', width=5)  # Koyu kırmızı ve kalın çizgi
                        )
                    )
                    
                    fig_scatter.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        margin=dict(l=0, r=0, t=50, b=0)
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # 3. Gelir Seviyesine Göre Mutluluk Dağılımı (Boxplot)
                st.markdown("### 3. Gelir Seviyesine Göre Mutluluk Dağılımı (Boxplot)")
                
                # GDP'ye göre ülkeleri kategorilere ayır
                df['income_level'] = pd.qcut(df['gdp_per_capita'], 
                                          q=3, 
                                          labels=['Düşük Gelir', 'Orta Gelir', 'Yüksek Gelir'])
                
                fig_box = px.box(
                    df,
                    x='income_level',
                    y='life_ladder',
                    title='Gelir Seviyesine Göre Mutluluk Dağılımı',
                    labels={
                        'income_level': 'Gelir Seviyesi',
                        'life_ladder': 'Mutluluk Skoru'
                    },
                    template="plotly_dark"
                )
                
                fig_box.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state.current_page == 'Soru-Cevap':
            st.markdown('<h1 class="dashboard-title">Yapay Zeka ile Sohbet</h1>', unsafe_allow_html=True)
            
            # Soru-cevap bölümü - Ortalanmış
            qa_container = st.container()
            with qa_container:
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                question = st.text_area(
                    "",
                    placeholder="Sorunuzu buraya yazın...",
                    key="question_input",
                    height=100
                )

                if st.button("Gönder", key="submit_button", use_container_width=True):
                    if question:
                        with st.spinner("Yanıt hazırlanıyor..."):
                            from llm_agents import MultiAgentSystem
                            multi_agent = MultiAgentSystem(df)
                            answer = multi_agent.get_answer(question)
                            if answer:
                                st.markdown("---")
                                st.write(answer)
                            else:
                                st.error("Yanıt alınamadı!")
                st.markdown('</div>', unsafe_allow_html=True)

        # Container'ları kapat
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.error("Lütfen sayfayı yenileyin veya daha sonra tekrar deneyin.")

if __name__ == "__main__":
    main() 