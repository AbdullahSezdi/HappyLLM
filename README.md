# WellGPT

Mutluluk ve well-being analizleri için LLM destekli veri analiz platformu. Bu platform, ülkelerin mutluluk ve refah düzeylerini analiz eder, karşılaştırır ve içgörüler sunar.

## Özellikler

- 🤖 Çok Ajanlı LLM Sistemi (Multi-Agent)
  - Veri Analiz Ajanı
  - Nedensel Analiz Ajanı
  - Genel Soru-Cevap Ajanı
- 📊 Detaylı Veri Analizleri
- 📈 İnteraktif Görselleştirmeler
- 💬 Doğal Dil Sorgu Desteği

## Kurulum

1. Repoyu klonlayın:
```bash
git clone [repo-url]
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. .env dosyasını oluşturun:
```bash
GOOGLE_API_KEY=your_api_key_here
```

4. Uygulamayı çalıştırın:
```bash
streamlit run ana_script.py
```

## Kullanım Örnekleri

1. Ülke Analizleri:
   - "Türkiye nasıl bir ülke?"
   - "Finlandiya neden bu kadar mutlu?"

2. Karşılaştırmalı Analizler:
   - "Türkiye ve Almanya'yı karşılaştır"
   - "G20 ülkelerinin mutluluk durumu nasıl?"

3. Trend Analizleri:
   - "Son 5 yılda Türkiye'nin mutluluğu nasıl değişti?"
   - "Hangi ülkenin mutluluğu en çok arttı?"

## Gereksinimler

- Python 3.8+
- Google API Key
- İnternet bağlantısı

## Veri Seti

Proje, kapsamlı bir well-being veri seti kullanmaktadır:
- Mutluluk skorları
- Ekonomik göstergeler
- Sosyal göstergeler
- Demografik veriler
- Teknolojik göstergeler



