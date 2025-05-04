
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CSV dosyasını oku
df = pd.read_csv("turkce_sarkilar_ornek.csv")

# Şarkı sözlerini temizleme fonksiyonu
def temizle(soz):
    soz = soz.lower()
    soz = re.sub(r'\[.*?\]', '', soz)  # [Nakarat] gibi kısımları temizle
    soz = re.sub(r'\n', ' ', soz)       # Satır sonlarını boşluk yap
    soz = re.sub(r'[^a-zçğıöşü\s]', '', soz)  # Türkçe karakterler dışında her şeyi sil
    return soz.strip()

# Temizlenmiş sözleri ekle
df["clean_lyrics"] = df["lyrics"].apply(temizle)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_lyrics"])

# Kullanıcıdan şarkı seçmesini iste
print("Şarkı Listesi:")
for i, row in df.iterrows():
    print(f"{i}: {row['title']} - {row['artist']}")

secim = int(input("\nBeğendiğiniz şarkının numarasını girin: "))

# Seçilen şarkıya benzer diğer şarkıları bul
cosine_similarities = cosine_similarity(tfidf_matrix[secim], tfidf_matrix).flatten()
similar_indices = cosine_similarities.argsort()[::-1][1:4]

print(f"\n'{df.iloc[secim]['title']}' şarkısına en benzer öneriler:")
for i in similar_indices:
    print("-", df.iloc[i]["title"], "-", df.iloc[i]["artist"])
python --version

