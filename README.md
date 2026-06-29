# Goruntu-Isleme-Teknikleri-ile-Plaka-Tespiti
Görüntü işleme teknikleri kullanılarak plaka tespiti yapılmaktadır.
# License Plate Detection with Computer Vision

Bu projede görüntü işleme teknikleri ve OCR kullanılarak araç plakalarının tespit edilmesi amaçlanmıştır. Sistem, görüntü ön işleme adımları ile plaka bölgesini belirler ve OCR ile plaka metnini okuyarak doğrulama yapar.

## Kullanılan Teknolojiler

* Python
* OpenCV
* NumPy
* EasyOCR
* Matplotlib

## Uygulanan Yöntemler

* Gri tonlama (Grayscale)
* Gürültü azaltma (Bilateral Filter)
* Gaussian Blur
* Canny Edge Detection
* Kontur Analizi
* Plaka format doğrulama (Regex + il kodu kontrolü)
* OCR ile metin tanıma

## Çalışma Mantığı

1. Görüntü okunur.
2. Ön işleme adımları uygulanır.
3. Kenarlar tespit edilir.
4. Plaka olabilecek dikdörtgen bölgeler bulunur.
5. En uygun aday bölge seçilir.
6. OCR ile plaka okunur.
7. Türkiye plaka formatına göre doğrulama yapılır.
8. Sonuç görselleştirilir.

## Özellikler

* Türk plaka formatına uygun doğrulama
* İl kodu kontrolü (01-81)
* Ara işlem adımlarının görselleştirilmesi
* Geçersiz plaka tespiti
* OCR destekli karakter okuma

## Kullanım

```bash
pip install opencv-python numpy easyocr matplotlib
python plaka_tespit.py
```

## Örnek Çıktı

Sistem plaka bölgesini tespit eder, metni okur ve sonucu görsel üzerinde gösterir.

## Amaç

Gerçek zamanlı plaka tanıma sistemleri için temel bir görüntü işleme pipeline’ı geliştirmek.
